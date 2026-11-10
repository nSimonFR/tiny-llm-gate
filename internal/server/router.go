package server

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"

	"github.com/nSimonFR/tiny-llm-gate/internal/auth"
	"github.com/nSimonFR/tiny-llm-gate/internal/config"
	"github.com/nSimonFR/tiny-llm-gate/internal/resolve"
	"github.com/nSimonFR/tiny-llm-gate/internal/router"
)

// servedModelHeader names the model a router picked. LiteLLM's header name,
// which Hermes reads for its served_model footer.
const servedModelHeader = "x-litellm-model-id"

type routeKey struct{}

func withRoute(ctx context.Context, d *router.Decision) context.Context {
	return context.WithValue(ctx, routeKey{}, d)
}

func routeOf(ctx context.Context) *router.Decision {
	d, _ := ctx.Value(routeKey{}).(*router.Decision)
	return d
}

// buildRouters turns config into routers. A classifier credential that fails
// to load disables only that router's classifier: its turns go to on_error.
func buildRouters(cfg *config.Config, client *http.Client, logger *slog.Logger) map[string]*router.Router {
	out := make(map[string]*router.Router, len(cfg.Routers))
	for name, rc := range cfg.Routers {
		tierIdx := make(map[string]int, len(rc.Tiers))
		for i, t := range rc.Tiers {
			tierIdx[t] = i
		}
		qs := make([]router.Question, 0, len(rc.Questions))
		for qn, q := range rc.Questions {
			qs = append(qs, router.Question{Name: qn, Instructions: q.Instructions, Escalate: tierIdx[q.EscalateTo], Threshold: q.Threshold})
		}
		c := router.Config{
			Name: name, Endpoint: rc.Classifier.Endpoint, Model: rc.Classifier.Model,
			Timeout: rc.Classifier.Timeout(), Tiers: rc.Tiers, OnError: rc.ErrorTier(),
			Questions: router.SortedQuestions(qs), MaxState: rc.MaxStateChars,
			TTL: rc.TTL(), Entries: rc.CacheEntries,
		}
		if c.Endpoint == "" {
			c.Endpoint = config.DefaultRouterEndpoint
		}
		if c.MaxState == 0 {
			c.MaxState = config.DefaultRouterMaxStateChars
		}
		if c.Entries == 0 {
			c.Entries = config.DefaultRouterCacheEntries
		}
		if a := rc.Classifier.Auth; a != nil {
			authn, err := auth.Build(authConfigFromConfig(a, logger))
			if err != nil {
				logger.Error("router classifier auth init failed; every turn goes to on_error",
					"router", name, "on_error", c.OnError, "err", err)
				c.Disabled = fmt.Errorf("classifier auth: %w", err)
			}
			c.Auth = authn
		}
		out[name] = router.New(c, client)
	}
	return out
}

// applyRouter resolves a router model to the target it picks for this turn.
// A non-router model passes through unchanged. On a router, the returned
// request carries the decision for logging and for modelAllowed; ok=false
// means a response was already written.
func (s *Server) applyRouter(w http.ResponseWriter, r *http.Request, frontend, model string, turn func() (router.Turn, error)) (string, *http.Request, bool) {
	name, isRouter := s.resolver.Router(model)
	if !isRouter {
		return model, r, true
	}
	rt := s.routers[name]
	if ck := clientOf(r.Context()); !ck.allows(model, name) {
		writeJSONError(w, http.StatusForbidden, fmt.Sprintf("key %q may not use model %q", ck.name, model))
		return "", r, false
	}
	var d router.Decision
	if t, err := turn(); err != nil {
		d = rt.Failed(fmt.Errorf("read conversation: %w", err))
	} else {
		d = rt.Decide(r.Context(), t)
	}
	if d.Err != nil && !d.CacheHit {
		s.logger.Warn("router classifier failed; using on_error",
			"request_id", requestID(r.Context()), "frontend", frontend,
			"router", name, "on_error", d.Target, "err", d.Err)
	}
	w.Header().Set(servedModelHeader, d.Target)
	return d.Target, r.WithContext(withRoute(r.Context(), &d)), true
}

// routeAttrs are the log fields of a routed request; nil when not routed.
func routeAttrs(r *http.Request) []any {
	d := routeOf(r.Context())
	if d == nil {
		return nil
	}
	attrs := []any{"router", d.Router, "routed_to", d.Target,
		"route_cache_hit", d.CacheHit, "classifier_ms", d.ClassifierMS}
	if d.Signals != nil {
		attrs = append(attrs, "route_signals", d.Signals)
	}
	if d.Err != nil {
		attrs = append(attrs, "route_err", d.Err.Error())
	}
	return attrs
}

// listingResolve resolves an id for model listings. A router is described by
// its strongest tier, renamed to the router.
func (s *Server) listingResolve(id string) (*resolve.Resolved, error) {
	name, ok := s.resolver.Router(id)
	if !ok {
		return s.resolver.Resolve(id)
	}
	tiers := s.cfg.Routers[name].Tiers
	var out resolve.Resolved
	if res, err := s.resolver.Resolve(tiers[len(tiers)-1]); err == nil {
		out = *res
	}
	out.ModelName, out.UpstreamModel, out.ProviderName, out.Slug = name, name, "router", "router/"+name
	out.Fallback = nil
	return &out, nil
}
