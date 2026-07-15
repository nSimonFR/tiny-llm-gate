// Package server wires HTTP handlers to the resolver and manages the upstream
// HTTP client. All request/response plumbing lives here.
package server

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"sync/atomic"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/auth"
	"github.com/nSimonFR/tiny-llm-gate/internal/config"
	"github.com/nSimonFR/tiny-llm-gate/internal/mcp"
	"github.com/nSimonFR/tiny-llm-gate/internal/resolve"
)

// Server is the public type exposed to main.
type Server struct {
	cfg      *config.Config
	resolver *resolve.Resolver
	client   *http.Client
	logger   *slog.Logger
	// auths is a per-provider Authenticator, built once at startup.
	auths map[string]auth.Authenticator
	// disabled maps a provider name to the reason its authenticator failed to
	// build at startup (e.g. a missing/unreadable oauth_chatgpt credentials
	// file). Such a provider is NOT fatal to the gate: its hops fail with this
	// reason (and fall back), while every other provider + the Anthropic
	// passthrough keep serving.
	disabled map[string]string
	bridges  []*mcp.Bridge
	// anthropicAccounts is the pool of accounts applied to /v1/messages
	// requests forwarded to the Anthropic API, in priority order. Empty when
	// no auth is configured (unauthenticated passthrough).
	anthropicAccounts []*anthropicAccount
	// currentAnthropic is the index of the sticky "current" account. The gate
	// stays on one account until it 429s, then advances this and stays on
	// the new one — see pickAnthropicAccount / nextAnthropicAccount.
	currentAnthropic atomic.Int32
}

// New builds a Server. The *http.Client has generous timeouts for streaming
// LLM responses but tight idle-connection limits to keep memory bounded.
func New(cfg *config.Config, logger *slog.Logger) (*Server, error) {
	transport := &http.Transport{
		// Bounded pool keeps FD + memory tight.
		MaxIdleConns:          16,
		MaxIdleConnsPerHost:   4,
		IdleConnTimeout:       60 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		// Non-streaming completions (stream:false) send NO response headers
		// until the upstream finishes generating. A document/vision request
		// with a large max_tokens (e.g. Reactive Resume's PDF resume parse,
		// max_tokens 64000 to Anthropic) can legitimately take many minutes,
		// especially under shared-OAuth rate pressure, so a tight header
		// timeout aborts valid work. 600s gives generous headroom; DialContext
		// still fails dead connections in 10s, so this only extends the
		// header-wait phase (streaming requests get headers immediately).
		ResponseHeaderTimeout: 600 * time.Second,
		// Force HTTP/1.1 for predictable streaming and lower memory
		// overhead than HTTP/2's frame buffering.
		ForceAttemptHTTP2: false,
		DialContext: (&net.Dialer{
			Timeout:   10 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
	}
	auths, disabled, err := buildAuthenticators(cfg, logger)
	if err != nil {
		return nil, err
	}

	// Build MCP bridges. They share a separate HTTP client with longer
	// timeouts suited to MCP streaming responses.
	var bridges []*mcp.Bridge
	if len(cfg.MCPBridges) > 0 {
		mcpTransport := &http.Transport{
			MaxIdleConns:          8,
			MaxIdleConnsPerHost:   2,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ResponseHeaderTimeout: 120 * time.Second,
			ForceAttemptHTTP2:     false,
			DialContext: (&net.Dialer{
				Timeout:   10 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
		}
		mcpClient := &http.Client{Transport: mcpTransport}
		for name, bcfg := range cfg.MCPBridges {
			br, err := mcp.NewBridge(name, bcfg, mcpClient, logger)
			if err != nil {
				return nil, fmt.Errorf("mcp_bridge %q: %w", name, err)
			}
			bridges = append(bridges, br)
		}
	}

	// Build the Anthropic frontend account pool (applied to /v1/messages
	// requests forwarded upstream). EffectiveAccounts folds a legacy single
	// Auth field into a one-element slice, so this handles both shapes.
	var anthropicAccounts []*anthropicAccount
	if cfg.Anthropic != nil {
		for _, ac := range cfg.Anthropic.EffectiveAccounts() {
			authn, err := auth.Build(authConfigFromConfig(ac.Auth, logger))
			if err != nil {
				return nil, fmt.Errorf("anthropic account %q: auth: %w", ac.Name, err)
			}
			anthropicAccounts = append(anthropicAccounts, &anthropicAccount{name: ac.Name, auth: authn})
		}
	}

	return &Server{
		cfg:      cfg,
		resolver: resolve.New(cfg),
		// No overall Timeout — streaming responses can legitimately run
		// for minutes. Per-phase timeouts live on the Transport.
		client:            &http.Client{Transport: transport},
		logger:            logger,
		auths:             auths,
		disabled:          disabled,
		bridges:           bridges,
		anthropicAccounts: anthropicAccounts,
	}, nil
}

// buildAuthenticators constructs one auth.Authenticator per provider based on
// config. Providers without authentication get no entry — sendUpstream treats
// missing as "send without auth header".
func buildAuthenticators(cfg *config.Config, logger *slog.Logger) (map[string]auth.Authenticator, map[string]string, error) {
	out := make(map[string]auth.Authenticator, len(cfg.Providers))
	disabled := make(map[string]string)
	for name, p := range cfg.Providers {
		a := p.EffectiveAuth()
		if a == nil {
			continue
		}
		authn, err := auth.Build(authConfigFromConfig(a, logger))
		if err != nil {
			// Non-fatal: one provider's auth failing (e.g. a missing or
			// unreadable oauth_chatgpt credentials file) must NOT take down the
			// whole gate — and with it every other provider plus the Anthropic
			// passthrough. Log loudly and disable just this provider; hops
			// routed to it fail with this reason (and fall back), while the
			// rest of the gate keeps serving.
			logger.Error("provider auth init failed; provider disabled",
				"provider", name, "err", err)
			disabled[name] = err.Error()
			continue
		}
		if authn != nil {
			out[name] = authn
		}
	}
	return out, disabled, nil
}

// authConfigFromConfig converts a config.Auth to an auth.AuthConfig. The
// logger is used by the oauth_chatgpt strategy to log refresh events.
func authConfigFromConfig(a *config.Auth, logger *slog.Logger) *auth.AuthConfig {
	if a == nil {
		return nil
	}
	return &auth.AuthConfig{
		Type:      a.Type,
		Token:     a.Token,
		TokenFile: a.TokenFile,
		File:      a.File,
		Issuer:    a.Issuer,
		ClientID:  a.ClientID,
		Logger:    logger,
	}
}

// Handler returns the HTTP handler for this server.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// OpenAI frontend
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("POST /v1/embeddings", s.handleEmbeddings)
	mux.HandleFunc("GET /v1/models", s.handleModels)

	// Gemini frontend. The Gemini URL form is /v1beta/models/{model}:action
	// where `:action` is a suffix on the final path segment, not a separator
	// Go's ServeMux handles natively. We route by prefix and dispatch on the
	// action in a single handler.
	mux.HandleFunc("GET /v1beta/models", s.handleGeminiModels)
	mux.HandleFunc("POST /v1beta/models/", s.routeGemini)

	// Anthropic frontend (pass-through proxy to api.anthropic.com).
	// Registered only when configured so clients hit 404 for /v1/messages
	// on gates that don't have Anthropic support enabled.
	if s.cfg.Anthropic != nil {
		mux.HandleFunc("POST /v1/messages", s.handleAnthropicMessages)
	}

	// Health and readiness
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /ready", s.handleReady)

	// MCP bridges
	for _, br := range s.bridges {
		br.RegisterRoutes(mux)
	}

	return withRequestID(mux)
}

// routeGemini dispatches /v1beta/models/{model}:action to the right handler.
// Splitting on the last colon gives us the action suffix.
func (s *Server) routeGemini(w http.ResponseWriter, r *http.Request) {
	// Find last ':' in the path — the action separator Gemini uses.
	path := r.URL.Path
	colon := -1
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == ':' {
			colon = i
			break
		}
		if path[i] == '/' {
			break
		}
	}
	if colon < 0 {
		writeJSONError(w, http.StatusNotFound, "expected /v1beta/models/{model}:action")
		return
	}
	action := path[colon+1:]
	switch action {
	case "generateContent":
		s.handleGenerateContent(w, r)
	case "streamGenerateContent":
		s.handleStreamGenerateContent(w, r)
	case "embedContent":
		s.handleEmbedContent(w, r)
	case "batchEmbedContents":
		s.handleBatchEmbedContents(w, r)
	default:
		writeJSONError(w, http.StatusNotFound, "unknown Gemini action: "+action)
	}
}

// Shutdown gracefully closes idle upstream connections and MCP bridges.
func (s *Server) Shutdown(ctx context.Context) error {
	for _, br := range s.bridges {
		_ = br.Shutdown(ctx)
	}
	s.client.CloseIdleConnections()
	return nil
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok"}` + "\n"))
}

func (s *Server) handleReady(w http.ResponseWriter, _ *http.Request) {
	// Phase 1: readiness == config loaded. Upstream probes come later.
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ready"}` + "\n"))
}
