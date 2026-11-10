package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// codexCatalogTTL matches Codex CLI's own models_cache.json TTL.
const codexCatalogTTL = 5 * time.Minute

type codexCatalogCache struct {
	mu      sync.Mutex
	version string
	fetched time.Time
	models  []map[string]json.RawMessage
}

// handleCodexCatalog serves GET /codex/models: the configured models, each
// cloned from its upstream ChatGPT catalog entry and renamed to the gate id.
// A failed upstream fetch serves the last good catalog when there is one.
func (s *Server) handleCodexCatalog(w http.ResponseWriter, r *http.Request) {
	cc := s.cfg.CodexCatalog
	version := r.URL.Query().Get("client_version")
	if version == "" {
		version = cc.ClientVersion
	}
	if version == "" {
		version = "1.0.0"
	}
	upstream, err := s.upstreamCodexCatalog(r, version)
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, err.Error())
		return
	}

	bySlug := make(map[string]map[string]json.RawMessage, len(upstream))
	for _, m := range upstream {
		var slug string
		_ = json.Unmarshal(m["slug"], &slug)
		bySlug[slug] = m
	}
	template := upstream[0]
	if t, ok := bySlug[cc.Template]; ok {
		template = t
	}

	out := make([]map[string]json.RawMessage, 0, len(cc.Models))
	for i, id := range cc.Models {
		res, err := s.resolver.Resolve(id)
		if err != nil {
			continue
		}
		// Same upstream id on another provider (asale) keeps its instructions
		// but not the plan's name or upgrade prompt.
		src, exact := bySlug[res.UpstreamModel]
		if !exact {
			src = template
		}
		native := exact && res.ProviderName == cc.Provider
		m := make(map[string]json.RawMessage, len(src))
		for k, v := range src {
			m[k] = v
		}
		m["slug"] = rawJSON(id)
		m["visibility"] = rawJSON("list")
		m["priority"] = rawJSON(i)
		switch {
		case res.Info != nil && res.Info.Name != "":
			m["display_name"] = rawJSON(res.Info.Name)
		case !native:
			m["display_name"] = rawJSON(id)
		}
		if !native {
			m["description"] = rawJSON(fmt.Sprintf("%s via %s", res.UpstreamModel, res.ProviderName))
			m["upgrade"] = json.RawMessage("null")
			m["availability_nux"] = json.RawMessage("null")
		}
		if res.Info != nil && res.Info.ContextLength > 0 {
			m["context_window"] = rawJSON(res.Info.ContextLength)
		}
		out = append(out, m)
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"models": out})
}

func (s *Server) upstreamCodexCatalog(r *http.Request, version string) ([]map[string]json.RawMessage, error) {
	c := &s.codexCatalog
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.models != nil && c.version == version && time.Since(c.fetched) < codexCatalogTTL {
		return c.models, nil
	}
	models, err := s.fetchCodexCatalog(r, version)
	if err != nil {
		if c.models != nil {
			s.logger.Warn("codex catalog fetch failed; serving cached", "err", err)
			return c.models, nil
		}
		return nil, err
	}
	c.models, c.version, c.fetched = models, version, time.Now()
	return models, nil
}

func (s *Server) fetchCodexCatalog(r *http.Request, version string) ([]map[string]json.RawMessage, error) {
	name := s.cfg.CodexCatalog.Provider
	if reason, ok := s.disabled[name]; ok {
		return nil, fmt.Errorf("codex catalog: provider %q disabled: %s", name, reason)
	}
	p := s.cfg.Providers[name]
	u := strings.TrimRight(p.BaseURL, "/") + "/models?client_version=" + url.QueryEscape(version)
	req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("originator", codexOriginator)
	req.Header.Set("User-Agent", codexUserAgent)
	req.Header.Set("x-codex-installation-id", codexInstallationID)
	if authz, ok := s.auths[name]; ok {
		if err := authz.Apply(r.Context(), req); err != nil {
			return nil, fmt.Errorf("codex catalog auth: %w", err)
		}
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("codex catalog: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, fmt.Errorf("codex catalog: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("codex catalog: upstream status %d", resp.StatusCode)
	}
	var parsed struct {
		Models []map[string]json.RawMessage `json:"models"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, fmt.Errorf("codex catalog: %w", err)
	}
	if len(parsed.Models) == 0 {
		return nil, fmt.Errorf("codex catalog: upstream listed no models for client_version %s", version)
	}
	return parsed.Models, nil
}

func rawJSON(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}
