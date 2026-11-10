package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

// RunDiscovery polls every provider with `discover: true` now and then every
// DiscoverInterval until ctx ends. A failed poll keeps that provider's last list.
func (s *Server) RunDiscovery(ctx context.Context) {
	if !s.anyDiscover() {
		return
	}
	t := time.NewTicker(s.cfg.DiscoverEvery())
	defer t.Stop()
	for {
		s.DiscoverOnce(ctx)
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

func (s *Server) anyDiscover() bool {
	for _, p := range s.cfg.Providers {
		if p.Discover {
			return true
		}
	}
	return false
}

// DiscoverOnce polls each discovering provider once.
func (s *Server) DiscoverOnce(ctx context.Context) {
	names := make([]string, 0, len(s.cfg.Providers))
	for name, p := range s.cfg.Providers {
		if p.Discover {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	for _, name := range names {
		pctx, cancel := context.WithTimeout(ctx, 15*time.Second)
		ids, err := s.listUpstreamModels(pctx, name)
		cancel()
		if err != nil {
			s.logger.Warn("model discovery failed; keeping last list", "provider", name, "err", err)
			continue
		}
		if only := s.cfg.Providers[name].DiscoverOnly; len(only) > 0 {
			kept := ids[:0]
			for _, id := range ids {
				for _, pat := range only {
					if globMatch(pat, id) {
						kept = append(kept, id)
						break
					}
				}
			}
			ids = kept
		}
		s.resolver.SetDiscovered(name, ids)
		s.logger.Info("models discovered", "provider", name, "count", len(ids))
	}
}

func (s *Server) listUpstreamModels(ctx context.Context, name string) ([]string, error) {
	if reason, ok := s.disabled[name]; ok {
		return nil, fmt.Errorf("provider disabled: %s", reason)
	}
	p := s.cfg.Providers[name]
	if p.Type == "codex" {
		models, err := s.fetchCodexCatalog(ctx, name, "1.0.0")
		if err != nil {
			return nil, err
		}
		var ids []string
		for _, m := range models {
			var slug, vis string
			_ = json.Unmarshal(m["slug"], &slug)
			_ = json.Unmarshal(m["visibility"], &vis)
			if slug != "" && vis == "list" {
				ids = append(ids, slug)
			}
		}
		return ids, nil
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(p.BaseURL, "/")+"/models", nil)
	if err != nil {
		return nil, err
	}
	if authz, ok := s.auths[name]; ok {
		if err := authz.Apply(ctx, req); err != nil {
			return nil, fmt.Errorf("auth: %w", err)
		}
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("upstream status %d", resp.StatusCode)
	}
	var parsed struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(parsed.Data))
	for _, d := range parsed.Data {
		if d.ID != "" {
			ids = append(ids, d.ID)
		}
	}
	return ids, nil
}
