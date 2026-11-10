package server

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
)

func discoveryServer(t *testing.T, openaiStatus *int, pattern []string) *Server {
	t.Helper()
	codex := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, upstreamCatalog)
	}))
	t.Cleanup(codex.Close)
	market := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" || r.Header.Get("Authorization") != "Bearer k" {
			w.WriteHeader(401)
			return
		}
		w.WriteHeader(*openaiStatus)
		_, _ = io.WriteString(w, `{"data":[{"id":"gpt-6-luna"},{"id":"gpt-6.1-sol"},{"id":"other-model"}]}`)
	}))
	t.Cleanup(market.Close)
	creds := writeCodexCreds(t, t.TempDir())
	cfg := &config.Config{
		Listen: "127.0.0.1:0",
		Providers: map[string]config.Provider{
			"codex": {Type: "codex", BaseURL: codex.URL, Discover: true,
				Auth: &config.Auth{Type: "oauth_chatgpt", File: creds, Issuer: "http://unused"}},
			"asale": {Type: "openai", BaseURL: market.URL + "/v1", APIKey: "k",
				Discover: true, DiscoverPrefix: "asale-", DiscoverOnly: []string{"gpt-*", "other-*"}},
		},
		Models: map[string]config.Model{
			"gpt-6": {Provider: "codex", UpstreamModel: "gpt-6-astra"},
		},
		CodexCatalog: &config.CodexCatalog{Provider: "codex", Models: pattern},
	}
	s, err := New(cfg, discardLogger())
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func TestDiscovery_RoutesListedIDs(t *testing.T) {
	status := 200
	s := discoveryServer(t, &status, []string{"gpt-6"})
	if _, err := s.resolver.Resolve("asale-other-model"); err == nil {
		t.Fatal("resolved before discovery")
	}
	s.DiscoverOnce(context.Background())

	res, err := s.resolver.Resolve("asale-gpt-6-luna")
	if err != nil || res.ProviderName != "asale" || res.UpstreamModel != "gpt-6-luna" {
		t.Fatalf("asale-gpt-6-luna = %+v, %v", res, err)
	}
	if res, err := s.resolver.Resolve("gpt-6.1-sol"); err != nil || res.ProviderName != "codex" {
		t.Fatalf("codex gpt-6.1-sol = %+v, %v", res, err)
	}
	// visibility "hide" upstream entries are not discovered.
	if _, err := s.resolver.Resolve("gpt-6-astra"); err == nil {
		t.Error("hidden codex model was discovered")
	}
	// The declared name still wins.
	if res, _ := s.resolver.Resolve("gpt-6"); res.UpstreamModel != "gpt-6-astra" {
		t.Errorf("declared gpt-6 shadowed: %+v", res)
	}

	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if !strings.Contains(rec.Body.String(), `"asale-other-model"`) {
		t.Errorf("/v1/models misses discovered ids: %s", rec.Body.String())
	}
}

func TestDiscovery_OnlyFilter(t *testing.T) {
	status := 200
	s := discoveryServer(t, &status, []string{"gpt-6"})
	p := s.cfg.Providers["asale"]
	p.DiscoverOnly = []string{"gpt-6.1*"}
	s.cfg.Providers["asale"] = p
	s.DiscoverOnce(context.Background())
	if _, err := s.resolver.Resolve("asale-gpt-6.1-sol"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.resolver.Resolve("asale-gpt-6-luna"); err == nil {
		t.Error("discover_only did not filter")
	}
}

func TestDiscovery_FailureKeepsLastList(t *testing.T) {
	status := 200
	s := discoveryServer(t, &status, []string{"gpt-6"})
	s.DiscoverOnce(context.Background())
	status = 503
	s.DiscoverOnce(context.Background())
	if _, err := s.resolver.Resolve("asale-other-model"); err != nil {
		t.Fatalf("lost discovered model after a failed poll: %v", err)
	}
}

func TestCatalogIDs_Patterns(t *testing.T) {
	status := 200
	s := discoveryServer(t, &status, []string{"gpt-6*", "asale-gpt-*", "@asale", "!*other*"})
	s.DiscoverOnce(context.Background())
	got := strings.Join(s.catalogIDs(), ",")
	// gpt-6 (declared) first; discovered codex gpt-6-luna and gpt-6.1-sol
	// next; asale ids; `@asale` adds nothing new except other-model, which
	// `!*other*` then drops.
	want := "gpt-6,gpt-6-luna,gpt-6.1-sol,asale-gpt-6-luna,asale-gpt-6.1-sol"
	if got != want {
		t.Fatalf("catalogIDs = %s\nwant          %s", got, want)
	}
}

func TestCatalogIDs_DedupesSameUpstream(t *testing.T) {
	status := 200
	s := discoveryServer(t, &status, []string{"gpt-6", "gpt-6*"})
	s.cfg.Models["gpt-6-luna-alias"] = config.Model{Provider: "codex", UpstreamModel: "gpt-6-luna"}
	s.DiscoverOnce(context.Background())
	got := strings.Join(s.catalogIDs(), ",")
	if got != "gpt-6,gpt-6-luna-alias,gpt-6.1-sol" {
		t.Fatalf("catalogIDs = %s", got)
	}
}

func TestGlobMatch(t *testing.T) {
	for _, c := range []struct {
		p, n string
		ok   bool
	}{
		{"gpt-6*", "gpt-6.1-sol", true},
		{"gpt-6*", "gpt-5.6", false},
		{"*embed*", "qwen3-embedding:8b", true},
		{"*", "anything", true},
		{"a*b*c", "aXbYc", true},
		{"a*b*c", "aXbYcZ", false},
		{"exact", "exact", true},
		{"exact", "exactly", false},
	} {
		if globMatch(c.p, c.n) != c.ok {
			t.Errorf("globMatch(%q, %q) != %v", c.p, c.n, c.ok)
		}
	}
}
