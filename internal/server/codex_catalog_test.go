package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
)

const upstreamCatalog = `{"models":[
 {"slug":"gpt-6-luna","display_name":"GPT-6 Luna","visibility":"list","priority":3,"base_instructions":"luna","context_window":272000,"upgrade":{"model":"x"}},
 {"slug":"gpt-6.1-sol","display_name":"GPT-6.1 Sol","visibility":"list","priority":1,"base_instructions":"sol","context_window":272000},
 {"slug":"gpt-6-astra","display_name":"GPT-6 Astra","visibility":"hide","priority":2,"base_instructions":"astra"}
]}`

func catalogServer(t *testing.T, status *int, hits *int, gotQuery *string) *Server {
	t.Helper()
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		*hits++
		*gotQuery = r.URL.RawQuery
		if r.URL.Path != "/models" || r.Header.Get("ChatGPT-Account-Id") != "acct_test" {
			w.WriteHeader(400)
			return
		}
		w.WriteHeader(*status)
		_, _ = io.WriteString(w, upstreamCatalog)
	}))
	t.Cleanup(backend.Close)
	creds := writeCodexCreds(t, t.TempDir())
	cfg := &config.Config{
		Listen: "127.0.0.1:0",
		Providers: map[string]config.Provider{
			"codex": {Type: "codex", BaseURL: backend.URL,
				Auth: &config.Auth{Type: "oauth_chatgpt", File: creds, Issuer: "http://unused"}},
			"asale":  {Type: "openai", BaseURL: "http://asale.invalid/v1"},
			"ollama": {Type: "openai", BaseURL: "http://ollama.invalid/v1"},
		},
		Models: map[string]config.Model{
			"gpt-6-luna":       {Provider: "codex", UpstreamModel: "gpt-6-luna"},
			"gpt-6":            {Provider: "codex", UpstreamModel: "gpt-6-astra"},
			"asale-gpt-6-luna": {Provider: "asale", UpstreamModel: "gpt-6-luna"},
			"gemma4:26b": {Provider: "ollama", UpstreamModel: "gemma4:26b",
				Info: &config.ModelInfo{Name: "Gemma 4 26B", ContextLength: 131072}},
			"unlisted": {Provider: "codex", UpstreamModel: "gpt-6.1-sol"},
		},
		CodexCatalog: &config.CodexCatalog{
			Provider: "codex",
			Models:   []string{"gpt-6-luna", "gpt-6", "asale-gpt-6-luna", "gemma4:26b"},
			Template: "gpt-6.1-sol",
		},
	}
	s, err := New(cfg, discardLogger())
	if err != nil {
		t.Fatal(err)
	}
	return s
}

func getCatalog(t *testing.T, s *Server, query string) (int, []map[string]any) {
	t.Helper()
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/codex/models"+query, nil))
	var body struct {
		Models []map[string]any `json:"models"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &body)
	return rec.Code, body.Models
}

func TestCodexCatalog_FiltersRenamesAndClones(t *testing.T) {
	status, hits, q := 200, 0, ""
	s := catalogServer(t, &status, &hits, &q)
	code, models := getCatalog(t, s, "?client_version=0.164.0")
	if code != 200 {
		t.Fatalf("status %d", code)
	}
	if q != "client_version=0.164.0" {
		t.Errorf("upstream query = %q", q)
	}
	var slugs []string
	for _, m := range models {
		slugs = append(slugs, m["slug"].(string))
	}
	if got := strings.Join(slugs, ","); got != "gpt-6-luna,gpt-6,asale-gpt-6-luna,gemma4:26b" {
		t.Fatalf("slugs = %s", got)
	}
	luna, astra, asale, gemma := models[0], models[1], models[2], models[3]
	if luna["base_instructions"] != "luna" || luna["upgrade"] == nil {
		t.Errorf("exact match must keep upstream fields: %v", luna)
	}
	if astra["display_name"] != "GPT-6 Astra" || astra["visibility"] != "list" || astra["priority"] != 1.0 {
		t.Errorf("astra = %v", astra)
	}
	if asale["base_instructions"] != "luna" || asale["display_name"] != "asale-gpt-6-luna" || asale["upgrade"] != nil {
		t.Errorf("same upstream id on another provider: %v", asale)
	}
	if gemma["base_instructions"] != "sol" || gemma["display_name"] != "Gemma 4 26B" || gemma["context_window"] != 131072.0 {
		t.Errorf("gemma = %v", gemma)
	}
}

func TestCodexCatalog_CachesAndServesStale(t *testing.T) {
	status, hits, q := 200, 0, ""
	s := catalogServer(t, &status, &hits, &q)
	getCatalog(t, s, "")
	getCatalog(t, s, "")
	if hits != 1 {
		t.Fatalf("hits = %d, want 1 (cached)", hits)
	}
	if q != "client_version=1.0.0" {
		t.Errorf("default client_version: %q", q)
	}
	status = 500
	if code, models := getCatalog(t, s, "?client_version=9.9.9"); code != 200 || len(models) != 4 {
		t.Fatalf("stale fallback: code %d, %d models", code, len(models))
	}
}

func TestCodexCatalog_UpstreamDownNoCache(t *testing.T) {
	status, hits, q := 503, 0, ""
	s := catalogServer(t, &status, &hits, &q)
	if code, _ := getCatalog(t, s, ""); code != http.StatusBadGateway {
		t.Fatalf("code = %d", code)
	}
}

func TestCodexCatalog_ConfigValidation(t *testing.T) {
	base := `
providers:
  codex: {type: codex, base_url: http://x, auth: {type: oauth_chatgpt, file: /tmp/c}}
  o: {type: openai, base_url: http://y}
models:
  a: {provider: codex, upstream_model: a}
`
	for name, tc := range map[string]struct{ extra, want string }{
		"ok":           {"codex_catalog: {provider: codex, models: [a]}", ""},
		"non-codex":    {"codex_catalog: {provider: o, models: [a]}", "must be a codex provider"},
		"unknown":      {"codex_catalog: {provider: codex, models: [zz]}", "not a model or alias"},
		"empty models": {"codex_catalog: {provider: codex}", "models is empty"},
	} {
		_, err := config.Parse([]byte(base + tc.extra))
		if (err == nil) != (tc.want == "") || (err != nil && !strings.Contains(err.Error(), tc.want)) {
			t.Errorf("%s: err = %v, want %q", name, err, tc.want)
		}
	}
}
