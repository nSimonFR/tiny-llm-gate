package server

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
)

// fakeJev answers every question "yes" (0.9) when the state mentions
// "debug", else "no" (0.1). status != 200 makes it fail.
type fakeJev struct {
	*httptest.Server
	hits   atomic.Int32
	status int
	last   atomic.Value // last request body
}

func newFakeJev(t *testing.T, status int) *fakeJev {
	t.Helper()
	j := &fakeJev{status: status}
	j.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		j.hits.Add(1)
		var in struct {
			Model     string                     `json:"model"`
			State     string                     `json:"state"`
			Questions map[string]json.RawMessage `json:"questions"`
		}
		b, _ := io.ReadAll(r.Body)
		j.last.Store(string(b))
		_ = json.Unmarshal(b, &in)
		if j.status != http.StatusOK {
			w.WriteHeader(j.status)
			return
		}
		p := 0.1
		if strings.Contains(in.State, "debug") {
			p = 0.9
		}
		answers := map[string]any{}
		for q := range in.Questions {
			answers[q] = map[string]any{"noul": p}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"answers": answers, "usage": map[string]any{"input_tokens": 42}})
	}))
	t.Cleanup(j.Close)
	return j
}

func routerConfig(upstreamURL, jevURL string) *config.Config {
	return &config.Config{
		Listen: "127.0.0.1:0",
		Providers: map[string]config.Provider{
			"up": {Type: "openai", BaseURL: upstreamURL + "/v1"},
		},
		Models: map[string]config.Model{
			"cheap":  {Provider: "up", UpstreamModel: "cheap-up"},
			"strong": {Provider: "up", UpstreamModel: "strong-up"},
		},
		Aliases: map[string]string{"openai/smart": "smart"},
		Routers: map[string]config.Router{
			"smart": {
				Classifier: config.RouterClassifier{Endpoint: jevURL, Model: "jev-test"},
				Tiers:      []string{"cheap", "strong"},
				Questions: map[string]config.RouterQuestion{
					"hard": {Instructions: "Is it hard?", EscalateTo: "strong", Threshold: 0.5},
				},
			},
		},
	}
}

func newRouterServer(t *testing.T, cfg *config.Config) http.Handler {
	t.Helper()
	s, err := New(cfg, discardLogger())
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	return s.Handler()
}

func chatBody(model, user string) string {
	b, _ := json.Marshal(map[string]any{"model": model, "messages": []map[string]any{
		{"role": "system", "content": "be brief"},
		{"role": "user", "content": user},
	}})
	return string(b)
}

func TestRouterChatEscalates(t *testing.T) {
	up := newRecordingUpstream(t, 200, "application/json", okChat)
	jev := newFakeJev(t, 200)
	h := newRouterServer(t, routerConfig(up.URL, jev.URL))

	for _, tc := range []struct{ user, wantUp, wantHeader string }{
		{"hi there", "cheap-up", "cheap"},
		{"please debug my backup", "strong-up", "strong"},
	} {
		rec := do(t, h, "POST", "/v1/chat/completions", chatBody("smart", tc.user), nil)
		if rec.Code != 200 {
			t.Fatalf("%q: status %d: %s", tc.user, rec.Code, rec.Body)
		}
		if got := up.last(t)["model"]; got != tc.wantUp {
			t.Errorf("%q: upstream model %v, want %s", tc.user, got, tc.wantUp)
		}
		if got := rec.Header().Get(servedModelHeader); got != tc.wantHeader {
			t.Errorf("%q: %s = %q, want %q", tc.user, servedModelHeader, got, tc.wantHeader)
		}
	}
	var sent struct {
		Model     string                     `json:"model"`
		Questions map[string]json.RawMessage `json:"questions"`
	}
	_ = json.Unmarshal([]byte(jev.last.Load().(string)), &sent)
	if sent.Model != "jev-test" || sent.Questions["hard"] == nil {
		t.Errorf("classifier request = %s", jev.last.Load())
	}
}

func TestRouterOncePerTurn(t *testing.T) {
	up := newRecordingUpstream(t, 200, "application/json", okChat)
	jev := newFakeJev(t, 200)
	h := newRouterServer(t, routerConfig(up.URL, jev.URL))

	step := func(extra ...map[string]any) {
		msgs := []map[string]any{{"role": "user", "content": "debug this"}}
		msgs = append(msgs, extra...)
		b, _ := json.Marshal(map[string]any{"model": "openai/smart", "messages": msgs})
		if rec := do(t, h, "POST", "/v1/chat/completions", string(b), nil); rec.Code != 200 {
			t.Fatalf("status %d: %s", rec.Code, rec.Body)
		}
	}
	step()
	step(map[string]any{"role": "assistant", "tool_calls": []any{}},
		map[string]any{"role": "tool", "tool_call_id": "1", "content": "logs"})
	if n := jev.hits.Load(); n != 1 {
		t.Fatalf("classifier called %d times for one turn, want 1", n)
	}
	step(map[string]any{"role": "assistant", "content": "done"},
		map[string]any{"role": "user", "content": "thanks"})
	if n := jev.hits.Load(); n != 2 {
		t.Fatalf("classifier called %d times after a new user message, want 2", n)
	}
}

func TestRouterClassifierDown(t *testing.T) {
	up := newRecordingUpstream(t, 200, "application/json", okChat)
	jev := newFakeJev(t, 500)
	h := newRouterServer(t, routerConfig(up.URL, jev.URL))

	rec := do(t, h, "POST", "/v1/chat/completions", chatBody("smart", "hi"), nil)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if got := up.last(t)["model"]; got != "strong-up" {
		t.Errorf("classifier down served %v, want the last tier", got)
	}
}

func TestRouterTargetFallback(t *testing.T) {
	bad := newMockUpstream(503, `{"error":"down"}`)
	defer bad.Close()
	good := newRecordingUpstream(t, 200, "application/json", okChat)
	jev := newFakeJev(t, 200)
	cfg := routerConfig(good.URL, jev.URL)
	cfg.Providers["bad"] = config.Provider{Type: "openai", BaseURL: bad.URL + "/v1"}
	cfg.Models["cheap"] = config.Model{Provider: "bad", UpstreamModel: "cheap-up", Fallback: []string{"strong"}}
	h := newRouterServer(t, cfg)

	rec := do(t, h, "POST", "/v1/chat/completions", chatBody("smart", "hi"), nil)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if got := good.last(t)["model"]; got != "strong-up" {
		t.Errorf("fallback served %v, want strong-up", got)
	}
}

func TestRouterOtherFrontends(t *testing.T) {
	up := newRecordingUpstream(t, 200, "application/json", okChat)
	jev := newFakeJev(t, 200)
	h := newRouterServer(t, routerConfig(up.URL, jev.URL))

	cases := []struct{ name, path, body string }{
		{"responses", "/v1/responses", `{"model":"smart","instructions":"x","input":[{"role":"user","content":[{"type":"input_text","text":"debug it"}]}]}`},
		{"gemini", "/v1beta/models/smart:generateContent", `{"contents":[{"role":"user","parts":[{"text":"debug it"}]}]}`},
		{"messages", "/v1/messages", `{"model":"smart","max_tokens":10,"messages":[{"role":"user","content":"debug it"}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := do(t, h, "POST", tc.path, tc.body, nil)
			if rec.Code != 200 {
				t.Fatalf("status %d: %s", rec.Code, rec.Body)
			}
			if got := up.last(t)["model"]; got != "strong-up" {
				t.Errorf("upstream model %v, want strong-up", got)
			}
			if got := rec.Header().Get(servedModelHeader); got != "strong" {
				t.Errorf("%s = %q", servedModelHeader, got)
			}
		})
	}
}

func TestRouterAnthropicRouteTarget(t *testing.T) {
	up := newRecordingUpstream(t, 200, "application/json", okChat)
	anth := newRecordingUpstream(t, 200, "application/json", `{"id":"m","type":"message","content":[]}`)
	jev := newFakeJev(t, 200)
	cfg := routerConfig(up.URL, jev.URL)
	cfg.Anthropic = &config.Anthropic{
		Upstream: anth.URL,
		Routes:   map[string]config.AnthropicRoute{"claude-big": {}},
	}
	r := cfg.Routers["smart"]
	r.Tiers = []string{"cheap", "claude-big"}
	r.Questions = map[string]config.RouterQuestion{"hard": {Instructions: "x", EscalateTo: "claude-big", Threshold: 0.5}}
	cfg.Routers["smart"] = r
	h := newRouterServer(t, cfg)

	rec := do(t, h, "POST", "/v1/messages", `{"model":"smart","max_tokens":10,"messages":[{"role":"user","content":"debug it"}]}`, nil)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if got := anth.last(t)["model"]; got != "claude-big" {
		t.Errorf("route upstream got model %v, want claude-big", got)
	}
}

func TestRouterAllowlistAndListing(t *testing.T) {
	up := newRecordingUpstream(t, 200, "application/json", okChat)
	jev := newFakeJev(t, 200)
	cfg := routerConfig(up.URL, jev.URL)
	cfg.ClientKeys = []config.ClientKey{
		{Name: "routed", Key: "k-routed", Models: []string{"smart"}},
		{Name: "cheap", Key: "k-cheap", Models: []string{"cheap"}},
	}
	h := newRouterServer(t, cfg)
	auth := func(k string) map[string]string { return map[string]string{"Authorization": "Bearer " + k} }

	if rec := do(t, h, "POST", "/v1/chat/completions", chatBody("smart", "debug it"), auth("k-routed")); rec.Code != 200 {
		t.Errorf("router allowlisted, target not: status %d: %s", rec.Code, rec.Body)
	}
	if rec := do(t, h, "POST", "/v1/chat/completions", chatBody("smart", "hi"), auth("k-cheap")); rec.Code != 403 {
		t.Errorf("router not allowlisted: status %d, want 403", rec.Code)
	}
	if n := jev.hits.Load(); n != 1 {
		t.Errorf("classifier called %d times, want 1 (a forbidden key must not reach it)", n)
	}

	rec := do(t, h, "GET", "/v1/models", "", auth("k-routed"))
	if !strings.Contains(rec.Body.String(), `"id":"smart"`) {
		t.Errorf("/v1/models misses the router: %s", rec.Body)
	}
}
