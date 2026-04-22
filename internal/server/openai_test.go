package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
)

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// mockUpstream returns a server that records every request it received.
type mockUpstream struct {
	*httptest.Server
	mu       *requestLog
	status   int
	respBody string
	// When set, the upstream streams chunks with 5ms gaps between them.
	stream []string
}

type requestLog struct {
	requests atomic.Int32
	bodies   []string
	paths    []string
	headers  []http.Header
}

func newMockUpstream(status int, respBody string) *mockUpstream {
	m := &mockUpstream{status: status, respBody: respBody, mu: &requestLog{}}
	m.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		m.mu.requests.Add(1)
		body, _ := io.ReadAll(r.Body)
		m.mu.bodies = append(m.mu.bodies, string(body))
		m.mu.paths = append(m.mu.paths, r.URL.Path)
		m.mu.headers = append(m.mu.headers, r.Header.Clone())

		if len(m.stream) > 0 {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)
			flusher, _ := w.(http.Flusher)
			for _, chunk := range m.stream {
				_, _ = w.Write([]byte(chunk))
				if flusher != nil {
					flusher.Flush()
				}
				time.Sleep(5 * time.Millisecond)
			}
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(m.status)
		_, _ = w.Write([]byte(m.respBody))
	}))
	return m
}

func buildServer(t *testing.T, providers map[string]config.Provider, models map[string]config.Model, aliases map[string]string) *Server {
	t.Helper()
	cfg := &config.Config{
		Listen:    "127.0.0.1:0",
		Providers: providers,
		Models:    models,
		Aliases:   aliases,
	}
	s, err := New(cfg, discardLogger())
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	return s
}

func postJSON(t *testing.T, h http.Handler, path string, body map[string]any) *httptest.ResponseRecorder {
	t.Helper()
	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestChatCompletionsNonStreaming(t *testing.T) {
	upstream := newMockUpstream(200, `{"id":"x","choices":[{"message":{"content":"hi"}}]}`)
	defer upstream.Close()

	s := buildServer(t,
		map[string]config.Provider{
			"ollama": {Type: "openai", BaseURL: upstream.URL + "/v1", APIKey: "ollama"},
		},
		map[string]config.Model{
			"gemma4": {Provider: "ollama", UpstreamModel: "gemma4:e4b"},
		},
		map[string]string{"gpt-4o": "gemma4"},
	)

	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "gpt-4o",
		"messages": []map[string]any{{"role": "user", "content": "hello"}},
	})

	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	if got := upstream.mu.requests.Load(); got != 1 {
		t.Errorf("expected 1 upstream request, got %d", got)
	}
	if upstream.mu.paths[0] != "/v1/chat/completions" {
		t.Errorf("upstream path = %q", upstream.mu.paths[0])
	}
	// Verify model was rewritten.
	var sent map[string]any
	_ = json.Unmarshal([]byte(upstream.mu.bodies[0]), &sent)
	if sent["model"] != "gemma4:e4b" {
		t.Errorf("expected rewritten model 'gemma4:e4b', got %v", sent["model"])
	}
	// Verify auth header was added.
	if upstream.mu.headers[0].Get("Authorization") != "Bearer ollama" {
		t.Errorf("expected Authorization header, got %q", upstream.mu.headers[0].Get("Authorization"))
	}
}

func TestChatCompletionsStreamingPreservesChunks(t *testing.T) {
	upstream := newMockUpstream(200, "")
	upstream.stream = []string{
		"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
		"data: [DONE]\n\n",
	}
	defer upstream.Close()

	s := buildServer(t,
		map[string]config.Provider{
			"ollama": {Type: "openai", BaseURL: upstream.URL + "/v1", APIKey: "ollama"},
		},
		map[string]config.Model{
			"gemma4": {Provider: "ollama", UpstreamModel: "gemma4:e4b"},
		},
		nil,
	)

	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "gemma4",
		"stream":   true,
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	ct := rec.Header().Get("Content-Type")
	if !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("expected SSE content-type, got %q", ct)
	}
	body := rec.Body.String()
	for _, want := range []string{"Hello", " world", "[DONE]"} {
		if !strings.Contains(body, want) {
			t.Errorf("expected body to contain %q; got:\n%s", want, body)
		}
	}
}

func TestFallbackOn5xx(t *testing.T) {
	primary := newMockUpstream(500, `{"error":"overloaded"}`)
	defer primary.Close()
	secondary := newMockUpstream(200, `{"id":"x","choices":[{"message":{"content":"hi"}}]}`)
	defer secondary.Close()

	s := buildServer(t,
		map[string]config.Provider{
			"codex":  {Type: "openai", BaseURL: primary.URL + "/v1"},
			"ollama": {Type: "openai", BaseURL: secondary.URL + "/v1"},
		},
		map[string]config.Model{
			"gpt-5":  {Provider: "codex", UpstreamModel: "gpt-5", Fallback: []string{"gemma4"}},
			"gemma4": {Provider: "ollama", UpstreamModel: "gemma4:e4b"},
		},
		nil,
	)

	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "gpt-5",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 200 {
		t.Fatalf("expected 200 after fallback, got %d body=%s", rec.Code, rec.Body.String())
	}
	if primary.mu.requests.Load() != 1 {
		t.Errorf("expected 1 request to primary, got %d", primary.mu.requests.Load())
	}
	if secondary.mu.requests.Load() != 1 {
		t.Errorf("expected 1 request to secondary, got %d", secondary.mu.requests.Load())
	}
	// Secondary must have received the rewritten fallback model.
	var sent map[string]any
	_ = json.Unmarshal([]byte(secondary.mu.bodies[0]), &sent)
	if sent["model"] != "gemma4:e4b" {
		t.Errorf("secondary received model %v; expected gemma4:e4b", sent["model"])
	}
}

func TestFallbackExhausted(t *testing.T) {
	a := newMockUpstream(500, `{"error":"a down"}`)
	defer a.Close()
	b := newMockUpstream(502, `{"error":"b down"}`)
	defer b.Close()

	s := buildServer(t,
		map[string]config.Provider{
			"pa": {Type: "openai", BaseURL: a.URL + "/v1"},
			"pb": {Type: "openai", BaseURL: b.URL + "/v1"},
		},
		map[string]config.Model{
			"ma": {Provider: "pa", UpstreamModel: "ma", Fallback: []string{"mb"}},
			"mb": {Provider: "pb", UpstreamModel: "mb"},
		},
		nil,
	)

	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "ma",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	// Last hop has no fallback; its status must surface directly.
	if rec.Code != 502 {
		t.Errorf("expected 502 (last fallback's status), got %d", rec.Code)
	}
}

func TestNoFallbackOnWrapped4xxInvalidRequestError(t *testing.T) {
	// Native OpenAI error with type "invalid_request_error".
	primary := newMockUpstream(500, `{"error":{"message":"Invalid tool name","type":"invalid_request_error"}}`)
	defer primary.Close()
	secondary := newMockUpstream(200, `{"id":"x","choices":[{"message":{"content":"hi"}}]}`)
	defer secondary.Close()

	s := buildServer(t,
		map[string]config.Provider{
			"codex":  {Type: "openai", BaseURL: primary.URL + "/v1"},
			"ollama": {Type: "openai", BaseURL: secondary.URL + "/v1"},
		},
		map[string]config.Model{
			"gpt-5":  {Provider: "codex", UpstreamModel: "gpt-5", Fallback: []string{"gemma4"}},
			"gemma4": {Provider: "ollama", UpstreamModel: "gemma4:e4b"},
		},
		nil,
	)

	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "gpt-5",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 400 {
		t.Fatalf("expected 400 (wrapped client error), got %d body=%s", rec.Code, rec.Body.String())
	}
	if secondary.mu.requests.Load() != 0 {
		t.Errorf("expected 0 requests to secondary (no fallback), got %d", secondary.mu.requests.Load())
	}
}

func TestNoFallbackOnWrapped4xx(t *testing.T) {
	// Simulate openai-oauth wrapping a 400 as 500 with
	// type "server_error" but an OpenAI validation message prefix.
	primary := newMockUpstream(500, `{"error":{"message":"Invalid 'input[1].name': string does not match pattern.","type":"server_error"}}`)
	defer primary.Close()
	secondary := newMockUpstream(200, `{"id":"x","choices":[{"message":{"content":"hi"}}]}`)
	defer secondary.Close()

	s := buildServer(t,
		map[string]config.Provider{
			"codex":  {Type: "openai", BaseURL: primary.URL + "/v1"},
			"ollama": {Type: "openai", BaseURL: secondary.URL + "/v1"},
		},
		map[string]config.Model{
			"gpt-5":  {Provider: "codex", UpstreamModel: "gpt-5", Fallback: []string{"gemma4"}},
			"gemma4": {Provider: "ollama", UpstreamModel: "gemma4:e4b"},
		},
		nil,
	)

	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "gpt-5",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	// Should NOT fall back — should pass through as 400.
	if rec.Code != 400 {
		t.Fatalf("expected 400 (wrapped client error), got %d body=%s", rec.Code, rec.Body.String())
	}
	if secondary.mu.requests.Load() != 0 {
		t.Errorf("expected 0 requests to secondary (no fallback), got %d", secondary.mu.requests.Load())
	}
	// Body should contain the original error.
	if !strings.Contains(rec.Body.String(), "Invalid '") {
		t.Errorf("expected original error message in body, got: %s", rec.Body.String())
	}
}

func TestFallbackOnReal5xx(t *testing.T) {
	// A genuine 500 with type "server_error" should still trigger fallback.
	primary := newMockUpstream(500, `{"error":{"message":"internal error","type":"server_error"}}`)
	defer primary.Close()
	secondary := newMockUpstream(200, `{"id":"x","choices":[{"message":{"content":"hi"}}]}`)
	defer secondary.Close()

	s := buildServer(t,
		map[string]config.Provider{
			"codex":  {Type: "openai", BaseURL: primary.URL + "/v1"},
			"ollama": {Type: "openai", BaseURL: secondary.URL + "/v1"},
		},
		map[string]config.Model{
			"gpt-5":  {Provider: "codex", UpstreamModel: "gpt-5", Fallback: []string{"gemma4"}},
			"gemma4": {Provider: "ollama", UpstreamModel: "gemma4:e4b"},
		},
		nil,
	)

	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "gpt-5",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	// Should fall back to secondary and succeed.
	if rec.Code != 200 {
		t.Fatalf("expected 200 after fallback, got %d body=%s", rec.Code, rec.Body.String())
	}
	if secondary.mu.requests.Load() != 1 {
		t.Errorf("expected 1 request to secondary, got %d", secondary.mu.requests.Load())
	}
}

func TestUnknownModelReturns404(t *testing.T) {
	s := buildServer(t,
		map[string]config.Provider{
			"ollama": {Type: "openai", BaseURL: "http://localhost:1/v1"},
		},
		map[string]config.Model{
			"gemma4": {Provider: "ollama", UpstreamModel: "gemma4:e4b"},
		},
		nil,
	)
	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "nonexistent",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 404 {
		t.Errorf("expected 404 for unknown model, got %d", rec.Code)
	}
}

func TestMissingModelField(t *testing.T) {
	s := buildServer(t,
		map[string]config.Provider{"ollama": {Type: "openai", BaseURL: "http://x/v1"}},
		map[string]config.Model{"gemma4": {Provider: "ollama", UpstreamModel: "gemma4:e4b"}},
		nil,
	)
	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 400 {
		t.Errorf("expected 400, got %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestEmbeddingsRewritesModel(t *testing.T) {
	upstream := newMockUpstream(200, `{"data":[{"embedding":[0.1,0.2]}]}`)
	defer upstream.Close()

	s := buildServer(t,
		map[string]config.Provider{
			"ollama": {Type: "openai", BaseURL: upstream.URL + "/v1", APIKey: "ollama"},
		},
		map[string]config.Model{
			"qwen3-embed": {Provider: "ollama", UpstreamModel: "qwen3-embedding:8b"},
		},
		map[string]string{"text-embedding-3-small": "qwen3-embed"},
	)

	rec := postJSON(t, s.Handler(), "/v1/embeddings", map[string]any{
		"model": "text-embedding-3-small",
		"input": []string{"hello"},
	})
	if rec.Code != 200 {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if upstream.mu.paths[0] != "/v1/embeddings" {
		t.Errorf("expected /v1/embeddings, got %s", upstream.mu.paths[0])
	}
	var sent map[string]any
	_ = json.Unmarshal([]byte(upstream.mu.bodies[0]), &sent)
	if sent["model"] != "qwen3-embedding:8b" {
		t.Errorf("expected model rewritten to qwen3-embedding:8b, got %v", sent["model"])
	}
}

func TestModelsList(t *testing.T) {
	s := buildServer(t,
		map[string]config.Provider{
			"ollama": {Type: "openai", BaseURL: "http://x/v1"},
		},
		map[string]config.Model{
			"gemma4": {Provider: "ollama", UpstreamModel: "gemma4:e4b"},
		},
		map[string]string{"gpt-4o": "gemma4"},
	)
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}
	var out struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	ids := make([]string, len(out.Data))
	for i, it := range out.Data {
		ids[i] = it.ID
	}
	if len(ids) != 2 {
		t.Errorf("expected 2 ids, got %v", ids)
	}
}

func TestRewriteModelFieldPreservesOtherFields(t *testing.T) {
	in := []byte(`{"model":"old","temperature":0.7,"messages":[{"role":"user","content":"hi"}]}`)
	out, err := rewriteModelField(in, "new")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	var parsed map[string]any
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("output invalid JSON: %v\n%s", err, out)
	}
	if parsed["model"] != "new" {
		t.Errorf("model = %v", parsed["model"])
	}
	if parsed["temperature"] != 0.7 {
		t.Errorf("temperature lost: %v", parsed["temperature"])
	}
	if _, ok := parsed["messages"]; !ok {
		t.Errorf("messages lost")
	}
}

func TestRewriteModelFieldMissingModel(t *testing.T) {
	if _, err := rewriteModelField([]byte(`{"messages":[]}`), "new"); err == nil {
		t.Error("expected error when model field missing")
	}
}

func TestRewriteModelFieldInvalidJSON(t *testing.T) {
	if _, err := rewriteModelField([]byte(`not json`), "new"); err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestOAuthChatGPTBackendAppliesBearer(t *testing.T) {
	newAccess := makeTestJWT(3600)

	issuer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"access_token":"` + newAccess + `","token_type":"Bearer","expires_in":3600}`))
	}))
	defer issuer.Close()

	upstream := newMockUpstream(200, `{"id":"x","choices":[{"message":{"content":"hi"}}]}`)
	defer upstream.Close()

	authFile := filepath.Join(t.TempDir(), "auth.json")
	authContent := `{
  "tokens": {"access_token":"` + makeTestJWT(-120) + `","refresh_token":"rt"},
  "last_refresh": "2020-01-01T00:00:00Z"
}`
	if err := os.WriteFile(authFile, []byte(authContent), 0o600); err != nil {
		t.Fatal(err)
	}

	cfg := &config.Config{
		Providers: map[string]config.Provider{
			"codex": {
				Type:    "openai",
				BaseURL: upstream.URL + "/v1",
				Auth: &config.Auth{
					Type:     "oauth_chatgpt",
					File:     authFile,
					Issuer:   issuer.URL,
					ClientID: "test",
				},
			},
		},
		Models: map[string]config.Model{
			"gpt-5": {Provider: "codex", UpstreamModel: "gpt-5"},
		},
	}
	s, err := New(cfg, discardLogger())
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "gpt-5",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 200 {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	got := upstream.mu.headers[0].Get("Authorization")
	if got != "Bearer "+newAccess {
		t.Errorf("upstream Authorization = %q; expected Bearer %s", got, newAccess)
	}
}

func TestInjectDefaultDimensions(t *testing.T) {
	// Without dimensions — should inject.
	body := []byte(`{"model":"emb","input":["hello"]}`)
	got := injectDefaultDimensions(body, 1024)
	var parsed map[string]json.RawMessage
	if err := json.Unmarshal(got, &parsed); err != nil {
		t.Fatal(err)
	}
	var dims int
	if err := json.Unmarshal(parsed["dimensions"], &dims); err != nil {
		t.Fatal(err)
	}
	if dims != 1024 {
		t.Errorf("expected 1024, got %d", dims)
	}

	// With dimensions already set — should not override.
	body2 := []byte(`{"model":"emb","input":["hello"],"dimensions":512}`)
	got2 := injectDefaultDimensions(body2, 1024)
	var parsed2 map[string]json.RawMessage
	if err := json.Unmarshal(got2, &parsed2); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(parsed2["dimensions"], &dims); err != nil {
		t.Fatal(err)
	}
	if dims != 512 {
		t.Errorf("expected 512 (original), got %d", dims)
	}
}

func makeTestJWT(expOffsetSec int64) string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none"}`))
	body := base64.RawURLEncoding.EncodeToString(
		[]byte(`{"exp":` + strconv.FormatInt(time.Now().Unix()+expOffsetSec, 10) + `}`),
	)
	return header + "." + body + ".sig"
}
