package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
)

// buildAnthropicServer builds a Server with an Anthropic config pointing at
// the given upstream URL and (optionally) a bearer token. Minimum providers
// + models are included to satisfy config validation.
func buildAnthropicServer(t *testing.T, upstreamURL, bearerToken string) *Server {
	t.Helper()
	cfg := &config.Config{
		Listen: "127.0.0.1:0",
		Providers: map[string]config.Provider{
			"stub": {Type: "openai", BaseURL: "http://localhost:1/v1"},
		},
		Models: map[string]config.Model{
			"stub": {Provider: "stub", UpstreamModel: "stub"},
		},
		Anthropic: &config.Anthropic{
			Upstream: upstreamURL,
		},
	}
	if bearerToken != "" {
		cfg.Anthropic.Auth = &config.Auth{Type: "bearer", Token: bearerToken}
	}
	s, err := New(cfg, discardLogger())
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	return s
}

// postMessages sends a POST /v1/messages request with a fake OAuth
// Authorization header and the Anthropic-specific metadata headers.
func postMessages(t *testing.T, h http.Handler, body map[string]any) (*httptest.ResponseRecorder, *http.Request) {
	t.Helper()
	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages?beta=true", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer sk-ant-oat01-incoming-token")
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("anthropic-beta", "oauth-2025-04-20")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec, req
}

func TestAnthropicPassthroughHeaders(t *testing.T) {
	var gotHeaders http.Header
	var gotPath string
	var gotBody []byte

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeaders = r.Header.Clone()
		gotPath = r.URL.Path + "?" + r.URL.RawQuery
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"type":"message","id":"msg_1","model":"claude-opus-4-6","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":2}}`))
	}))
	defer upstream.Close()

	s := buildAnthropicServer(t, upstream.URL, "")
	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 100,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})

	if rec.Code != 200 {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	// Non-auth headers pass through.
	if got := gotHeaders.Get("Anthropic-Version"); got != "2023-06-01" {
		t.Errorf("anthropic-version = %q", got)
	}
	if got := gotHeaders.Get("Anthropic-Beta"); got != "oauth-2025-04-20" {
		t.Errorf("anthropic-beta = %q", got)
	}
	// Query string preserved.
	if gotPath != "/v1/messages?beta=true" {
		t.Errorf("path = %q", gotPath)
	}
	// Body passes through unchanged.
	if !bytes.Contains(gotBody, []byte(`"claude-opus-4-6"`)) {
		t.Errorf("body missing model: %s", gotBody)
	}
}

func TestAnthropicStripsIncomingAuthWhenNoneConfigured(t *testing.T) {
	// Without configured auth, the handler must STRIP the incoming
	// Authorization header so we don't leak Aperture's apikey upstream.
	var gotAuth string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"type":"message","id":"msg_1","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer upstream.Close()

	s := buildAnthropicServer(t, upstream.URL, "")
	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})

	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}
	if gotAuth != "" {
		t.Errorf("expected Authorization stripped, got %q", gotAuth)
	}
}

func TestAnthropicAppliesConfiguredAuth(t *testing.T) {
	// With configured bearer token, the handler must replace the incoming
	// Authorization with the configured one (not pass the incoming one through).
	var gotAuth string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"type":"message","id":"msg_1","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer upstream.Close()

	s := buildAnthropicServer(t, upstream.URL, "sk-ant-oat01-configured-bot-token")
	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})

	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}
	if gotAuth != "Bearer sk-ant-oat01-configured-bot-token" {
		t.Errorf("upstream Authorization = %q; expected configured bot token", gotAuth)
	}
}

func TestAnthropicStreamingRelayedUnchanged(t *testing.T) {
	sseChunks := []string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-6\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher, _ := w.(http.Flusher)
		for _, chunk := range sseChunks {
			_, _ = w.Write([]byte(chunk))
			if flusher != nil {
				flusher.Flush()
			}
		}
	}))
	defer upstream.Close()

	s := buildAnthropicServer(t, upstream.URL, "")
	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
		"stream":     true,
	})

	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}
	body := rec.Body.String()
	for _, want := range []string{"message_start", "Hello", "message_stop"} {
		if !strings.Contains(body, want) {
			t.Errorf("streamed body missing %q:\n%s", want, body)
		}
	}
}

func TestAnthropicUpstreamErrorPassesThrough(t *testing.T) {
	// 4xx/5xx responses from upstream are forwarded to the client as-is,
	// no rewriting.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(401)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"authentication_error","message":"invalid token"}}`))
	}))
	defer upstream.Close()

	s := buildAnthropicServer(t, upstream.URL, "sk-ant-oat01-bad-token")
	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})

	if rec.Code != 401 {
		t.Fatalf("expected 401 pass-through, got %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "authentication_error") {
		t.Errorf("body missing upstream error message: %s", rec.Body.String())
	}
}

func TestAnthropicRouteNotRegisteredWithoutConfig(t *testing.T) {
	// When the anthropic config section is absent, /v1/messages must not
	// be routed (returns 404 / 405).
	s := buildServer(t,
		map[string]config.Provider{
			"ollama": {Type: "openai", BaseURL: "http://localhost:1/v1"},
		},
		map[string]config.Model{
			"stub": {Provider: "ollama", UpstreamModel: "stub"},
		},
		nil,
	)

	b, _ := json.Marshal(map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code == 200 {
		t.Errorf("expected non-200 when anthropic not configured, got %d", rec.Code)
	}
}

func TestAnthropicBodyPassesThroughUnchanged(t *testing.T) {
	// Verify we do NOT rewrite the model field or any other field in the
	// body — unlike /v1/chat/completions which rewrites client-facing
	// model names to upstream model ids.
	var gotBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"type":"message","id":"msg_1","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer upstream.Close()

	s := buildAnthropicServer(t, upstream.URL, "")
	originalBody := map[string]any{
		"model":       "claude-opus-4-6",
		"max_tokens":  100,
		"temperature": 0.7,
		"system":      "You are helpful.",
		"messages": []map[string]any{
			{"role": "user", "content": "hi"},
		},
	}
	rec, _ := postMessages(t, s.Handler(), originalBody)
	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}

	var parsed map[string]any
	if err := json.Unmarshal(gotBody, &parsed); err != nil {
		t.Fatalf("upstream body not valid JSON: %v", err)
	}
	if parsed["model"] != "claude-opus-4-6" {
		t.Errorf("model rewritten: %v", parsed["model"])
	}
	if parsed["system"] != "You are helpful." {
		t.Errorf("system prompt lost: %v", parsed["system"])
	}
	if parsed["temperature"] != 0.7 {
		t.Errorf("temperature lost: %v", parsed["temperature"])
	}
}
