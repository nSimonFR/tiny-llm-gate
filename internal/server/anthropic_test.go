package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
)

// buildAnthropicServer creates a Server with an Anthropic config pointing at
// the given upstream URL, plus minimal provider/model entries to satisfy
// config validation and the cc/ echo handler.
func buildAnthropicServer(t *testing.T, upstreamURL, shadowURL string) *Server {
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
			Upstream:  upstreamURL,
			ShadowURL: shadowURL,
		},
	}
	s, err := New(cfg, discardLogger())
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	return s
}

func postAnthropicJSON(t *testing.T, h http.Handler, body map[string]any) *httptest.ResponseRecorder {
	t.Helper()
	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages?beta=true", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer sk-ant-test-token")
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("anthropic-beta", "oauth-2025-04-20,interleaved-thinking-2025-05-14")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestAnthropicPassthroughHeaders(t *testing.T) {
	var gotHeaders http.Header
	var gotPath string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeaders = r.Header.Clone()
		gotPath = r.URL.Path + "?" + r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"type":"message","id":"msg_1","model":"claude-opus-4-6","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":2}}`))
	}))
	defer upstream.Close()

	s := buildAnthropicServer(t, upstream.URL, "")
	rec := postAnthropicJSON(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 100,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})

	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}

	// Verify auth header passed through.
	if got := gotHeaders.Get("Authorization"); got != "Bearer sk-ant-test-token" {
		t.Errorf("Authorization = %q", got)
	}
	// Verify anthropic-version passed through.
	if got := gotHeaders.Get("Anthropic-Version"); got != "2023-06-01" {
		t.Errorf("anthropic-version = %q", got)
	}
	// Verify anthropic-beta passed through.
	if got := gotHeaders.Get("Anthropic-Beta"); !strings.Contains(got, "oauth-2025-04-20") {
		t.Errorf("anthropic-beta = %q", got)
	}
	// Verify query string preserved.
	if gotPath != "/v1/messages?beta=true" {
		t.Errorf("upstream path = %q", gotPath)
	}
}

func TestAnthropicStreamingUsageExtraction(t *testing.T) {
	sseChunks := []string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-6\",\"usage\":{\"input_tokens\":100,\"cache_read_input_tokens\":5000,\"cache_creation_input_tokens\":200}}}\n\n",
		"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
		"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":42}}\n\n",
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
			time.Sleep(2 * time.Millisecond)
		}
	}))
	defer upstream.Close()

	// Use a channel-based approach to capture the shadow request.
	var shadowReceived atomic.Int32
	shadow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		shadowReceived.Add(1)
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		_ = json.Unmarshal(body, &req)
		if model, ok := req["model"].(string); !ok || !strings.HasPrefix(model, "cc/") {
			t.Errorf("shadow model = %v", req["model"])
		}
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"id":"shadow","object":"chat.completion","choices":[]}`))
	}))
	defer shadow.Close()

	s := buildAnthropicServer(t, upstream.URL, shadow.URL)
	rec := postAnthropicJSON(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 100,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
		"stream":     true,
	})

	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}

	// Verify SSE content was forwarded.
	body := rec.Body.String()
	if !strings.Contains(body, "Hello") {
		t.Errorf("expected streamed content 'Hello' in body:\n%s", body)
	}
	if !strings.Contains(body, "message_stop") {
		t.Errorf("expected message_stop in body:\n%s", body)
	}

	// Wait briefly for async shadow goroutine.
	time.Sleep(100 * time.Millisecond)
	if shadowReceived.Load() != 1 {
		t.Errorf("expected 1 shadow request, got %d", shadowReceived.Load())
	}
}

func TestAnthropicNonStreamingUsageExtraction(t *testing.T) {
	respBody := `{"type":"message","id":"msg_1","model":"claude-opus-4-6","content":[{"type":"text","text":"Hello!"}],"stop_reason":"end_turn","usage":{"input_tokens":50,"output_tokens":5,"cache_read_input_tokens":1000,"cache_creation_input_tokens":30}}`

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(respBody))
	}))
	defer upstream.Close()

	var shadowBody string
	shadow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		shadowBody = string(b)
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{}`))
	}))
	defer shadow.Close()

	s := buildAnthropicServer(t, upstream.URL, shadow.URL)
	rec := postAnthropicJSON(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 100,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})

	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "Hello!") {
		t.Errorf("response body missing expected content:\n%s", rec.Body.String())
	}

	time.Sleep(100 * time.Millisecond)
	if shadowBody == "" {
		t.Fatal("shadow request not received")
	}
	if !strings.Contains(shadowBody, "cc/claude-opus-4-6") {
		t.Errorf("shadow body missing cc/ model prefix:\n%s", shadowBody)
	}
}

func TestAnthropicUpstreamErrorNoShadow(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(401)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"authentication_error","message":"invalid api key"}}`))
	}))
	defer upstream.Close()

	var shadowReceived atomic.Int32
	shadow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		shadowReceived.Add(1)
		w.WriteHeader(200)
	}))
	defer shadow.Close()

	s := buildAnthropicServer(t, upstream.URL, shadow.URL)
	rec := postAnthropicJSON(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 100,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})

	if rec.Code != 401 {
		t.Fatalf("expected 401, got %d", rec.Code)
	}

	time.Sleep(100 * time.Millisecond)
	if shadowReceived.Load() != 0 {
		t.Errorf("expected no shadow on error, got %d", shadowReceived.Load())
	}
}

func TestShadowEchoReturnsValidOpenAI(t *testing.T) {
	s := buildServer(t,
		map[string]config.Provider{
			"ollama": {Type: "openai", BaseURL: "http://localhost:1/v1"},
		},
		map[string]config.Model{
			"stub": {Provider: "ollama", UpstreamModel: "stub"},
		},
		nil,
	)

	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "cc/claude-opus-4-6",
		"messages": []map[string]any{{"role": "user", "content": "shadow"}},
	})

	if rec.Code != 200 {
		t.Fatalf("expected 200 for cc/ echo, got %d body=%s", rec.Code, rec.Body.String())
	}

	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON response: %v", err)
	}
	if resp["object"] != "chat.completion" {
		t.Errorf("object = %v", resp["object"])
	}
	if resp["model"] != "cc/claude-opus-4-6" {
		t.Errorf("model = %v", resp["model"])
	}
	if resp["usage"] == nil {
		t.Error("missing usage field")
	}
}

func TestShadowEchoReturnsUsageFromXUsage(t *testing.T) {
	s := buildServer(t,
		map[string]config.Provider{
			"ollama": {Type: "openai", BaseURL: "http://localhost:1/v1"},
		},
		map[string]config.Model{
			"stub": {Provider: "ollama", UpstreamModel: "stub"},
		},
		nil,
	)

	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "cc/claude-opus-4-6",
		"messages": []map[string]any{{"role": "user", "content": "shadow"}},
		"x_usage":  map[string]int{"prompt_tokens": 500, "completion_tokens": 42, "total_tokens": 542},
	})

	if rec.Code != 200 {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}

	var resp struct {
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if resp.Usage.PromptTokens != 500 {
		t.Errorf("prompt_tokens = %d, want 500", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 42 {
		t.Errorf("completion_tokens = %d, want 42", resp.Usage.CompletionTokens)
	}
	if resp.Usage.TotalTokens != 542 {
		t.Errorf("total_tokens = %d, want 542", resp.Usage.TotalTokens)
	}
}

func TestAnthropicDisabledWhenNilConfig(t *testing.T) {
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

	// /v1/messages should not be registered — expect 405 or 404.
	if rec.Code == 200 {
		t.Errorf("expected non-200 when anthropic not configured, got %d", rec.Code)
	}
}

func TestAnthropicNoShadowWhenURLEmpty(t *testing.T) {
	respBody := `{"type":"message","id":"msg_1","model":"claude-opus-4-6","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(respBody))
	}))
	defer upstream.Close()

	// No shadow URL — should not panic or send shadow.
	s := buildAnthropicServer(t, upstream.URL, "")
	rec := postAnthropicJSON(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})

	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}
}

func TestAnthropicStreamTeeUsageUnit(t *testing.T) {
	// Test the streaming tee function directly with known SSE data.
	sse := strings.Join([]string{
		"event: message_start\n",
		fmt.Sprintf("data: %s\n\n", `{"type":"message_start","message":{"usage":{"input_tokens":77,"cache_read_input_tokens":999,"cache_creation_input_tokens":42}}}`),
		"event: content_block_delta\n",
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}` + "\n\n",
		"event: message_delta\n",
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":13}}` + "\n\n",
		"event: message_stop\n",
		"data: {\"type\":\"message_stop\"}\n\n",
	}, "")

	w := httptest.NewRecorder()
	usage := anthropicStreamTee(w, strings.NewReader(sse), "claude-opus-4-6")

	if usage.InputTokens != 77 {
		t.Errorf("InputTokens = %d, want 77", usage.InputTokens)
	}
	if usage.OutputTokens != 13 {
		t.Errorf("OutputTokens = %d, want 13", usage.OutputTokens)
	}
	if usage.CacheReadInputTokens != 999 {
		t.Errorf("CacheReadInputTokens = %d, want 999", usage.CacheReadInputTokens)
	}
	if usage.CacheCreateTokens != 42 {
		t.Errorf("CacheCreateTokens = %d, want 42", usage.CacheCreateTokens)
	}

	// Verify all content was forwarded.
	body := w.Body.String()
	if !strings.Contains(body, "Hello") {
		t.Errorf("expected forwarded content 'Hello' in body")
	}
}
