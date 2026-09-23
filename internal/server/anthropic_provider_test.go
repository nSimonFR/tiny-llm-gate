package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
)

// mockAnthropicBackend stands in for api.anthropic.com's /v1/messages. It
// records the request and replies with a canned Messages SSE stream.
type mockAnthropicBackend struct {
	*httptest.Server
	gotBody string
	gotAuth string
	gotPath string
}

func newMockAnthropicBackend(t *testing.T, sseBody string) *mockAnthropicBackend {
	t.Helper()
	m := &mockAnthropicBackend{}
	m.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		m.gotBody = string(b)
		m.gotAuth = r.Header.Get("Authorization")
		m.gotPath = r.URL.Path
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = io.WriteString(w, sseBody)
	}))
	return m
}

// anthropicSSE joins Anthropic Messages events into an SSE body (no [DONE]).
func anthropicSSE(events ...string) string {
	var sb strings.Builder
	for _, e := range events {
		sb.WriteString("event: x\ndata: ")
		sb.WriteString(e)
		sb.WriteString("\n\n")
	}
	return sb.String()
}

// anthropicProviderServer wires a routable type:anthropic provider backed by a
// single-account pool, plus a gemini alias, exercising the agnostic core.
func anthropicProviderServer(t *testing.T, backendURL string, accounts ...config.AnthropicAccount) *Server {
	t.Helper()
	if len(accounts) == 0 {
		accounts = []config.AnthropicAccount{{Name: "acct1", Auth: &config.Auth{Type: "bearer", Token: "acct1-token"}}}
	}
	cfg := &config.Config{
		Providers: map[string]config.Provider{
			"claude": {Type: "anthropic", BaseURL: backendURL},
		},
		Models:  map[string]config.Model{"claude-test": {Provider: "claude", UpstreamModel: "claude-sonnet-4-5"}},
		Aliases: map[string]string{"gemini-2.5-flash": "claude-test"},
		Anthropic: &config.Anthropic{
			Upstream: backendURL,
			Accounts: accounts,
		},
	}
	s, err := New(cfg, discardLogger())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s
}

func textSSE() string {
	return anthropicSSE(
		`{"type":"message_start","message":{"id":"m","usage":{"input_tokens":8}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Bon"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"jour"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`,
		`{"type":"message_stop"}`,
	)
}

func TestAnthropicProvider_OpenAINonStream(t *testing.T) {
	backend := newMockAnthropicBackend(t, textSSE())
	defer backend.Close()
	s := anthropicProviderServer(t, backend.URL)

	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "claude-test",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	// Pool auth applied + hit /v1/messages + Claude-Code sentinel in the body.
	if backend.gotAuth != "Bearer acct1-token" {
		t.Errorf("upstream auth = %q", backend.gotAuth)
	}
	if !strings.HasSuffix(backend.gotPath, "/v1/messages") {
		t.Errorf("upstream path = %q", backend.gotPath)
	}
	if !strings.Contains(backend.gotBody, "You are Claude Code") {
		t.Errorf("translated body missing sentinel: %s", backend.gotBody)
	}
	var resp struct {
		Choices []struct {
			Message struct{ Content string } `json:"message"`
		} `json:"choices"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Choices[0].Message.Content != "Bonjour" {
		t.Fatalf("content = %q", resp.Choices[0].Message.Content)
	}
	if resp.Usage.TotalTokens != 10 {
		t.Fatalf("total tokens = %d", resp.Usage.TotalTokens)
	}
}

func TestAnthropicProvider_OpenAIStreamToolCall(t *testing.T) {
	backend := newMockAnthropicBackend(t, anthropicSSE(
		`{"type":"message_start","message":{"id":"m","usage":{"input_tokens":5}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}`,
		`{"type":"message_stop"}`,
	))
	defer backend.Close()
	s := anthropicProviderServer(t, backend.URL)

	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "claude-test",
		"stream":   true,
		"messages": []map[string]any{{"role": "user", "content": "weather?"}},
	})
	if rec.Code != 200 {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var name, args string
	for _, block := range strings.Split(rec.Body.String(), "\n\n") {
		block = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(block), "data:"))
		if block == "" || block == "[DONE]" {
			continue
		}
		var c struct {
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						Function struct{ Name, Arguments string } `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(block), &c) != nil || len(c.Choices) == 0 {
			continue
		}
		for _, tc := range c.Choices[0].Delta.ToolCalls {
			if tc.Function.Name != "" {
				name = tc.Function.Name
			}
			args += tc.Function.Arguments
		}
	}
	if name != "get_weather" || args != `{"city":"Paris"}` {
		t.Fatalf("tool call name=%q args=%q", name, args)
	}
}

// The routable anthropic provider must also work behind the GEMINI frontend
// (proving the agnostic core: any provider × any frontend).
func TestAnthropicProvider_GeminiNonStream(t *testing.T) {
	backend := newMockAnthropicBackend(t, textSSE())
	defer backend.Close()
	s := anthropicProviderServer(t, backend.URL)

	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:generateContent",
		strings.NewReader(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if got := geminiCandidateText(resp); got != "Bonjour" {
		t.Fatalf("gemini candidate text = %q\n%s", got, rec.Body.String())
	}
}

// Disconnection: a streaming client drop mid-anthropic-stream must cancel the
// upstream and unwind the goroutine (no leak) — same guarantee as codex.
func TestAnthropicProvider_ClientDisconnectUnwinds(t *testing.T) {
	upstreamCanceled := make(chan struct{}, 1)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		fl, _ := w.(http.Flusher)
		_, _ = io.WriteString(w, "event: x\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"usage\":{\"input_tokens\":1}}}\n\n")
		_, _ = io.WriteString(w, "event: x\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n")
		_, _ = io.WriteString(w, "event: x\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n")
		if fl != nil {
			fl.Flush()
		}
		<-r.Context().Done()
		select {
		case upstreamCanceled <- struct{}{}:
		default:
		}
	}))
	defer backend.Close()

	s := anthropicProviderServer(t, backend.URL)
	gate := httptest.NewServer(s.Handler())
	defer gate.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, gate.URL+"/v1/chat/completions",
		strings.NewReader(`{"model":"claude-test","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	buf := make([]byte, 1)
	if _, err := resp.Body.Read(buf); err != nil {
		t.Fatalf("read first byte: %v", err)
	}
	cancel()
	_ = resp.Body.Close()

	select {
	case <-upstreamCanceled:
	case <-time.After(5 * time.Second):
		t.Fatal("upstream not canceled after client disconnect — goroutine/connection leak")
	}
}

// The routable provider and the passthrough SHARE the account pool: a 429 that
// cools acct1 during a routed request must also steer a subsequent passthrough
// request to acct2 (shared sticky-until-429 state, not a fork).
func TestAnthropicProvider_SharedPoolCooldown(t *testing.T) {
	var mu struct {
		tokens []string
	}
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tok := r.Header.Get("Authorization")
		mu.tokens = append(mu.tokens, tok)
		if tok == "Bearer acct1-token" {
			w.Header().Set("Retry-After", "300")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"type":"error","error":{"type":"rate_limit_error"}}`)
			return
		}
		// acct2 (or anything else): succeed.
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = io.WriteString(w, textSSE())
	}))
	defer backend.Close()

	s := anthropicProviderServer(t, backend.URL,
		config.AnthropicAccount{Name: "acct1", Auth: &config.Auth{Type: "bearer", Token: "acct1-token"}},
		config.AnthropicAccount{Name: "acct2", Auth: &config.Auth{Type: "bearer", Token: "acct2-token"}},
	)

	// (1) Routed request: acct1 429s → fails over to acct2 → 200.
	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "claude-test",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 200 {
		t.Fatalf("routed status = %d body=%s", rec.Code, rec.Body.String())
	}

	// (2) Passthrough request now: acct1 is cooling (shared state) → must use acct2.
	before := len(mu.tokens)
	prec := postJSON(t, s.Handler(), "/v1/messages", map[string]any{
		"model":      "claude-sonnet-4-5",
		"max_tokens": 16,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	if prec.Code != 200 {
		t.Fatalf("passthrough status = %d body=%s", prec.Code, prec.Body.String())
	}
	// The passthrough attempt(s) after `before` must NOT include acct1 (cooling).
	for _, tok := range mu.tokens[before:] {
		if tok == "Bearer acct1-token" {
			t.Fatalf("passthrough hit cooled acct1 — pool state not shared: %v", mu.tokens)
		}
	}
}
