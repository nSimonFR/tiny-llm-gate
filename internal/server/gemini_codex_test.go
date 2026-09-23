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

// geminiCodexServer wires the Gemini frontend to a codex-type provider via a
// gemini-name alias, exercising the provider-agnostic chat core.
func geminiCodexServer(t *testing.T, backendURL string) *Server {
	t.Helper()
	creds := writeCodexCreds(t, t.TempDir())
	return buildServer(t,
		map[string]config.Provider{
			"codex": {Type: "codex", BaseURL: backendURL, Auth: &config.Auth{Type: "oauth_chatgpt", File: creds, Issuer: "http://unused"}},
		},
		map[string]config.Model{"gpt-5.5": {Provider: "codex", UpstreamModel: "gpt-5.5"}},
		map[string]string{"gemini-2.5-flash": "gpt-5.5"},
	)
}

// geminiCandidateText concatenates all candidates[].content.parts[].text.
func geminiCandidateText(m map[string]any) string {
	var b strings.Builder
	cands, _ := m["candidates"].([]any)
	for _, ci := range cands {
		content, _ := ci.(map[string]any)["content"].(map[string]any)
		parts, _ := content["parts"].([]any)
		for _, p := range parts {
			if s, ok := p.(map[string]any)["text"].(string); ok {
				b.WriteString(s)
			}
		}
	}
	return b.String()
}

// TestGeminiCodex_NonStream: a Gemini /v1beta request that resolves to a codex
// provider is translated (Gemini→OpenAI→Codex Responses→OpenAI→Gemini) and
// returns a Gemini-shaped completion. This is the fix for the codex-migration
// regression where the Gemini path byte-forwarded to a codex backend and 502'd.
func TestGeminiCodex_NonStream(t *testing.T) {
	backend := newMockCodexBackend(t, codexSSE(
		`{"type":"response.output_text.delta","delta":"Bon"}`,
		`{"type":"response.output_text.delta","delta":"jour"}`,
		`{"type":"response.completed","response":{"id":"r","usage":{"input_tokens":3,"output_tokens":2}}}`,
	))
	defer backend.Close()
	s := geminiCodexServer(t, backend.URL)

	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:generateContent",
		strings.NewReader(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	// The backend must have been hit at the Codex /responses path (proving the
	// codex translation ran, not a byte-forward to /chat/completions).
	if !strings.HasSuffix(backend.gotPath, "/responses") {
		t.Fatalf("codex backend path = %q, want .../responses", backend.gotPath)
	}
	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("response not JSON: %v\n%s", err, rec.Body.String())
	}
	if got := geminiCandidateText(resp); got != "Bonjour" {
		t.Fatalf("candidate text = %q, want %q\n%s", got, "Bonjour", rec.Body.String())
	}
}

// TestGeminiCodex_StreamToolCall: a streaming Gemini request to a codex provider
// surfaces a tool call as a Gemini functionCall with fully-populated args (the
// item_id arg-delta path, end-to-end through both translators).
func TestGeminiCodex_StreamToolCall(t *testing.T) {
	backend := newMockCodexBackend(t, codexSSE(
		`{"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"get_weather","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\"city\":"}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"\"Paris\"}"}`,
		`{"type":"response.function_call_arguments.done","item_id":"fc_1","arguments":"{\"city\":\"Paris\"}"}`,
		`{"type":"response.completed","response":{"id":"r","usage":{"input_tokens":5,"output_tokens":7}}}`,
	))
	defer backend.Close()
	s := geminiCodexServer(t, backend.URL)

	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
		strings.NewReader(`{"contents":[{"role":"user","parts":[{"text":"weather in Paris? call get_weather"}]}]}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	out := rec.Body.String()
	// The tool call must round-trip: a Gemini functionCall with the name and the
	// fully-assembled args.
	var name, city string
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "data:"))
		if line == "" || line == "[DONE]" {
			continue
		}
		var c map[string]any
		if json.Unmarshal([]byte(line), &c) != nil {
			continue
		}
		cands, _ := c["candidates"].([]any)
		for _, ci := range cands {
			content, _ := ci.(map[string]any)["content"].(map[string]any)
			parts, _ := content["parts"].([]any)
			for _, p := range parts {
				fc, _ := p.(map[string]any)["functionCall"].(map[string]any)
				if fc == nil {
					continue
				}
				if n, ok := fc["name"].(string); ok {
					name = n
				}
				if args, ok := fc["args"].(map[string]any); ok {
					if cv, ok := args["city"].(string); ok {
						city = cv
					}
				}
			}
		}
	}
	if name != "get_weather" {
		t.Fatalf("functionCall name = %q, want get_weather\n%s", name, out)
	}
	if city != "Paris" {
		t.Fatalf("functionCall args.city = %q, want Paris\n%s", city, out)
	}
}

// TestGeminiCodex_ClientDisconnectUnwinds is the disconnection guardrail: when a
// streaming client drops mid-stream, the codex translation goroutine must unwind
// and CLOSE the upstream connection (no goroutine/connection leak). We prove it
// by blocking the codex backend on its request context and asserting that
// context becomes Done shortly after the client cancels — which only happens if
// the gate propagated the cancellation and closed the upstream body.
func TestGeminiCodex_ClientDisconnectUnwinds(t *testing.T) {
	upstreamCanceled := make(chan struct{}, 1)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fl, _ := w.(http.Flusher)
		// Emit one text delta so the gate produces a downstream Gemini chunk the
		// client can read, then block until the gate cancels the upstream.
		_, _ = io.WriteString(w, "event: x\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n")
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

	s := geminiCodexServer(t, backend.URL)
	gate := httptest.NewServer(s.Handler())
	defer gate.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		gate.URL+"/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
		strings.NewReader(`{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	// Read at least one byte to ensure the stream is live (gate goroutine active,
	// upstream connection open, blocked on the next event).
	buf := make([]byte, 1)
	if _, err := resp.Body.Read(buf); err != nil {
		t.Fatalf("read first byte: %v", err)
	}

	// Simulate a client disconnect mid-stream.
	cancel()
	_ = resp.Body.Close()

	select {
	case <-upstreamCanceled:
		// The upstream request context became Done → the gate's goroutine
		// unwound and closed the upstream body. No leak.
	case <-time.After(5 * time.Second):
		t.Fatal("upstream not canceled 5s after client disconnect — goroutine/connection leak")
	}
}
