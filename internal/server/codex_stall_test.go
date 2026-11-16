package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
)

// stallingCodexBackend answers the first `stalls` requests with 200, then the
// given prefix, then silence until the gate cancels; later requests get `ok`.
func stallingCodexBackend(t *testing.T, stalls int32, prefix, ok string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.ReadAll(r.Body)
		n := hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		if n > stalls {
			_, _ = io.WriteString(w, ok)
			return
		}
		_, _ = io.WriteString(w, prefix)
		w.(http.Flusher).Flush()
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)
	return srv, &hits
}

func shortCodexTimeouts(t *testing.T) {
	t.Helper()
	first, idle := codexFirstByteTimeout, codexIdleTimeout
	codexFirstByteTimeout, codexIdleTimeout = 100*time.Millisecond, 100*time.Millisecond
	t.Cleanup(func() { codexFirstByteTimeout, codexIdleTimeout = first, idle })
}

func codexChat(t *testing.T, backendURL string, stream bool) *httptest.ResponseRecorder {
	t.Helper()
	creds := writeCodexCreds(t, t.TempDir())
	s := buildServer(t,
		map[string]config.Provider{
			"codex": {
				Type:    "codex",
				BaseURL: backendURL,
				Auth:    &config.Auth{Type: "oauth_chatgpt", File: creds, Issuer: "http://unused"},
			},
		},
		map[string]config.Model{"gpt-5.5": {Provider: "codex", UpstreamModel: "gpt-5.5"}},
		nil,
	)
	return postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "gpt-5.5",
		"stream":   stream,
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
}

var okSSE = codexSSE(
	`{"type":"response.output_text.delta","delta":"Hello"}`,
	`{"type":"response.completed","response":{"id":"r","usage":{"input_tokens":1,"output_tokens":1}}}`,
)

func TestCodexStall_SilentStreamRetriesTransparently(t *testing.T) {
	cases := map[string]string{
		"no byte at all":         "",
		"silent after created":   "event: x\ndata: {\"type\":\"response.created\",\"response\":{\"id\":\"r\"}}\n\n",
		"silent after keepalive": ": keepalive\n\n",
	}
	for name, prefix := range cases {
		t.Run(name, func(t *testing.T) {
			shortCodexTimeouts(t)
			backend, hits := stallingCodexBackend(t, 1, prefix, okSSE)

			rec := codexChat(t, backend.URL, true)
			body := rec.Body.String()
			if hits.Load() != 2 {
				t.Fatalf("backend hits = %d, want 2", hits.Load())
			}
			if !strings.Contains(body, `"content":"Hello"`) || !strings.Contains(body, "data: [DONE]") {
				t.Fatalf("retried stream incomplete:\n%s", body)
			}
			if n := strings.Count(body, `"role":"assistant"`); n != 1 {
				t.Fatalf("role chunks = %d, want 1:\n%s", n, body)
			}
		})
	}
}

func TestCodexStall_AfterContentEndsStreamWithoutRetry(t *testing.T) {
	shortCodexTimeouts(t)
	partial := "event: x\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"Hel\"}\n\n"
	backend, hits := stallingCodexBackend(t, 1, partial, okSSE)

	body := codexChat(t, backend.URL, true).Body.String()
	if hits.Load() != 1 {
		t.Fatalf("backend hits = %d, want 1 (no retry once content is out)", hits.Load())
	}
	if !strings.Contains(body, `"content":"Hel"`) || strings.Contains(body, "data: [DONE]") {
		t.Fatalf("want the partial content and no [DONE]:\n%s", body)
	}
}

func TestCodexStall_GivesUpAfterBoundedRetries(t *testing.T) {
	shortCodexTimeouts(t)
	backend, hits := stallingCodexBackend(t, 100, "", okSSE)

	body := codexChat(t, backend.URL, true).Body.String()
	if hits.Load() != 3 {
		t.Fatalf("backend hits = %d, want 3 (first try + 2 retries)", hits.Load())
	}
	if strings.Contains(body, "data: [DONE]") {
		t.Fatalf("a stream that never recovered must not end cleanly:\n%s", body)
	}
}

func TestCodexStall_NonStreamRetries(t *testing.T) {
	shortCodexTimeouts(t)
	backend, hits := stallingCodexBackend(t, 1, "", okSSE)

	rec := codexChat(t, backend.URL, false)
	if rec.Code != 200 || hits.Load() != 2 {
		t.Fatalf("status = %d, hits = %d; body = %s", rec.Code, hits.Load(), rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"content":"Hello"`) {
		t.Fatalf("non-stream body missing content: %s", rec.Body.String())
	}
}
