package server

import (
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
)

// mockCodexBackend stands in for chatgpt.com/backend-api/codex. It records the
// request it received and replies with a canned Codex Responses SSE stream.
type mockCodexBackend struct {
	*httptest.Server
	gotBody    string
	gotHeaders http.Header
	gotPath    string
}

func newMockCodexBackend(t *testing.T, sseBody string) *mockCodexBackend {
	t.Helper()
	m := &mockCodexBackend{}
	m.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		m.gotBody = string(b)
		m.gotHeaders = r.Header.Clone()
		m.gotPath = r.URL.Path
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		_, _ = io.WriteString(w, sseBody)
	}))
	return m
}

// writeCodexCreds writes an oauth_chatgpt credentials file with a long-lived
// access token so no refresh is attempted during the test.
func writeCodexCreds(t *testing.T, dir string) string {
	t.Helper()
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none"}`))
	claims, _ := json.Marshal(map[string]any{
		"exp": time.Now().Add(time.Hour).Unix(),
		"https://api.openai.com/auth": map[string]any{
			"chatgpt_account_id": "acct_test",
		},
	})
	tok := header + "." + base64.RawURLEncoding.EncodeToString(claims) + ".sig"
	p := filepath.Join(dir, "creds.json")
	b, _ := json.Marshal(map[string]any{"access_token": tok, "refresh_token": "rt"})
	if err := os.WriteFile(p, b, 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func codexSSE(events ...string) string {
	var sb strings.Builder
	for _, e := range events {
		sb.WriteString("event: x\ndata: ")
		sb.WriteString(e)
		sb.WriteString("\n\n")
	}
	sb.WriteString("data: [DONE]\n\n")
	return sb.String()
}

func TestCodexProvider_StreamingEndToEnd(t *testing.T) {
	backend := newMockCodexBackend(t, codexSSE(
		`{"type":"response.output_text.delta","delta":"Hello"}`,
		`{"type":"response.output_text.delta","delta":" world"}`,
		`{"type":"response.completed","response":{"id":"r","usage":{"input_tokens":8,"output_tokens":2}}}`,
	))
	defer backend.Close()

	creds := writeCodexCreds(t, t.TempDir())
	s := buildServer(t,
		map[string]config.Provider{
			"codex": {
				Type:    "codex",
				BaseURL: backend.URL,
				Auth:    &config.Auth{Type: "oauth_chatgpt", File: creds, Issuer: "http://unused"},
			},
		},
		map[string]config.Model{
			"gpt-5.5": {Provider: "codex", UpstreamModel: "gpt-5.5"},
		},
		nil,
	)

	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "gpt-5.5",
		"stream":   true,
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}

	// Backend must have been hit at the /responses path with a translated body.
	if !strings.HasSuffix(backend.gotPath, "/responses") {
		t.Fatalf("backend path = %q", backend.gotPath)
	}
	var sent map[string]any
	if err := json.Unmarshal([]byte(backend.gotBody), &sent); err != nil {
		t.Fatalf("backend body not JSON: %v\n%s", err, backend.gotBody)
	}
	if sent["model"] != "gpt-5.5" || sent["store"] != false || sent["instructions"] == nil {
		t.Fatalf("translated body missing codex fields: %v", sent)
	}
	// Auth + fingerprint headers applied.
	if backend.gotHeaders.Get("Authorization") == "" {
		t.Error("missing Authorization header")
	}
	if backend.gotHeaders.Get("ChatGPT-Account-Id") != "acct_test" {
		t.Errorf("ChatGPT-Account-Id = %q", backend.gotHeaders.Get("ChatGPT-Account-Id"))
	}
	if backend.gotHeaders.Get("originator") != codexOriginator {
		t.Errorf("originator = %q", backend.gotHeaders.Get("originator"))
	}

	// Client gets OpenAI-shaped SSE with the concatenated content + finish.
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("content-type = %q", ct)
	}
	body := rec.Body.String()
	var content, finish string
	for _, block := range strings.Split(body, "\n\n") {
		block = strings.TrimSpace(block)
		if !strings.HasPrefix(block, "data:") {
			continue
		}
		payload := strings.TrimSpace(block[len("data:"):])
		if payload == "[DONE]" || payload == "" {
			continue
		}
		var c struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
				FinishReason *string `json:"finish_reason"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &c); err != nil {
			t.Fatalf("bad client chunk %q: %v", payload, err)
		}
		content += c.Choices[0].Delta.Content
		if c.Choices[0].FinishReason != nil {
			finish = *c.Choices[0].FinishReason
		}
	}
	if content != "Hello world" {
		t.Fatalf("client content = %q", content)
	}
	if finish != "stop" {
		t.Fatalf("finish = %q", finish)
	}
}

func TestCodexProvider_NonStreamingAggregates(t *testing.T) {
	backend := newMockCodexBackend(t, codexSSE(
		`{"type":"response.output_text.delta","delta":"Answer"}`,
		`{"type":"response.completed","response":{"id":"r","usage":{"input_tokens":3,"output_tokens":1}}}`,
	))
	defer backend.Close()

	creds := writeCodexCreds(t, t.TempDir())
	s := buildServer(t,
		map[string]config.Provider{
			"codex": {Type: "codex", BaseURL: backend.URL, Auth: &config.Auth{Type: "oauth_chatgpt", File: creds, Issuer: "http://unused"}},
		},
		map[string]config.Model{"gpt-5.5": {Provider: "codex", UpstreamModel: "gpt-5.5"}},
		nil,
	)

	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "gpt-5.5",
		"messages": []map[string]any{{"role": "user", "content": "q"}},
	})
	if rec.Code != 200 {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("content-type = %q", ct)
	}
	var resp struct {
		Object  string `json:"object"`
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("client response not JSON: %v\n%s", err, rec.Body.String())
	}
	if resp.Object != "chat.completion" || resp.Choices[0].Message.Content != "Answer" {
		t.Fatalf("resp = %+v", resp)
	}
	if resp.Usage.TotalTokens != 4 {
		t.Fatalf("total tokens = %d", resp.Usage.TotalTokens)
	}
}

// A codex hop that 5xxes should fall back to a healthy OpenAI provider.
func TestCodexProvider_FallsBackOn5xx(t *testing.T) {
	badCodex := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		_, _ = io.WriteString(w, "upstream boom")
	}))
	defer badCodex.Close()
	fallback := newMockUpstream(200, `{"id":"x","choices":[{"message":{"content":"from-fallback"}}]}`)
	defer fallback.Close()

	creds := writeCodexCreds(t, t.TempDir())
	s := buildServer(t,
		map[string]config.Provider{
			"codex":  {Type: "codex", BaseURL: badCodex.URL, Auth: &config.Auth{Type: "oauth_chatgpt", File: creds, Issuer: "http://unused"}},
			"ollama": {Type: "openai", BaseURL: fallback.URL + "/v1", APIKey: "k"},
		},
		map[string]config.Model{
			"gpt-5.5": {Provider: "codex", UpstreamModel: "gpt-5.5", Fallback: []string{"gemma"}},
			"gemma":   {Provider: "ollama", UpstreamModel: "gemma4:e4b"},
		},
		nil,
	)

	rec := postJSON(t, s.Handler(), "/v1/chat/completions", map[string]any{
		"model":    "gpt-5.5",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 200 {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "from-fallback") {
		t.Fatalf("expected fallback response, got %s", rec.Body.String())
	}
}
