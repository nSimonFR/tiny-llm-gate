package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
)

func TestExtractGeminiModel(t *testing.T) {
	cases := map[string]struct {
		in    string
		want  string
		ok    bool
	}{
		"simple":         {in: "/v1beta/models/gemma4:generateContent", want: "gemma4", ok: true},
		"colons in name": {in: "/v1beta/models/gpt-5.4:streamGenerateContent", want: "gpt-5.4", ok: true},
		"model with slash?": {in: "/v1beta/models/openai/gpt-5.4:generateContent", want: "openai/gpt-5.4", ok: true},
		"model with colon in id": {in: "/v1beta/models/gemma4:e4b:generateContent", want: "gemma4:e4b", ok: true},
		"no colon":  {in: "/v1beta/models/gemma4", want: "", ok: false},
		"wrong prefix": {in: "/foo/gemma4:x", want: "", ok: false},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			got, ok := extractGeminiModel(tc.in)
			if got != tc.want || ok != tc.ok {
				t.Errorf("got (%q,%v), want (%q,%v)", got, ok, tc.want, tc.ok)
			}
		})
	}
}

// newGeminiServer builds a server wired to one OpenAI-compat upstream.
func newGeminiServer(t *testing.T, upstream string, aliasFromGeminiName string) *Server {
	t.Helper()
	cfg := &config.Config{
		Providers: map[string]config.Provider{
			"ollama": {Type: "openai", BaseURL: upstream + "/v1"},
		},
		Models: map[string]config.Model{
			"gemma4": {Provider: "ollama", UpstreamModel: "gemma4:e4b"},
		},
		Aliases: map[string]string{aliasFromGeminiName: "gemma4"},
	}
	s, err := New(cfg, discardLogger())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s
}

func TestGenerateContentEndToEnd(t *testing.T) {
	upstream := newMockUpstream(200, `{
		"id":"x","object":"chat.completion","choices":[
			{"index":0,"message":{"role":"assistant","content":"Hello!"},"finish_reason":"stop"}
		],
		"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}
	}`)
	defer upstream.Close()

	s := newGeminiServer(t, upstream.URL, "gemini-2.5-flash")

	body := map[string]any{
		"contents": []map[string]any{
			{"role": "user", "parts": []map[string]any{{"text": "Hi"}}},
		},
	}
	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:generateContent", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	// The upstream received an OpenAI-shaped request with the correct model.
	if upstream.mu.requests.Load() != 1 {
		t.Fatalf("expected 1 upstream request, got %d", upstream.mu.requests.Load())
	}
	var sent map[string]any
	_ = json.Unmarshal([]byte(upstream.mu.bodies[0]), &sent)
	if sent["model"] != "gemma4:e4b" {
		t.Errorf("upstream model = %v; want gemma4:e4b", sent["model"])
	}
	msgs, _ := sent["messages"].([]any)
	if len(msgs) != 1 {
		t.Errorf("expected 1 message, got %d", len(msgs))
	}

	// Client received Gemini-shaped response.
	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("response not JSON: %v", err)
	}
	cands, ok := resp["candidates"].([]any)
	if !ok || len(cands) != 1 {
		t.Fatalf("candidates missing or wrong: %v", resp)
	}
}

func TestStreamGenerateContentEndToEnd(t *testing.T) {
	upstream := newMockUpstream(200, "")
	upstream.stream = []string{
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n",
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n",
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"}}]}\n\n",
		"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
		"data: [DONE]\n\n",
	}
	defer upstream.Close()

	s := newGeminiServer(t, upstream.URL, "gemini-2.5-flash")

	body := `{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:streamGenerateContent", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	// Output should be newline-delimited JSON chunks with only the content
	// and finish_reason deltas surfaced (the role-only delta is suppressed).
	lines := strings.Split(strings.TrimRight(rec.Body.String(), "\n"), "\n")
	if len(lines) < 2 {
		t.Fatalf("expected at least 2 chunks, got %d:\n%s", len(lines), rec.Body.String())
	}
	// Each line is a valid JSON Gemini chunk.
	for i, line := range lines {
		var c map[string]any
		if err := json.Unmarshal([]byte(line), &c); err != nil {
			t.Errorf("line %d not valid JSON: %v\n%s", i, err, line)
		}
	}
	// Assemble text from candidate parts; should concatenate to "Hi!".
	var text strings.Builder
	for _, line := range lines {
		var c map[string]any
		_ = json.Unmarshal([]byte(line), &c)
		cands, _ := c["candidates"].([]any)
		for _, ci := range cands {
			content, _ := ci.(map[string]any)["content"].(map[string]any)
			parts, _ := content["parts"].([]any)
			for _, p := range parts {
				if s, ok := p.(map[string]any)["text"].(string); ok {
					text.WriteString(s)
				}
			}
		}
	}
	if text.String() != "Hi!" {
		t.Errorf("assembled text = %q, want %q", text.String(), "Hi!")
	}
}

func TestEmbedContentEndToEnd(t *testing.T) {
	upstream := newMockUpstream(200, `{"data":[{"index":0,"embedding":[0.1,0.2,0.3]}]}`)
	defer upstream.Close()

	cfg := &config.Config{
		Providers: map[string]config.Provider{
			"ollama": {Type: "openai", BaseURL: upstream.URL + "/v1"},
		},
		Models: map[string]config.Model{
			"qwen3-embed": {Provider: "ollama", UpstreamModel: "qwen3-embedding:8b"},
		},
		Aliases: map[string]string{"gemini-embedding-001": "qwen3-embed"},
	}
	s, err := New(cfg, discardLogger())
	if err != nil {
		t.Fatal(err)
	}

	body := `{"content":{"parts":[{"text":"hello"}]}}`
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-embedding-001:embedContent", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	// Verify the upstream received an OpenAI /v1/embeddings request
	// with the rewritten model and a single input string.
	if upstream.mu.paths[0] != "/v1/embeddings" {
		t.Errorf("upstream path = %q", upstream.mu.paths[0])
	}
	var sent map[string]any
	_ = json.Unmarshal([]byte(upstream.mu.bodies[0]), &sent)
	if sent["model"] != "qwen3-embedding:8b" {
		t.Errorf("upstream model = %v", sent["model"])
	}

	var resp map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &resp)
	emb, _ := resp["embedding"].(map[string]any)
	vals, _ := emb["values"].([]any)
	if len(vals) != 3 || vals[0].(float64) != 0.1 {
		t.Errorf("bad embedding response: %v", resp)
	}
}

func TestBatchEmbedContentsEndToEnd(t *testing.T) {
	upstream := newMockUpstream(200, `{"data":[
		{"index":0,"embedding":[1]},
		{"index":1,"embedding":[2]},
		{"index":2,"embedding":[3]}
	]}`)
	defer upstream.Close()

	cfg := &config.Config{
		Providers: map[string]config.Provider{
			"ollama": {Type: "openai", BaseURL: upstream.URL + "/v1"},
		},
		Models: map[string]config.Model{
			"qwen3-embed": {Provider: "ollama", UpstreamModel: "qwen3-embedding:8b"},
		},
		Aliases: map[string]string{"gemini-embedding-001": "qwen3-embed"},
	}
	s, err := New(cfg, discardLogger())
	if err != nil {
		t.Fatal(err)
	}

	body := `{"requests":[
		{"content":{"parts":[{"text":"a"}]}},
		{"content":{"parts":[{"text":"b"}]}},
		{"content":{"parts":[{"text":"c"}]}}
	]}`
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-embedding-001:batchEmbedContents", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var resp map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &resp)
	embs, _ := resp["embeddings"].([]any)
	if len(embs) != 3 {
		t.Fatalf("expected 3 embeddings, got %d", len(embs))
	}
}

func TestGeminiUnknownAction(t *testing.T) {
	s := newGeminiServer(t, "http://unused", "gemini-2.5-flash")
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-2.5-flash:bogusAction", strings.NewReader(`{}`))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 404 {
		t.Errorf("expected 404 for unknown action, got %d", rec.Code)
	}
}

func TestGeminiUnknownModel(t *testing.T) {
	s := newGeminiServer(t, "http://unused", "gemini-2.5-flash")
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/does-not-exist:generateContent", strings.NewReader(`{"contents":[{"parts":[{"text":"x"}]}]}`))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 404 {
		t.Errorf("expected 404, got %d body=%s", rec.Code, rec.Body.String())
	}
}
