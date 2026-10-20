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

// Codex CLI's body for code-mode models: no top-level tools, everything in an
// additional_tools item, the shell as a custom grammar tool.
const codexCLIResponsesBody = `{
  "model": "gpt-6-luna", "stream": true, "store": false,
  "input": [
    {"type":"additional_tools","id":"at_1","role":"developer","tools":[
      {"type":"namespace","name":"functions","tools":[
        {"type":"custom","name":"exec","format":{"type":"grammar","syntax":"lark","definition":"start: /.+/"}}]}]},
    {"type":"message","role":"developer","content":[{"type":"input_text","text":"be codex"}]},
    {"type":"message","role":"user","content":[{"type":"input_text","text":"ls"}]},
    {"type":"custom_tool_call","call_id":"c1","name":"exec","input":"ls"},
    {"type":"custom_tool_call_output","call_id":"c1","output":"a b"}
  ],
  "tool_choice": "auto", "prompt_cache_key": "k"
}`

func codexResponsesServer(t *testing.T, backendURL string, models map[string]config.Model, extra map[string]config.Provider) *Server {
	t.Helper()
	providers := map[string]config.Provider{
		"codex": {Type: "codex", BaseURL: backendURL,
			Auth: &config.Auth{Type: "oauth_chatgpt", File: writeCodexCreds(t, t.TempDir()), Issuer: "http://unused"}},
	}
	for k, v := range extra {
		providers[k] = v
	}
	return buildServer(t, providers, models, nil)
}

func postRaw(t *testing.T, h http.Handler, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestCodexResponses_PassesThroughCodexCLIBody(t *testing.T) {
	upstreamSSE := codexSSE(
		`{"type":"response.output_item.done","item":{"type":"custom_tool_call","call_id":"c2","name":"exec","input":"pwd"}}`,
		`{"type":"response.completed","response":{"id":"r","output":[]}}`,
	)
	backend := newMockCodexBackend(t, upstreamSSE)
	defer backend.Close()
	s := codexResponsesServer(t, backend.URL, map[string]config.Model{
		"luna": {Provider: "codex", UpstreamModel: "gpt-6-luna"},
	}, nil)

	rec := postRaw(t, s.Handler(), "/v1/responses", strings.Replace(codexCLIResponsesBody, `"gpt-6-luna"`, `"luna"`, 1))
	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var sent map[string]json.RawMessage
	if err := json.Unmarshal([]byte(backend.gotBody), &sent); err != nil {
		t.Fatal(err)
	}
	if string(sent["model"]) != `"gpt-6-luna"` {
		t.Errorf("model = %s", sent["model"])
	}
	if _, ok := sent["instructions"]; ok {
		t.Error("instructions added although input carries a developer message")
	}
	var want, got any
	_ = json.Unmarshal([]byte(codexCLIResponsesBody), &want)
	_ = json.Unmarshal(sent["input"], &got)
	wantInput, _ := json.Marshal(want.(map[string]any)["input"])
	gotInput, _ := json.Marshal(got)
	if string(wantInput) != string(gotInput) {
		t.Errorf("input changed:\n got %s\nwant %s", gotInput, wantInput)
	}
	if string(sent["prompt_cache_key"]) != `"k"` {
		t.Errorf("prompt_cache_key = %s", sent["prompt_cache_key"])
	}
	if backend.gotHeaders.Get("ChatGPT-Account-Id") != "acct_test" {
		t.Errorf("auth not applied: %v", backend.gotHeaders)
	}
	if rec.Body.String() != upstreamSSE {
		t.Errorf("SSE not relayed verbatim:\n%s", rec.Body.String())
	}
}

func TestCodexResponses_NonStreamCollectsOutput(t *testing.T) {
	backend := newMockCodexBackend(t, codexSSE(
		`{"type":"response.output_text.delta","delta":"hi"}`,
		`{"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}}`,
		`{"type":"response.completed","response":{"id":"r","object":"response","output":[],"usage":{"input_tokens":3,"output_tokens":1}}}`,
	))
	defer backend.Close()
	s := codexResponsesServer(t, backend.URL, map[string]config.Model{
		"gpt-6": {Provider: "codex", UpstreamModel: "gpt-6"},
	}, nil)

	rec := postRaw(t, s.Handler(), "/v1/responses", `{"model":"gpt-6","input":"hello","temperature":0.2}`)
	if rec.Code != 200 {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var sent map[string]json.RawMessage
	_ = json.Unmarshal([]byte(backend.gotBody), &sent)
	if string(sent["stream"]) != "true" || string(sent["store"]) != "false" {
		t.Errorf("stream/store not forced: %s", backend.gotBody)
	}
	if _, ok := sent["temperature"]; ok {
		t.Error("temperature forwarded")
	}
	if !strings.Contains(string(sent["input"]), `"role":"user"`) || sent["instructions"] == nil {
		t.Errorf("string input not normalized: %s", backend.gotBody)
	}
	var out struct {
		ID     string            `json:"id"`
		Output []json.RawMessage `json:"output"`
		Usage  map[string]int    `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("bad body %s: %v", rec.Body.String(), err)
	}
	if out.ID != "r" || len(out.Output) != 1 || out.Usage["output_tokens"] != 1 {
		t.Errorf("collected response = %s", rec.Body.String())
	}
}

func TestCodexResponses_FallsBackToTranslatedHop(t *testing.T) {
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		_, _ = io.WriteString(w, "boom")
	}))
	defer bad.Close()
	fallback := newMockUpstream(200, `{"id":"x","choices":[{"message":{"content":"from-fallback"}}]}`)
	defer fallback.Close()
	s := codexResponsesServer(t, bad.URL, map[string]config.Model{
		"luna":  {Provider: "codex", UpstreamModel: "gpt-6-luna", Fallback: []string{"gemma"}},
		"gemma": {Provider: "ollama", UpstreamModel: "gemma4:e4b"},
	}, map[string]config.Provider{
		"ollama": {Type: "openai", BaseURL: fallback.URL + "/v1", APIKey: "k"},
	})

	rec := postRaw(t, s.Handler(), "/v1/responses", strings.Replace(codexCLIResponsesBody, `"gpt-6-luna"`, `"luna"`, 1))
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "from-fallback") {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
}

func TestCodexResponses_ForwardsUpstreamError(t *testing.T) {
	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(400)
		_, _ = io.WriteString(w, `{"error":{"message":"bad tool"}}`)
	}))
	defer bad.Close()
	s := codexResponsesServer(t, bad.URL, map[string]config.Model{
		"gpt-6": {Provider: "codex", UpstreamModel: "gpt-6"},
	}, nil)
	rec := postRaw(t, s.Handler(), "/v1/responses", `{"model":"gpt-6","stream":true,"input":"x"}`)
	if rec.Code != 400 || !strings.Contains(rec.Body.String(), "bad tool") {
		t.Fatalf("status = %d, body = %s", rec.Code, rec.Body.String())
	}
}
