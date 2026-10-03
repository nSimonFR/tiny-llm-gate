package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
)

// recordingUpstream answers /v1/chat/completions with a fixed body and keeps
// every request body it saw.
type recordingUpstream struct {
	*httptest.Server
	mu     sync.Mutex
	bodies []map[string]any
}

func newRecordingUpstream(t *testing.T, status int, contentType, body string) *recordingUpstream {
	t.Helper()
	u := &recordingUpstream{}
	u.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		var m map[string]any
		_ = json.Unmarshal(b, &m)
		u.mu.Lock()
		u.bodies = append(u.bodies, m)
		u.mu.Unlock()
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(u.Close)
	return u
}

func (u *recordingUpstream) last(t *testing.T) map[string]any {
	t.Helper()
	u.mu.Lock()
	defer u.mu.Unlock()
	if len(u.bodies) == 0 {
		t.Fatal("upstream saw no request")
	}
	return u.bodies[len(u.bodies)-1]
}

func sseChunks(chunks ...string) string {
	var b strings.Builder
	for _, c := range chunks {
		b.WriteString("data: " + c + "\n\n")
	}
	b.WriteString("data: [DONE]\n\n")
	return b.String()
}

const okChat = `{"id":"x","choices":[{"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}`

func orServer(t *testing.T, upstreamURL string, keys []config.ClientKey) *Server {
	t.Helper()
	cfg := &config.Config{
		Listen: "127.0.0.1:0",
		Providers: map[string]config.Provider{
			"ollama": {Type: "openai", BaseURL: upstreamURL + "/v1"},
		},
		Models: map[string]config.Model{
			"gemma4": {Provider: "ollama", UpstreamModel: "gemma4:e4b"},
			"qwen": {Provider: "ollama", UpstreamModel: "qwen3:8b", Info: &config.ModelInfo{
				Vendor: "qwen", Name: "Qwen 3 8B", ContextLength: 32768, MaxOutputTokens: 8192,
				InputModalities: []string{"text", "image"},
				Pricing:         &config.Pricing{Prompt: "0.0000001", Completion: "0.0000002"},
			}},
		},
		Aliases:    map[string]string{"gpt-4o": "gemma4"},
		ClientKeys: keys,
	}
	s, err := New(cfg, discardLogger())
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	return s
}

func do(t *testing.T, h http.Handler, method, path, body string, hdr map[string]string) *httptest.ResponseRecorder {
	t.Helper()
	var rd io.Reader
	if body != "" {
		rd = strings.NewReader(body)
	}
	req := httptest.NewRequest(method, path, rd)
	req.Header.Set("Content-Type", "application/json")
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

// ── client keys ──────────────────────────────────────────────────

func TestClientKeys(t *testing.T) {
	up := newRecordingUpstream(t, 200, "application/json", okChat)
	h := orServer(t, up.URL, []config.ClientKey{
		{Name: "full", Key: "k-full"},
		{Name: "narrow", Key: "k-narrow", Models: []string{"gpt-4o", "qwen/*"}},
	}).Handler()
	chat := func(model string) string {
		return `{"model":"` + model + `","messages":[{"role":"user","content":"x"}]}`
	}

	cases := []struct {
		name  string
		hdr   map[string]string
		model string
		want  int
	}{
		{"no key", nil, "gemma4", 401},
		{"bad key", map[string]string{"Authorization": "Bearer nope"}, "gemma4", 401},
		{"bearer", map[string]string{"Authorization": "Bearer k-full"}, "gemma4", 200},
		{"x-api-key", map[string]string{"x-api-key": "k-full"}, "gemma4", 200},
		{"alias allowed by name", map[string]string{"Authorization": "Bearer k-narrow"}, "gpt-4o", 200},
		{"canonical of allowed alias", map[string]string{"Authorization": "Bearer k-narrow"}, "gemma4", 200},
		{"glob on vendor slug", map[string]string{"Authorization": "Bearer k-narrow"}, "qwen", 200},
		{"outside allowlist", map[string]string{"Authorization": "Bearer k-narrow"}, "ollama/gemma4x", 404},
	}
	for _, c := range cases {
		rec := do(t, h, "POST", "/v1/chat/completions", chat(c.model), c.hdr)
		if rec.Code != c.want {
			t.Errorf("%s: status %d, want %d (%s)", c.name, rec.Code, c.want, rec.Body)
		}
	}
}

func TestClientKeyRPMWindow(t *testing.T) {
	ck := &clientKey{rpm: 2}
	t0 := time.Unix(600, 0) // start of a minute
	for i, want := range []bool{true, true, false} {
		if _, ok := ck.take(t0.Add(time.Duration(i) * time.Second)); ok != want {
			t.Fatalf("request %d: ok=%v want %v", i, ok, want)
		}
	}
	if after, _ := ck.take(t0.Add(15 * time.Second)); after != 45 {
		t.Errorf("Retry-After = %d, want 45", after)
	}
	if _, ok := ck.take(t0.Add(time.Minute)); !ok {
		t.Error("next window should reset the count")
	}
}

func TestRateLimitedRequestGets429(t *testing.T) {
	up := newRecordingUpstream(t, 200, "application/json", okChat)
	s := orServer(t, up.URL, []config.ClientKey{{Name: "slow", Key: "k", RPM: 1}})
	for _, ck := range s.clientKeys {
		ck.window, ck.count = time.Now().Unix()/60, 1 // window already spent
	}
	rec := do(t, s.Handler(), "POST", "/v1/chat/completions", `{"model":"gemma4","messages":[]}`, map[string]string{"Authorization": "Bearer k"})
	if rec.Code != 429 || rec.Header().Get("Retry-After") == "" {
		t.Fatalf("status %d, Retry-After %q", rec.Code, rec.Header().Get("Retry-After"))
	}
}

func TestClientKeyAllowlistDenies(t *testing.T) {
	up := newRecordingUpstream(t, 200, "application/json", okChat)
	h := orServer(t, up.URL, []config.ClientKey{{Name: "narrow", Key: "k", Models: []string{"qwen/*"}}}).Handler()
	hdr := map[string]string{"Authorization": "Bearer k"}
	if rec := do(t, h, "POST", "/v1/chat/completions", `{"model":"gemma4","messages":[]}`, hdr); rec.Code != 403 {
		t.Fatalf("status %d, want 403", rec.Code)
	}
	if rec := do(t, h, "POST", "/v1/chat/completions", `{"model":"qwen/qwen","messages":[]}`, hdr); rec.Code != 200 {
		t.Fatalf("slug: status %d, want 200 (%s)", rec.Code, rec.Body)
	}
	rec := do(t, h, "GET", "/api/v1/models", "", hdr)
	var list struct {
		Data []struct{ ID string } `json:"data"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &list)
	var ids []string
	for _, m := range list.Data {
		ids = append(ids, m.ID)
	}
	if strings.Join(ids, ",") != "qwen,qwen/qwen" {
		t.Errorf("filtered models = %v", ids)
	}
}

func TestKeyEndpoint(t *testing.T) {
	up := newRecordingUpstream(t, 200, "application/json", okChat)
	h := orServer(t, up.URL, []config.ClientKey{{Name: "app", Key: "k", RPM: 30}}).Handler()
	rec := do(t, h, "GET", "/api/v1/key", "", map[string]string{"Authorization": "Bearer k"})
	var out struct {
		Data struct {
			Label     string `json:"label"`
			RateLimit struct {
				Requests int `json:"requests"`
			} `json:"rate_limit"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil || out.Data.Label != "app" || out.Data.RateLimit.Requests != 30 {
		t.Fatalf("key = %s", rec.Body)
	}
	if rec := do(t, h, "GET", "/api/v1/credits", "", map[string]string{"Authorization": "Bearer k"}); rec.Code != 200 {
		t.Fatalf("credits status %d", rec.Code)
	}
}

// ── OpenRouter request dialect ───────────────────────────────────

func TestVendorSlugResolves(t *testing.T) {
	up := newRecordingUpstream(t, 200, "application/json", okChat)
	h := orServer(t, up.URL, nil).Handler()
	for slug, upstream := range map[string]string{"ollama/gemma4": "gemma4:e4b", "qwen/qwen": "qwen3:8b", "ollama/qwen": "qwen3:8b"} {
		rec := do(t, h, "POST", "/api/v1/chat/completions", `{"model":"`+slug+`","messages":[]}`, nil)
		if rec.Code != 200 {
			t.Fatalf("%s: status %d", slug, rec.Code)
		}
		if got := up.last(t)["model"]; got != upstream {
			t.Errorf("%s → upstream model %v, want %s", slug, got, upstream)
		}
	}
}

func TestOpenRouterFieldsNormalized(t *testing.T) {
	up := newRecordingUpstream(t, 200, "application/json", okChat)
	h := orServer(t, up.URL, nil).Handler()
	rec := do(t, h, "POST", "/api/v1/chat/completions", `{"model":"gemma4","messages":[],"stream":true,
		"reasoning":{"max_tokens":2000},"provider":{"order":["x"]},"transforms":["middle-out"],
		"usage":{"include":true},"route":"fallback"}`, nil)
	if rec.Code != 200 {
		t.Fatalf("status %d", rec.Code)
	}
	got := up.last(t)
	for _, k := range openRouterOnly {
		if _, ok := got[k]; ok {
			t.Errorf("%q leaked upstream", k)
		}
	}
	if got["reasoning_effort"] != "low" {
		t.Errorf("reasoning_effort = %v", got["reasoning_effort"])
	}
	if so, _ := got["stream_options"].(map[string]any); so["include_usage"] != true {
		t.Errorf("stream_options = %v", got["stream_options"])
	}
}

func TestPlainBodyForwardedByteStable(t *testing.T) {
	in := []byte(`{"z":1,"model":"gemma4","a":2}`)
	out, models, err := normalizeOpenRouter(in)
	if err != nil || models != nil || !bytes.Equal(in, out) {
		t.Fatalf("plain body changed: %s", out)
	}
}

func TestModelsParamIsFallbackChain(t *testing.T) {
	var mu sync.Mutex
	var seen []string
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var m map[string]any
		_ = json.NewDecoder(r.Body).Decode(&m)
		mu.Lock()
		seen = append(seen, m["model"].(string))
		mu.Unlock()
		if m["model"] == "gemma4:e4b" {
			w.WriteHeader(503)
			return
		}
		_, _ = io.WriteString(w, okChat)
	}))
	defer up.Close()
	h := orServer(t, up.URL, nil).Handler()
	rec := do(t, h, "POST", "/api/v1/chat/completions", `{"models":["gemma4","qwen/qwen"],"messages":[]}`, nil)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if strings.Join(seen, ",") != "gemma4:e4b,qwen3:8b" {
		t.Errorf("hops = %v", seen)
	}
}

func TestCatalogShape(t *testing.T) {
	up := newRecordingUpstream(t, 200, "application/json", okChat)
	h := orServer(t, up.URL, nil).Handler()
	rec := do(t, h, "GET", "/api/v1/models/qwen/qwen", "", nil)
	if rec.Code != 200 {
		t.Fatalf("status %d", rec.Code)
	}
	var m orModel
	_ = json.Unmarshal(rec.Body.Bytes(), &m)
	if m.ID != "qwen/qwen" || m.CanonicalSlug != "qwen/qwen" || m.Name != "Qwen 3 8B" ||
		m.ContextLength == nil || *m.ContextLength != 32768 ||
		m.TopProvider.MaxCompletionTokens == nil || *m.TopProvider.MaxCompletionTokens != 8192 ||
		m.Architecture.Modality != "text+image->text" || m.Pricing.Prompt != "0.0000001" || m.OwnedBy != "ollama" {
		t.Errorf("entry = %+v", m)
	}
	rec = do(t, h, "GET", "/v1/models?supported_parameters=tools,reasoning", "", nil)
	if !strings.Contains(rec.Body.String(), `"gemma4"`) {
		t.Errorf("default params should include tools,reasoning: %s", rec.Body)
	}
	if rec := do(t, h, "GET", "/v1/models?supported_parameters=seed", "", nil); strings.Contains(rec.Body.String(), `"id"`) {
		t.Errorf("seed filter should match nothing: %s", rec.Body)
	}
}

// ── /v1/responses ────────────────────────────────────────────────

func TestResponsesNonStream(t *testing.T) {
	up := newRecordingUpstream(t, 200, "text/event-stream", sseChunks(
		`{"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}`,
		`{"choices":[{"delta":{"content":"lo"}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"get","arguments":"{\"a\""}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":1}"}}]},"finish_reason":"tool_calls"}]}`,
		`{"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`,
	))
	h := orServer(t, up.URL, nil).Handler()
	rec := do(t, h, "POST", "/v1/responses", `{"model":"gpt-4o","instructions":"be brief",
		"input":[{"role":"user","content":[{"type":"input_text","text":"hi"}]},
		  {"type":"function_call","call_id":"c0","name":"get","arguments":"{}"},
		  {"type":"function_call_output","call_id":"c0","output":"42"}],
		"tools":[{"type":"function","name":"get","parameters":{"type":"object"}}],
		"tool_choice":{"type":"function","name":"get"},"reasoning":{"effort":"low"},"max_output_tokens":99}`, nil)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var out struct {
		Object string `json:"object"`
		Status string `json:"status"`
		Model  string `json:"model"`
		Output []struct {
			Type      string                  `json:"type"`
			Content   []struct{ Text string } `json:"content"`
			CallID    string                  `json:"call_id"`
			Name      string                  `json:"name"`
			Arguments string                  `json:"arguments"`
		} `json:"output"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out.Object != "response" || out.Status != "completed" || out.Model != "gpt-4o" || len(out.Output) != 2 ||
		out.Output[0].Content[0].Text != "Hello" || out.Output[1].CallID != "call_1" ||
		out.Output[1].Arguments != `{"a":1}` || out.Usage.InputTokens != 10 || out.Usage.OutputTokens != 5 {
		t.Errorf("response = %s", rec.Body)
	}

	got := up.last(t)
	msgs := got["messages"].([]any)
	if len(msgs) != 4 || msgs[0].(map[string]any)["role"] != "system" ||
		msgs[1].(map[string]any)["content"] != "hi" ||
		msgs[2].(map[string]any)["tool_calls"] == nil || msgs[3].(map[string]any)["tool_call_id"] != "c0" {
		t.Errorf("chat messages = %v", msgs)
	}
	if got["reasoning_effort"] != "low" || got["max_tokens"] != float64(99) || got["model"] != "gemma4:e4b" {
		t.Errorf("chat body = %v", got)
	}
	if tc := got["tool_choice"].(map[string]any); tc["function"].(map[string]any)["name"] != "get" {
		t.Errorf("tool_choice = %v", tc)
	}
}

func TestResponsesStreamEvents(t *testing.T) {
	up := newRecordingUpstream(t, 200, "text/event-stream", sseChunks(
		`{"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}`,
	))
	h := orServer(t, up.URL, nil).Handler()
	rec := do(t, h, "POST", "/api/v1/responses", `{"model":"gemma4","input":"yo","stream":true}`, nil)
	var types []string
	for _, line := range strings.Split(rec.Body.String(), "\n") {
		if strings.HasPrefix(line, "event: ") {
			types = append(types, strings.TrimPrefix(line, "event: "))
		}
	}
	want := "response.created,response.in_progress,response.output_item.added,response.content_part.added," +
		"response.output_text.delta,response.output_text.done,response.content_part.done," +
		"response.output_item.done,response.completed"
	if strings.Join(types, ",") != want {
		t.Errorf("events:\n got %v\nwant %s", types, want)
	}
}

func TestResponsesRejectsPreviousResponseID(t *testing.T) {
	up := newRecordingUpstream(t, 200, "application/json", okChat)
	h := orServer(t, up.URL, nil).Handler()
	if rec := do(t, h, "POST", "/v1/responses", `{"model":"gemma4","input":"x","previous_response_id":"resp_1"}`, nil); rec.Code != 400 {
		t.Fatalf("status %d", rec.Code)
	}
}

// ── /v1/messages → non-Anthropic models ──────────────────────────

func TestMessagesTranslatedNonStream(t *testing.T) {
	up := newRecordingUpstream(t, 200, "text/event-stream", sseChunks(
		`{"choices":[{"delta":{"content":"ok"}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_9","function":{"name":"ls","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`,
		`{"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}`,
	))
	h := orServer(t, up.URL, nil).Handler() // no anthropic block configured
	rec := do(t, h, "POST", "/v1/messages", `{"model":"qwen/qwen","max_tokens":50,"system":[{"type":"text","text":"sys"}],
		"messages":[{"role":"user","content":"list"},
		  {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"ls","input":{"p":"/"}}]},
		  {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"a b"},{"type":"text","text":"again"}]}],
		"tools":[{"name":"ls","input_schema":{"type":"object"}}],"tool_choice":{"type":"any"}}`, nil)
	if rec.Code != 200 {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var out struct {
		Type       string `json:"type"`
		StopReason string `json:"stop_reason"`
		Content    []struct {
			Type  string          `json:"type"`
			Text  string          `json:"text"`
			ID    string          `json:"id"`
			Input json.RawMessage `json:"input"`
		} `json:"content"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	if out.Type != "message" || out.StopReason != "tool_use" || len(out.Content) != 2 ||
		out.Content[0].Text != "ok" || out.Content[1].ID != "call_9" || string(out.Content[1].Input) != "{}" ||
		out.Usage.InputTokens != 7 || out.Usage.OutputTokens != 2 {
		t.Errorf("message = %s", rec.Body)
	}
	got := up.last(t)
	msgs := got["messages"].([]any)
	roles := []string{}
	for _, m := range msgs {
		roles = append(roles, m.(map[string]any)["role"].(string))
	}
	if strings.Join(roles, ",") != "system,user,assistant,tool,user" {
		t.Errorf("roles = %v", roles)
	}
	if got["tool_choice"] != "required" || got["model"] != "qwen3:8b" {
		t.Errorf("chat body = %v", got)
	}
}

func TestMessagesTranslatedStream(t *testing.T) {
	up := newRecordingUpstream(t, 200, "text/event-stream", sseChunks(
		`{"choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}`,
	))
	h := orServer(t, up.URL, nil).Handler()
	rec := do(t, h, "POST", "/api/v1/messages", `{"model":"gemma4","max_tokens":5,"stream":true,"messages":[{"role":"user","content":"x"}]}`, nil)
	var types []string
	for _, line := range strings.Split(rec.Body.String(), "\n") {
		if strings.HasPrefix(line, "event: ") {
			types = append(types, strings.TrimPrefix(line, "event: "))
		}
	}
	want := "message_start,content_block_start,content_block_delta,content_block_stop,message_delta,message_stop"
	if strings.Join(types, ",") != want {
		t.Errorf("events = %v", types)
	}
}

func TestMessagesUnknownModelWithoutAnthropic(t *testing.T) {
	up := newRecordingUpstream(t, 200, "application/json", okChat)
	h := orServer(t, up.URL, nil).Handler()
	rec := do(t, h, "POST", "/v1/messages", `{"model":"claude-opus-4-8","messages":[]}`, nil)
	if rec.Code != 404 || !strings.Contains(rec.Body.String(), "not_found_error") {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
}
