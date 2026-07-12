package codex

import (
	"encoding/json"
	"testing"
)

// decodeCodex unmarshals a translated request for assertions.
func decodeCodex(t *testing.T, body []byte) codexRequest {
	t.Helper()
	var r codexRequest
	if err := json.Unmarshal(body, &r); err != nil {
		t.Fatalf("unmarshal codex request: %v\n%s", err, body)
	}
	return r
}

func TestTranslateRequest_SystemBecomesInstructions(t *testing.T) {
	in := `{"model":"gpt-5.5","messages":[
		{"role":"system","content":"be terse"},
		{"role":"developer","content":"prefer json"},
		{"role":"user","content":"hi"}
	]}`
	out, err := TranslateRequest([]byte(in), "gpt-5.5")
	if err != nil {
		t.Fatal(err)
	}
	r := decodeCodex(t, out)
	if r.Instructions != "be terse\n\nprefer json" {
		t.Fatalf("instructions = %q", r.Instructions)
	}
	if !r.Stream || r.Store {
		t.Fatalf("expected stream=true store=false, got stream=%v store=%v", r.Stream, r.Store)
	}
	if len(r.Input) != 1 || r.Input[0].Role != "user" {
		t.Fatalf("input = %+v", r.Input)
	}
	if s, _ := r.Input[0].Content.(string); s != "hi" {
		t.Fatalf("user content = %v", r.Input[0].Content)
	}
}

func TestTranslateRequest_DefaultInstructions(t *testing.T) {
	out, err := TranslateRequest([]byte(`{"model":"m","messages":[{"role":"user","content":"x"}]}`), "up")
	if err != nil {
		t.Fatal(err)
	}
	if r := decodeCodex(t, out); r.Instructions != defaultInstructions {
		t.Fatalf("instructions = %q", r.Instructions)
	}
}

func TestTranslateRequest_UpstreamModelOverride(t *testing.T) {
	out, err := TranslateRequest([]byte(`{"model":"client-name","messages":[{"role":"user","content":"x"}]}`), "gpt-5.5-upstream")
	if err != nil {
		t.Fatal(err)
	}
	if r := decodeCodex(t, out); r.Model != "gpt-5.5-upstream" {
		t.Fatalf("model = %q", r.Model)
	}
}

func TestTranslateRequest_AssistantToolCallsAndResult(t *testing.T) {
	in := `{"model":"m","messages":[
		{"role":"user","content":"weather?"},
		{"role":"assistant","content":"","tool_calls":[
			{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}
		]},
		{"role":"tool","tool_call_id":"call_1","content":"18C"}
	]}`
	out, err := TranslateRequest([]byte(in), "m")
	if err != nil {
		t.Fatal(err)
	}
	r := decodeCodex(t, out)
	// Expect: user msg, function_call, function_call_output. Assistant with
	// empty text + tool calls should not emit an empty assistant message.
	var fc, fco *codexInput
	for i := range r.Input {
		switch r.Input[i].Type {
		case "function_call":
			fc = &r.Input[i]
		case "function_call_output":
			fco = &r.Input[i]
		}
	}
	if fc == nil {
		t.Fatal("missing function_call item")
	}
	if fc.CallID != "call_1" || fc.Name != "get_weather" || fc.Arguments != `{"city":"Paris"}` {
		t.Fatalf("function_call = %+v", fc)
	}
	if fco == nil {
		t.Fatal("missing function_call_output item")
	}
	if fco.CallID != "call_1" || fco.Output != "18C" {
		t.Fatalf("function_call_output = %+v", fco)
	}
}

func TestTranslateRequest_ToolsAndToolChoice(t *testing.T) {
	in := `{"model":"m","messages":[{"role":"user","content":"x"}],
		"tools":[{"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object"},"strict":true}}],
		"tool_choice":{"type":"function","function":{"name":"f"}}}`
	out, err := TranslateRequest([]byte(in), "m")
	if err != nil {
		t.Fatal(err)
	}
	r := decodeCodex(t, out)
	if len(r.Tools) != 1 {
		t.Fatalf("tools = %+v", r.Tools)
	}
	tl := r.Tools[0]
	if tl.Type != "function" || tl.Name != "f" || tl.Description != "d" || !tl.Strict {
		t.Fatalf("tool = %+v", tl)
	}
	// normalizeSchema must have injected properties:{} into the object schema.
	var params map[string]any
	if err := json.Unmarshal(tl.Parameters, &params); err != nil {
		t.Fatalf("params: %v", err)
	}
	if _, ok := params["properties"]; !ok {
		t.Fatalf("expected properties injected, got %v", params)
	}
	// tool_choice flattened to {type, name}
	var tc map[string]string
	if err := json.Unmarshal(r.ToolChoice, &tc); err != nil {
		t.Fatalf("tool_choice: %v", err)
	}
	if tc["type"] != "function" || tc["name"] != "f" {
		t.Fatalf("tool_choice = %v", tc)
	}
}

func TestTranslateRequest_ToolChoiceString(t *testing.T) {
	out, err := TranslateRequest([]byte(`{"model":"m","messages":[{"role":"user","content":"x"}],"tool_choice":"auto"}`), "m")
	if err != nil {
		t.Fatal(err)
	}
	r := decodeCodex(t, out)
	if string(r.ToolChoice) != `"auto"` {
		t.Fatalf("tool_choice = %s", r.ToolChoice)
	}
}

func TestTranslateRequest_ReasoningEffort(t *testing.T) {
	out, err := TranslateRequest([]byte(`{"model":"m","messages":[{"role":"user","content":"x"}],"reasoning_effort":"high"}`), "m")
	if err != nil {
		t.Fatal(err)
	}
	r := decodeCodex(t, out)
	if r.Reasoning == nil || r.Reasoning.Effort != "high" || r.Reasoning.Summary != "auto" {
		t.Fatalf("reasoning = %+v", r.Reasoning)
	}
}

func TestTranslateRequest_ResponseFormatJSONSchema(t *testing.T) {
	in := `{"model":"m","messages":[{"role":"user","content":"x"}],
		"response_format":{"type":"json_schema","json_schema":{"name":"out","schema":{"type":"object"},"strict":true}}}`
	out, err := TranslateRequest([]byte(in), "m")
	if err != nil {
		t.Fatal(err)
	}
	r := decodeCodex(t, out)
	if r.Text == nil {
		t.Fatal("expected text.format")
	}
	var f map[string]any
	if err := json.Unmarshal(r.Text.Format, &f); err != nil {
		t.Fatalf("format: %v", err)
	}
	if f["type"] != "json_schema" || f["name"] != "out" || f["strict"] != true {
		t.Fatalf("format = %v", f)
	}
}

func TestTranslateRequest_MultimodalImage(t *testing.T) {
	in := `{"model":"m","messages":[{"role":"user","content":[
		{"type":"text","text":"what is this"},
		{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}
	]}]}`
	out, err := TranslateRequest([]byte(in), "m")
	if err != nil {
		t.Fatal(err)
	}
	r := decodeCodex(t, out)
	// Content should be an array of parts (input_text + input_image).
	raw, _ := json.Marshal(r.Input[0].Content)
	var parts []codexPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		t.Fatalf("parts: %v (%s)", err, raw)
	}
	if len(parts) != 2 || parts[0].Type != "input_text" || parts[1].Type != "input_image" {
		t.Fatalf("parts = %+v", parts)
	}
	if parts[1].ImageURL != "data:image/png;base64,AAAA" {
		t.Fatalf("image url = %q", parts[1].ImageURL)
	}
}
