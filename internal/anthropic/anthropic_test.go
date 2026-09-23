package anthropic

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// buf implements the flusher interface for Stream tests.
type buf struct{ bytes.Buffer }

func (b *buf) Flush() {}

func sse(events ...string) string {
	var sb strings.Builder
	for _, e := range events {
		sb.WriteString("event: x\ndata: ")
		sb.WriteString(e)
		sb.WriteString("\n\n")
	}
	return sb.String()
}

func parseChunks(t *testing.T, s string) []openAIChunk {
	t.Helper()
	var chunks []openAIChunk
	for _, block := range strings.Split(s, "\n\n") {
		block = strings.TrimSpace(block)
		if !strings.HasPrefix(block, "data:") {
			continue
		}
		payload := strings.TrimSpace(block[len("data:"):])
		if payload == "[DONE]" || payload == "" {
			continue
		}
		var c openAIChunk
		if err := json.Unmarshal([]byte(payload), &c); err != nil {
			t.Fatalf("bad chunk %q: %v", payload, err)
		}
		chunks = append(chunks, c)
	}
	return chunks
}

func TestTranslateRequest_SystemSentinelAndMessages(t *testing.T) {
	body := `{
		"model":"ignored",
		"messages":[
			{"role":"system","content":"be terse"},
			{"role":"user","content":"hi"}
		]
	}`
	out, err := TranslateRequest([]byte(body), "claude-x", 4096)
	if err != nil {
		t.Fatal(err)
	}
	var req anthropicRequest
	if err := json.Unmarshal(out, &req); err != nil {
		t.Fatal(err)
	}
	if req.Model != "claude-x" {
		t.Errorf("model = %q", req.Model)
	}
	if req.MaxTokens != 4096 {
		t.Errorf("max_tokens = %d", req.MaxTokens)
	}
	if !req.Stream {
		t.Error("stream should be true upstream")
	}
	if len(req.System) != 2 || req.System[0].Text != claudeCodeSystemSentinel || req.System[1].Text != "be terse" {
		t.Fatalf("system = %+v (want sentinel then 'be terse')", req.System)
	}
	if len(req.Messages) != 1 || req.Messages[0].Role != "user" {
		t.Fatalf("messages = %+v", req.Messages)
	}
}

func TestTranslateRequest_ToolsAndToolResultRoundTrip(t *testing.T) {
	body := `{
		"model":"m",
		"max_tokens":128,
		"messages":[
			{"role":"user","content":"weather?"},
			{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},
			{"role":"tool","tool_call_id":"call_1","content":"sunny"}
		],
		"tools":[{"type":"function","function":{"name":"get_weather","description":"w","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}],
		"tool_choice":"required"
	}`
	out, err := TranslateRequest([]byte(body), "m", 4096)
	if err != nil {
		t.Fatal(err)
	}
	var req anthropicRequest
	if err := json.Unmarshal(out, &req); err != nil {
		t.Fatal(err)
	}
	if req.MaxTokens != 128 {
		t.Errorf("max_tokens = %d, want 128 (client value)", req.MaxTokens)
	}
	if len(req.Tools) != 1 || req.Tools[0].Name != "get_weather" {
		t.Fatalf("tools = %+v", req.Tools)
	}
	if string(req.ToolChoice) != `{"type":"any"}` {
		t.Errorf("tool_choice = %s, want {\"type\":\"any\"}", req.ToolChoice)
	}
	// user, assistant(tool_use), user(tool_result)
	if len(req.Messages) != 3 {
		t.Fatalf("messages = %d: %+v", len(req.Messages), req.Messages)
	}
	// assistant turn has a tool_use block referencing call_1.
	asg, _ := json.Marshal(req.Messages[1])
	if !strings.Contains(string(asg), `"type":"tool_use"`) || !strings.Contains(string(asg), `"id":"call_1"`) || !strings.Contains(string(asg), `get_weather`) {
		t.Fatalf("assistant turn missing tool_use: %s", asg)
	}
	// tool result turn is a user turn with tool_result referencing call_1.
	tr, _ := json.Marshal(req.Messages[2])
	if req.Messages[2].Role != "user" || !strings.Contains(string(tr), `"type":"tool_result"`) || !strings.Contains(string(tr), `"tool_use_id":"call_1"`) {
		t.Fatalf("tool result turn wrong: %s", tr)
	}
}

func TestStream_TextAndUsage(t *testing.T) {
	stream := sse(
		`{"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":10}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Bon"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"jour"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`,
		`{"type":"message_stop"}`,
	)
	tr := NewTranslator("claude-x")
	out := &buf{}
	usage, err := tr.Stream(out, strings.NewReader(stream))
	if err != nil {
		t.Fatal(err)
	}
	if usage == nil || usage.PromptTokens != 10 || usage.CompletionTokens != 2 || usage.TotalTokens != 12 {
		t.Fatalf("usage = %+v", usage)
	}
	var content, finish string
	for _, c := range parseChunks(t, out.String()) {
		content += c.Choices[0].Delta.Content
		if c.Choices[0].FinishReason != nil {
			finish = *c.Choices[0].FinishReason
		}
	}
	if content != "Bonjour" {
		t.Errorf("content = %q", content)
	}
	if finish != "stop" {
		t.Errorf("finish = %q", finish)
	}
}

func TestStream_ToolCall(t *testing.T) {
	stream := sse(
		`{"type":"message_start","message":{"id":"m","usage":{"input_tokens":5}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}`,
		`{"type":"message_stop"}`,
	)
	tr := NewTranslator("claude-x")
	out := &buf{}
	if _, err := tr.Stream(out, strings.NewReader(stream)); err != nil {
		t.Fatal(err)
	}
	var id, name, args, finish string
	for _, c := range parseChunks(t, out.String()) {
		for _, tc := range c.Choices[0].Delta.ToolCalls {
			if tc.ID != "" {
				id = tc.ID
			}
			if tc.Function.Name != "" {
				name = tc.Function.Name
			}
			args += tc.Function.Arguments
		}
		if c.Choices[0].FinishReason != nil {
			finish = *c.Choices[0].FinishReason
		}
	}
	if id != "toolu_1" || name != "get_weather" {
		t.Fatalf("tool id/name = %q/%q", id, name)
	}
	if args != `{"city":"Paris"}` {
		t.Fatalf("tool args = %q", args)
	}
	if finish != "tool_calls" {
		t.Fatalf("finish = %q", finish)
	}
}

func TestCollect_ToolCall(t *testing.T) {
	stream := sse(
		`{"type":"message_start","message":{"id":"m","usage":{"input_tokens":5}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_9","name":"f"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}`,
		`{"type":"message_stop"}`,
	)
	tr := NewTranslator("m")
	out, err := tr.Collect(strings.NewReader(stream))
	if err != nil {
		t.Fatal(err)
	}
	var resp openAIResponse
	if err := json.Unmarshal(out, &resp); err != nil {
		t.Fatal(err)
	}
	msg := resp.Choices[0].Message
	if len(msg.ToolCalls) != 1 {
		t.Fatalf("tool calls = %+v", msg.ToolCalls)
	}
	tc := msg.ToolCalls[0]
	if tc.ID != "toolu_9" || tc.Function.Name != "f" || tc.Function.Arguments != `{"a":1}` {
		t.Fatalf("tool call = %+v", tc)
	}
	if resp.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("finish = %q", resp.Choices[0].FinishReason)
	}
	if resp.Usage == nil || resp.Usage.TotalTokens != 8 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestStream_UpstreamError(t *testing.T) {
	stream := sse(
		`{"type":"message_start","message":{"id":"m"}}`,
		`{"type":"error","error":{"type":"overloaded_error","message":"boom"}}`,
	)
	tr := NewTranslator("m")
	out := &buf{}
	_, err := tr.Stream(out, strings.NewReader(stream))
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("expected error carrying upstream message, got %v", err)
	}
}
