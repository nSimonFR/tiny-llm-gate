package codex

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// buf implements the flusher interface for Stream tests.
type buf struct{ bytes.Buffer }

func (b *buf) Flush() {}

// sse builds an SSE stream from a list of event JSON payloads.
func sse(events ...string) string {
	var sb strings.Builder
	for _, e := range events {
		sb.WriteString("event: x\ndata: ")
		sb.WriteString(e)
		sb.WriteString("\n\n")
	}
	sb.WriteString("data: [DONE]\n\n")
	return sb.String()
}

// parseChunks extracts the JSON objects from an OpenAI SSE chunk stream.
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

func TestStream_TextAndUsage(t *testing.T) {
	stream := sse(
		`{"type":"response.created","response":{"id":"resp_1"}}`,
		`{"type":"response.output_text.delta","delta":"Hel"}`,
		`{"type":"response.output_text.delta","delta":"lo"}`,
		`{"type":"response.completed","response":{"id":"resp_1","usage":{"input_tokens":10,"output_tokens":2,"cached_tokens":3,"reasoning_tokens":1}}}`,
	)
	tr := NewTranslator("gpt-5.5")
	out := &buf{}
	usage, err := tr.Stream(out, strings.NewReader(stream))
	if err != nil {
		t.Fatal(err)
	}
	if usage == nil || usage.PromptTokens != 10 || usage.CompletionTokens != 2 || usage.TotalTokens != 12 {
		t.Fatalf("usage = %+v", usage)
	}
	chunks := parseChunks(t, out.String())

	// First chunk is the role delta.
	if chunks[0].Choices[0].Delta.Role != "assistant" {
		t.Fatalf("first chunk not role: %+v", chunks[0])
	}
	// Concatenated content across content chunks.
	var content string
	var finish string
	for _, c := range chunks {
		content += c.Choices[0].Delta.Content
		if c.Choices[0].FinishReason != nil {
			finish = *c.Choices[0].FinishReason
		}
	}
	if content != "Hello" {
		t.Fatalf("content = %q", content)
	}
	if finish != "stop" {
		t.Fatalf("finish = %q", finish)
	}
}

func TestStream_ToolCalls(t *testing.T) {
	stream := sse(
		`{"type":"response.output_item.added","outputIndex":0,"item":{"type":"function_call","id":"item_1","call_id":"call_1","name":"get_weather"}}`,
		`{"type":"response.function_call_arguments.delta","call_id":"call_1","delta":"{\"city\":"}`,
		`{"type":"response.function_call_arguments.delta","call_id":"call_1","delta":"\"Paris\"}"}`,
		`{"type":"response.function_call_arguments.done","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Paris\"}"}`,
		`{"type":"response.completed","response":{"id":"r","usage":{"input_tokens":5,"output_tokens":7}}}`,
	)
	tr := NewTranslator("gpt-5.5")
	out := &buf{}
	if _, err := tr.Stream(out, strings.NewReader(stream)); err != nil {
		t.Fatal(err)
	}
	chunks := parseChunks(t, out.String())

	var name, args, finish string
	for _, c := range chunks {
		for _, tc := range c.Choices[0].Delta.ToolCalls {
			if tc.Function.Name != "" {
				name = tc.Function.Name
			}
			args += tc.Function.Arguments
		}
		if c.Choices[0].FinishReason != nil {
			finish = *c.Choices[0].FinishReason
		}
	}
	if name != "get_weather" {
		t.Fatalf("tool name = %q", name)
	}
	if args != `{"city":"Paris"}` {
		t.Fatalf("tool args = %q", args)
	}
	if finish != "tool_calls" {
		t.Fatalf("finish = %q", finish)
	}
}

func TestStream_UpstreamError(t *testing.T) {
	stream := sse(
		`{"type":"response.output_text.delta","delta":"partial"}`,
		`{"type":"response.failed","error":{"type":"server_error","code":"boom","message":"kaboom"}}`,
	)
	tr := NewTranslator("m")
	out := &buf{}
	_, err := tr.Stream(out, strings.NewReader(stream))
	if err == nil || !strings.Contains(err.Error(), "kaboom") {
		t.Fatalf("expected error carrying upstream message, got %v", err)
	}
}

func TestCollect_TextAndUsage(t *testing.T) {
	stream := sse(
		`{"type":"response.output_text.delta","delta":"Hi "}`,
		`{"type":"response.output_text.delta","delta":"there"}`,
		`{"type":"response.completed","response":{"id":"r","usage":{"input_tokens":4,"output_tokens":2}}}`,
	)
	tr := NewTranslator("gpt-5.5")
	out, err := tr.Collect(strings.NewReader(stream))
	if err != nil {
		t.Fatal(err)
	}
	var resp openAIResponse
	if err := json.Unmarshal(out, &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Object != "chat.completion" {
		t.Fatalf("object = %q", resp.Object)
	}
	if resp.Choices[0].Message.Content == nil || *resp.Choices[0].Message.Content != "Hi there" {
		t.Fatalf("content = %v", resp.Choices[0].Message.Content)
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish = %q", resp.Choices[0].FinishReason)
	}
	if resp.Usage == nil || resp.Usage.TotalTokens != 6 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestCollect_ToolCalls(t *testing.T) {
	stream := sse(
		`{"type":"response.output_item.added","item":{"type":"function_call","id":"item_1","call_id":"call_9","name":"f"}}`,
		`{"type":"response.function_call_arguments.delta","call_id":"call_9","delta":"{\"a\":1}"}`,
		`{"type":"response.function_call_arguments.done","call_id":"call_9","name":"f","arguments":"{\"a\":1}"}`,
		`{"type":"response.completed","response":{"id":"r","usage":{"input_tokens":1,"output_tokens":1}}}`,
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
	if tc.ID != "call_9" || tc.Function.Name != "f" || tc.Function.Arguments != `{"a":1}` {
		t.Fatalf("tool call = %+v", tc)
	}
	if resp.Choices[0].FinishReason != "tool_calls" {
		t.Fatalf("finish = %q", resp.Choices[0].FinishReason)
	}
}

// Deltas that reference an output-item id rather than the call_id must still
// resolve to the right tool call (via the item.id → call_id map).
func TestStream_ToolCallDeltaByItemID(t *testing.T) {
	stream := sse(
		`{"type":"response.output_item.added","item":{"type":"function_call","id":"item_42","call_id":"call_7","name":"g"}}`,
		`{"type":"response.function_call_arguments.delta","call_id":"item_42","delta":"{\"x\":1}"}`,
		`{"type":"response.completed","response":{"id":"r","usage":{"input_tokens":1,"output_tokens":1}}}`,
	)
	tr := NewTranslator("m")
	out := &buf{}
	if _, err := tr.Stream(out, strings.NewReader(stream)); err != nil {
		t.Fatal(err)
	}
	var id, args string
	for _, c := range parseChunks(t, out.String()) {
		for _, tc := range c.Choices[0].Delta.ToolCalls {
			if tc.ID != "" {
				id = tc.ID
			}
			args += tc.Function.Arguments
		}
	}
	if id != "call_7" {
		t.Fatalf("expected call_7 id, got %q", id)
	}
	if args != `{"x":1}` {
		t.Fatalf("args = %q", args)
	}
}
