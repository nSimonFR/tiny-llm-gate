package messages

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/nSimonFR/tiny-llm-gate/internal/oaichat"
)

// Flusher is the SSE sink (http.ResponseWriter + http.Flusher).
type Flusher interface {
	io.Writer
	Flush()
}

type contentBlock struct {
	kind  string // "text" | "thinking" | "tool_use"
	index int
	text  strings.Builder
	id    string
	name  string
	open  bool
}

func (b *contentBlock) json() map[string]any {
	switch b.kind {
	case "thinking":
		return map[string]any{"type": "thinking", "thinking": b.text.String(), "signature": ""}
	case "tool_use":
		input := json.RawMessage(b.text.String())
		if len(input) == 0 || !json.Valid(input) {
			input = json.RawMessage(`{}`)
		}
		return map[string]any{"type": "tool_use", "id": b.id, "name": b.name, "input": input}
	}
	return map[string]any{"type": "text", "text": b.text.String()}
}

// Encoder implements oaichat.Handler, re-encoding chat events as an
// Anthropic Message. With a nil sink it only aggregates.
type Encoder struct {
	sink    Flusher
	id      string
	model   string
	blocks  []*contentBlock
	current *contentBlock
	tools   map[int]*contentBlock
	finish  string
	usage   *oaichat.Usage
	werr    error
}

func NewEncoder(sink Flusher, model string) *Encoder {
	return &Encoder{sink: sink, id: "msg_" + randomHex(24), model: model, tools: map[int]*contentBlock{}}
}

func (e *Encoder) emit(typ string, fields map[string]any) error {
	if e.sink == nil || e.werr != nil {
		return e.werr
	}
	fields["type"] = typ
	b, _ := json.Marshal(fields)
	if _, err := fmt.Fprintf(e.sink, "event: %s\ndata: %s\n\n", typ, b); err != nil {
		e.werr = err
		return err
	}
	e.sink.Flush()
	return nil
}

func (e *Encoder) Start() error {
	return e.emit("message_start", map[string]any{"message": map[string]any{
		"id": e.id, "type": "message", "role": "assistant", "model": e.model,
		"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
		"usage": map[string]any{"input_tokens": 0, "output_tokens": 0},
	}})
}

func (e *Encoder) closeCurrent() error {
	b := e.current
	e.current = nil
	if b == nil || !b.open {
		return nil
	}
	b.open = false
	return e.emit("content_block_stop", map[string]any{"index": b.index})
}

func (e *Encoder) start(b *contentBlock) error {
	if err := e.closeCurrent(); err != nil {
		return err
	}
	b.index = len(e.blocks)
	b.open = true
	e.blocks = append(e.blocks, b)
	e.current = b
	start := map[string]any{"type": b.kind}
	switch b.kind {
	case "text":
		start["text"] = ""
	case "thinking":
		start["thinking"] = ""
		start["signature"] = ""
	case "tool_use":
		start["id"], start["name"], start["input"] = b.id, b.name, map[string]any{}
	}
	return e.emit("content_block_start", map[string]any{"index": b.index, "content_block": start})
}

func (e *Encoder) Text(delta string) error {
	if e.current == nil || e.current.kind != "text" {
		if err := e.start(&contentBlock{kind: "text"}); err != nil {
			return err
		}
	}
	e.current.text.WriteString(delta)
	return e.emit("content_block_delta", map[string]any{"index": e.current.index,
		"delta": map[string]any{"type": "text_delta", "text": delta}})
}

func (e *Encoder) Reasoning(delta string) error {
	if e.current == nil || e.current.kind != "thinking" {
		if err := e.start(&contentBlock{kind: "thinking"}); err != nil {
			return err
		}
	}
	e.current.text.WriteString(delta)
	return e.emit("content_block_delta", map[string]any{"index": e.current.index,
		"delta": map[string]any{"type": "thinking_delta", "thinking": delta}})
}

func (e *Encoder) ToolCall(index int, id, name, args string) error {
	b, ok := e.tools[index]
	if !ok {
		if id == "" {
			id = "toolu_" + randomHex(24)
		}
		b = &contentBlock{kind: "tool_use", id: id, name: name}
		e.tools[index] = b
		if err := e.start(b); err != nil {
			return err
		}
	}
	if args == "" {
		return nil
	}
	b.text.WriteString(args)
	if !b.open {
		return nil // blocks are sequential on the wire; a late fragment only aggregates
	}
	return e.emit("content_block_delta", map[string]any{"index": b.index,
		"delta": map[string]any{"type": "input_json_delta", "partial_json": args}})
}

func (e *Encoder) Finish(reason string) error { e.finish = reason; return nil }

func (e *Encoder) Usage(u *oaichat.Usage) error { e.usage = u; return nil }

func (e *Encoder) stopReason() string {
	switch e.finish {
	case "length":
		return "max_tokens"
	case "tool_calls", "function_call":
		return "tool_use"
	}
	if len(e.tools) > 0 {
		return "tool_use"
	}
	return "end_turn"
}

func (e *Encoder) usageJSON() map[string]any {
	u := map[string]any{"input_tokens": 0, "output_tokens": 0}
	if e.usage != nil {
		u["input_tokens"] = e.usage.PromptTokens - e.usage.Cached()
		u["output_tokens"] = e.usage.CompletionTokens
		if c := e.usage.Cached(); c > 0 {
			u["cache_read_input_tokens"] = c
		}
	}
	return u
}

// Complete closes the open block, emits message_delta/message_stop, and
// returns the aggregated Message for non-streaming clients.
func (e *Encoder) Complete() ([]byte, error) {
	if err := e.closeCurrent(); err != nil {
		return nil, err
	}
	stop := e.stopReason()
	if err := e.emit("message_delta", map[string]any{
		"delta": map[string]any{"stop_reason": stop, "stop_sequence": nil},
		"usage": e.usageJSON()}); err != nil {
		return nil, err
	}
	if err := e.emit("message_stop", map[string]any{}); err != nil {
		return nil, err
	}
	content := make([]any, 0, len(e.blocks))
	for _, b := range e.blocks {
		content = append(content, b.json())
	}
	return json.Marshal(map[string]any{
		"id": e.id, "type": "message", "role": "assistant", "model": e.model,
		"content": content, "stop_reason": stop, "stop_sequence": nil, "usage": e.usageJSON(),
	})
}

// Fail emits an Anthropic error event after a mid-stream upstream error.
func (e *Encoder) Fail(msg string) {
	_ = e.emit("error", map[string]any{"error": map[string]any{"type": "api_error", "message": msg}})
}

// ErrorBody is the Anthropic error envelope.
func ErrorBody(typ, msg string) []byte {
	b, _ := json.Marshal(map[string]any{"type": "error", "error": map[string]any{"type": typ, "message": msg}})
	return b
}

// ErrorType maps an HTTP status to the Anthropic error type.
func ErrorType(status int) string {
	switch status {
	case 400:
		return "invalid_request_error"
	case 401:
		return "authentication_error"
	case 403:
		return "permission_error"
	case 404:
		return "not_found_error"
	case 413:
		return "request_too_large"
	case 429:
		return "rate_limit_error"
	case 529:
		return "overloaded_error"
	}
	return "api_error"
}

func randomHex(n int) string {
	b := make([]byte, n/2)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
