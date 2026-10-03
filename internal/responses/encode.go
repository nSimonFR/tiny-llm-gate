package responses

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/oaichat"
)

// Flusher is the SSE sink (http.ResponseWriter + http.Flusher).
type Flusher interface {
	io.Writer
	Flush()
}

type item struct {
	kind   string // "message" | "function_call" | "reasoning"
	id     string
	index  int // output_index
	text   strings.Builder
	callID string
	name   string
	done   bool
}

func (it *item) json(status string) map[string]any {
	switch it.kind {
	case "function_call":
		return map[string]any{"type": "function_call", "id": it.id, "call_id": it.callID,
			"name": it.name, "arguments": it.text.String(), "status": status}
	case "reasoning":
		summary := []any{}
		if it.text.Len() > 0 {
			summary = append(summary, map[string]any{"type": "summary_text", "text": it.text.String()})
		}
		return map[string]any{"type": "reasoning", "id": it.id, "summary": summary}
	}
	content := []any{}
	if it.done {
		content = append(content, textPart(it.text.String()))
	}
	return map[string]any{"type": "message", "id": it.id, "status": status,
		"role": "assistant", "content": content}
}

func textPart(text string) map[string]any {
	return map[string]any{"type": "output_text", "text": text, "annotations": []any{}}
}

// Encoder implements oaichat.Handler, re-encoding chat events as a Response.
// With a nil sink it only aggregates (non-streaming clients).
type Encoder struct {
	sink    Flusher
	req     *Request
	id      string
	model   string
	created int64
	seq     int
	items   []*item
	current *item         // open message/reasoning item
	tools   map[int]*item // open function_call items by upstream index
	finish  string
	usage   *oaichat.Usage
	werr    error
}

func NewEncoder(sink Flusher, req *Request, model string) *Encoder {
	return &Encoder{sink: sink, req: req, id: "resp_" + randomHex(24), model: model,
		created: time.Now().Unix(), tools: map[int]*item{}}
}

func (e *Encoder) emit(typ string, fields map[string]any) error {
	if e.sink == nil || e.werr != nil {
		return e.werr
	}
	fields["type"] = typ
	fields["sequence_number"] = e.seq
	e.seq++
	b, _ := json.Marshal(fields)
	if _, err := fmt.Fprintf(e.sink, "event: %s\ndata: %s\n\n", typ, b); err != nil {
		e.werr = err
		return err
	}
	e.sink.Flush()
	return nil
}

// Start emits response.created / response.in_progress.
func (e *Encoder) Start() error {
	if err := e.emit("response.created", map[string]any{"response": e.response("in_progress")}); err != nil {
		return err
	}
	return e.emit("response.in_progress", map[string]any{"response": e.response("in_progress")})
}

func (e *Encoder) open(kind, prefix string) (*item, error) {
	it := &item{kind: kind, id: prefix + randomHex(24), index: len(e.items)}
	e.items = append(e.items, it)
	if err := e.emit("response.output_item.added", map[string]any{
		"output_index": it.index, "item": it.json("in_progress")}); err != nil {
		return nil, err
	}
	switch kind {
	case "message":
		err := e.emit("response.content_part.added", map[string]any{"item_id": it.id,
			"output_index": it.index, "content_index": 0, "part": textPart("")})
		return it, err
	case "reasoning":
		err := e.emit("response.reasoning_summary_part.added", map[string]any{"item_id": it.id,
			"output_index": it.index, "summary_index": 0, "part": map[string]any{"type": "summary_text", "text": ""}})
		return it, err
	}
	return it, nil
}

func (e *Encoder) close(it *item) error {
	if it == nil || it.done {
		return nil
	}
	it.done = true
	base := map[string]any{"item_id": it.id, "output_index": it.index}
	with := func(kv ...any) map[string]any {
		m := map[string]any{}
		for k, v := range base {
			m[k] = v
		}
		for i := 0; i+1 < len(kv); i += 2 {
			m[kv[i].(string)] = kv[i+1]
		}
		return m
	}
	var err error
	switch it.kind {
	case "message":
		if err = e.emit("response.output_text.done", with("content_index", 0, "text", it.text.String())); err == nil {
			err = e.emit("response.content_part.done", with("content_index", 0, "part", textPart(it.text.String())))
		}
	case "reasoning":
		if err = e.emit("response.reasoning_summary_text.done", with("summary_index", 0, "text", it.text.String())); err == nil {
			err = e.emit("response.reasoning_summary_part.done", with("summary_index", 0,
				"part", map[string]any{"type": "summary_text", "text": it.text.String()}))
		}
	case "function_call":
		err = e.emit("response.function_call_arguments.done", with("arguments", it.text.String()))
	}
	if err != nil {
		return err
	}
	return e.emit("response.output_item.done", map[string]any{"output_index": it.index, "item": it.json("completed")})
}

func (e *Encoder) switchTo(kind, prefix string) (*item, error) {
	if e.current != nil && e.current.kind == kind && !e.current.done {
		return e.current, nil
	}
	if err := e.close(e.current); err != nil {
		return nil, err
	}
	it, err := e.open(kind, prefix)
	e.current = it
	return it, err
}

func (e *Encoder) Text(delta string) error {
	it, err := e.switchTo("message", "msg_")
	if err != nil {
		return err
	}
	it.text.WriteString(delta)
	return e.emit("response.output_text.delta", map[string]any{"item_id": it.id,
		"output_index": it.index, "content_index": 0, "delta": delta})
}

func (e *Encoder) Reasoning(delta string) error {
	it, err := e.switchTo("reasoning", "rs_")
	if err != nil {
		return err
	}
	it.text.WriteString(delta)
	return e.emit("response.reasoning_summary_text.delta", map[string]any{"item_id": it.id,
		"output_index": it.index, "summary_index": 0, "delta": delta})
}

func (e *Encoder) ToolCall(index int, id, name, args string) error {
	it, ok := e.tools[index]
	if !ok {
		if err := e.close(e.current); err != nil {
			return err
		}
		e.current = nil
		it = &item{kind: "function_call", id: "fc_" + randomHex(24), index: len(e.items),
			callID: id, name: name}
		if it.callID == "" {
			it.callID = "call_" + randomHex(12)
		}
		e.items = append(e.items, it)
		e.tools[index] = it
		if err := e.emit("response.output_item.added", map[string]any{
			"output_index": it.index, "item": it.json("in_progress")}); err != nil {
			return err
		}
	} else if name != "" && it.name == "" {
		it.name = name
	}
	if args == "" {
		return nil
	}
	it.text.WriteString(args)
	return e.emit("response.function_call_arguments.delta", map[string]any{"item_id": it.id,
		"output_index": it.index, "delta": args})
}

func (e *Encoder) Finish(reason string) error { e.finish = reason; return nil }

func (e *Encoder) Usage(u *oaichat.Usage) error { e.usage = u; return nil }

// Complete closes every open item and emits response.completed (or
// .incomplete). It returns the final Response for non-streaming clients.
func (e *Encoder) Complete() ([]byte, error) {
	if err := e.close(e.current); err != nil {
		return nil, err
	}
	idx := make([]int, 0, len(e.tools))
	for i := range e.tools {
		idx = append(idx, i)
	}
	sort.Ints(idx)
	for _, i := range idx {
		if err := e.close(e.tools[i]); err != nil {
			return nil, err
		}
	}
	status := "completed"
	if e.finish == "length" || e.finish == "content_filter" {
		status = "incomplete"
	}
	resp := e.response(status)
	if err := e.emit("response."+status, map[string]any{"response": resp}); err != nil {
		return nil, err
	}
	return json.Marshal(resp)
}

// Fail emits response.failed after a mid-stream upstream error.
func (e *Encoder) Fail(msg string) {
	resp := e.response("failed")
	resp["error"] = map[string]any{"code": "server_error", "message": msg}
	_ = e.emit("response.failed", map[string]any{"response": resp})
}

func (e *Encoder) response(status string) map[string]any {
	output := make([]any, 0, len(e.items))
	if status != "in_progress" {
		for _, it := range e.items {
			output = append(output, it.json("completed"))
		}
	}
	tools := []any{}
	for _, t := range e.req.Tools {
		tools = append(tools, t)
	}
	var toolChoice any = "auto"
	if len(e.req.ToolChoice) > 0 {
		toolChoice = e.req.ToolChoice
	}
	metadata := any(map[string]any{})
	if len(e.req.Metadata) > 0 {
		metadata = e.req.Metadata
	}
	r := map[string]any{
		"id": e.id, "object": "response", "created_at": e.created, "status": status,
		"model": e.model, "output": output, "error": nil, "incomplete_details": nil,
		"instructions": nilIfEmpty(e.req.Instructions), "max_output_tokens": e.req.MaxOutputTokens,
		"parallel_tool_calls":  e.req.ParallelToolCalls == nil || *e.req.ParallelToolCalls,
		"previous_response_id": nil, "store": false, "temperature": e.req.Temperature,
		"top_p": e.req.TopP, "tool_choice": toolChoice, "tools": tools,
		"truncation": "disabled", "metadata": metadata, "text": map[string]any{"format": map[string]any{"type": "text"}},
	}
	if e.req.Text != nil && len(e.req.Text.Format) > 0 {
		r["text"] = map[string]any{"format": e.req.Text.Format}
	}
	if e.req.Reasoning != nil {
		r["reasoning"] = map[string]any{"effort": e.req.Reasoning.Effort, "summary": nil}
	}
	switch e.finish {
	case "length":
		r["incomplete_details"] = map[string]any{"reason": "max_output_tokens"}
	case "content_filter":
		r["incomplete_details"] = map[string]any{"reason": "content_filter"}
	}
	if e.usage != nil {
		r["usage"] = map[string]any{
			"input_tokens":          e.usage.PromptTokens,
			"input_tokens_details":  map[string]any{"cached_tokens": e.usage.Cached()},
			"output_tokens":         e.usage.CompletionTokens,
			"output_tokens_details": map[string]any{"reasoning_tokens": e.usage.Reasoning()},
			"total_tokens":          e.usage.PromptTokens + e.usage.CompletionTokens,
		}
	}
	return r
}

func nilIfEmpty(s string) any {
	if s == "" {
		return nil
	}
	return s
}

func randomHex(n int) string {
	b := make([]byte, n/2)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
