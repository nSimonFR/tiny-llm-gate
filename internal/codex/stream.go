package codex

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"
)

// Translator converts a Codex Responses SSE stream into OpenAI chat-completion
// output. It is single-use: create one per upstream response.
//
// For streaming clients, call Stream to pipe OpenAI chat.completion.chunk SSE
// to the writer as events arrive. For non-streaming clients, call Collect to
// aggregate the whole stream into a single chat.completion JSON object.
type Translator struct {
	model   string
	id      string
	created int64
}

// NewTranslator builds a Translator for the given client-facing model name.
func NewTranslator(model string) *Translator {
	return &Translator{
		model:   model,
		id:      "chatcmpl-" + randomID(),
		created: time.Now().Unix(),
	}
}

// flusher is the subset of http.ResponseWriter we need to push SSE chunks.
type flusher interface {
	io.Writer
	Flush()
}

// aggregate accumulates the parsed stream state shared by Stream and Collect.
type aggregate struct {
	content   strings.Builder
	toolCalls []*toolAccum
	usage     *codexUsage
	// itemIDToCall maps a Codex output-item id to its call_id/name, so
	// argument delta/done events referencing an item id resolve correctly.
	itemIDToCall map[string]*toolAccum
	byCallID     map[string]*toolAccum
	errEvent     *struct {
		Type    string
		Code    string
		Message string
	}
}

type toolAccum struct {
	index int
	id    string
	name  string
	args  strings.Builder
}

func newAggregate() *aggregate {
	return &aggregate{
		itemIDToCall: map[string]*toolAccum{},
		byCallID:     map[string]*toolAccum{},
	}
}

// Stream translates the Codex SSE stream in `src` into OpenAI chunk SSE on
// `dst`, flushing after every write. It returns the collected usage (may be
// nil) and any terminal error event surfaced by the backend.
func (t *Translator) Stream(dst flusher, src io.Reader) (*openAIUsage, error) {
	agg := newAggregate()

	// Opening role chunk.
	t.writeChunk(dst, openAIChunk{
		ID: t.id, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
		Choices: []chunkChoice{{Index: 0, Delta: chunkDelta{Role: "assistant"}}},
	})

	sc := newSSEScanner(src)
	for sc.next() {
		evt, ok := sc.event()
		if !ok {
			continue
		}
		switch evt.Type {
		case "response.output_text.delta":
			if evt.Delta != "" {
				t.writeChunk(dst, t.contentChunk(evt.Delta))
			}
		case "response.output_item.added":
			if evt.Item != nil && evt.Item.Type == "function_call" && evt.Item.CallID != "" {
				ta := agg.startTool(evt.Item.CallID, evt.Item.Name)
				agg.itemIDToCall[evt.Item.ID] = ta
				t.writeChunk(dst, t.toolStartChunk(ta))
			}
		case "response.function_call_arguments.delta":
			ta := agg.resolveTool(evt.CallID, "")
			if ta != nil && evt.Delta != "" {
				ta.args.WriteString(evt.Delta)
				t.writeChunk(dst, t.toolArgsChunk(ta, evt.Delta))
			}
		case "response.function_call_arguments.done":
			// Terminal for a tool call — arguments already streamed via deltas.
			_ = agg.resolveTool(evt.CallID, evt.Name)
		case "response.completed", "response.incomplete":
			if evt.Response != nil && evt.Response.Usage != nil {
				agg.usage = evt.Response.Usage
			}
		case "error", "response.failed":
			if evt.Error != nil {
				return nil, fmt.Errorf("codex upstream error (%s): %s", evt.Error.Code, evt.Error.Message)
			}
		}
	}
	if err := sc.err(); err != nil {
		return nil, err
	}

	finish := "stop"
	if len(agg.toolCalls) > 0 {
		finish = "tool_calls"
	}
	// Final chunk with finish_reason and (when present) usage.
	final := openAIChunk{
		ID: t.id, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
		Choices: []chunkChoice{{Index: 0, Delta: chunkDelta{}, FinishReason: &finish}},
	}
	usage := toOpenAIUsage(agg.usage)
	final.Usage = usage
	t.writeChunk(dst, final)
	_, _ = io.WriteString(dst, "data: [DONE]\n\n")
	dst.Flush()
	return usage, nil
}

// Collect consumes the whole Codex SSE stream and returns a single OpenAI
// chat.completion JSON object (for non-streaming clients).
func (t *Translator) Collect(src io.Reader) ([]byte, error) {
	agg := newAggregate()
	sc := newSSEScanner(src)
	for sc.next() {
		evt, ok := sc.event()
		if !ok {
			continue
		}
		switch evt.Type {
		case "response.output_text.delta":
			agg.content.WriteString(evt.Delta)
		case "response.output_item.added":
			if evt.Item != nil && evt.Item.Type == "function_call" && evt.Item.CallID != "" {
				ta := agg.startTool(evt.Item.CallID, evt.Item.Name)
				agg.itemIDToCall[evt.Item.ID] = ta
			}
		case "response.function_call_arguments.delta":
			if ta := agg.resolveTool(evt.CallID, ""); ta != nil {
				ta.args.WriteString(evt.Delta)
			}
		case "response.function_call_arguments.done":
			if ta := agg.resolveTool(evt.CallID, evt.Name); ta != nil && ta.args.Len() == 0 && evt.Args != "" {
				ta.args.WriteString(evt.Args)
			}
		case "response.completed", "response.incomplete":
			if evt.Response != nil && evt.Response.Usage != nil {
				agg.usage = evt.Response.Usage
			}
		case "error", "response.failed":
			if evt.Error != nil {
				return nil, fmt.Errorf("codex upstream error (%s): %s", evt.Error.Code, evt.Error.Message)
			}
		}
	}
	if err := sc.err(); err != nil {
		return nil, err
	}

	finish := "stop"
	msg := responseMessage{Role: "assistant"}
	if len(agg.toolCalls) > 0 {
		finish = "tool_calls"
		for _, ta := range agg.toolCalls {
			var tc responseToolCall
			tc.ID = ta.id
			tc.Type = "function"
			tc.Function.Name = ta.name
			tc.Function.Arguments = ta.args.String()
			msg.ToolCalls = append(msg.ToolCalls, tc)
		}
	}
	if c := agg.content.String(); c != "" || len(msg.ToolCalls) == 0 {
		msg.Content = &c
	}

	resp := openAIResponse{
		ID: strings.Replace(t.id, "chatcmpl-", "chatcmpl-", 1), Object: "chat.completion",
		Created: t.created, Model: t.model,
		Choices: []responseChoice{{Index: 0, Message: msg, FinishReason: finish}},
		Usage:   toOpenAIUsage(agg.usage),
	}
	return json.Marshal(resp)
}

// ── aggregate helpers ────────────────────────────────────────────

func (a *aggregate) startTool(callID, name string) *toolAccum {
	if ta, ok := a.byCallID[callID]; ok {
		if name != "" {
			ta.name = name
		}
		return ta
	}
	ta := &toolAccum{index: len(a.toolCalls), id: callID, name: name}
	a.toolCalls = append(a.toolCalls, ta)
	a.byCallID[callID] = ta
	return ta
}

// resolveTool finds the tool accumulator for a delta/done event. The event's
// call_id field may hold either a real call_id or an output-item id; try both.
// If neither is known yet (delta before output_item.added), start one.
func (a *aggregate) resolveTool(callID, name string) *toolAccum {
	if callID == "" {
		return nil
	}
	if ta, ok := a.byCallID[callID]; ok {
		if name != "" {
			ta.name = name
		}
		return ta
	}
	if ta, ok := a.itemIDToCall[callID]; ok {
		if name != "" {
			ta.name = name
		}
		return ta
	}
	return a.startTool(callID, name)
}

// ── chunk builders ───────────────────────────────────────────────

func (t *Translator) contentChunk(delta string) openAIChunk {
	return openAIChunk{
		ID: t.id, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
		Choices: []chunkChoice{{Index: 0, Delta: chunkDelta{Content: delta}}},
	}
}

func (t *Translator) toolStartChunk(ta *toolAccum) openAIChunk {
	var c chunkToolCall
	c.Index = ta.index
	c.ID = ta.id
	c.Type = "function"
	c.Function.Name = ta.name
	c.Function.Arguments = ""
	return openAIChunk{
		ID: t.id, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
		Choices: []chunkChoice{{Index: 0, Delta: chunkDelta{ToolCalls: []chunkToolCall{c}}}},
	}
}

func (t *Translator) toolArgsChunk(ta *toolAccum, delta string) openAIChunk {
	var c chunkToolCall
	c.Index = ta.index
	c.Function.Arguments = delta
	return openAIChunk{
		ID: t.id, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
		Choices: []chunkChoice{{Index: 0, Delta: chunkDelta{ToolCalls: []chunkToolCall{c}}}},
	}
}

func (t *Translator) writeChunk(dst flusher, c openAIChunk) {
	b, err := json.Marshal(c)
	if err != nil {
		return
	}
	_, _ = io.WriteString(dst, "data: ")
	_, _ = dst.Write(b)
	_, _ = io.WriteString(dst, "\n\n")
	dst.Flush()
}

func toOpenAIUsage(u *codexUsage) *openAIUsage {
	if u == nil {
		return nil
	}
	return &openAIUsage{
		PromptTokens:     u.InputTokens,
		CompletionTokens: u.OutputTokens,
		TotalTokens:      u.InputTokens + u.OutputTokens,
	}
}

func randomID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// ── SSE scanner ──────────────────────────────────────────────────

// sseScanner reads an SSE stream block-by-block (blocks separated by a blank
// line) and decodes each block's `data:` payload as a codexEvent.
type sseScanner struct {
	r     *bufio.Reader
	block strings.Builder
	cur   codexEvent
	curOK bool
	rerr  error
}

func newSSEScanner(src io.Reader) *sseScanner {
	return &sseScanner{r: bufio.NewReaderSize(src, 64*1024)}
}

// next advances to the next event block. Returns false at EOF or on error.
func (s *sseScanner) next() bool {
	s.block.Reset()
	for {
		line, err := s.r.ReadString('\n')
		if len(line) > 0 {
			trimmed := strings.TrimRight(line, "\r\n")
			if trimmed == "" {
				// End of a block.
				if s.block.Len() > 0 {
					s.parseBlock(s.block.String())
					return true
				}
				continue // skip leading/blank separators
			}
			if s.block.Len() > 0 {
				s.block.WriteByte('\n')
			}
			s.block.WriteString(trimmed)
		}
		if err != nil {
			if err != io.EOF {
				s.rerr = err
			}
			if s.block.Len() > 0 {
				s.parseBlock(s.block.String())
				return true
			}
			return false
		}
	}
}

func (s *sseScanner) parseBlock(block string) {
	s.curOK = false
	var data strings.Builder
	for _, line := range strings.Split(block, "\n") {
		if strings.HasPrefix(line, "data:") {
			if data.Len() > 0 {
				data.WriteByte('\n')
			}
			data.WriteString(strings.TrimSpace(line[len("data:"):]))
		}
	}
	raw := data.String()
	if raw == "" || raw == "[DONE]" {
		return
	}
	var evt codexEvent
	if json.Unmarshal([]byte(raw), &evt) != nil {
		return
	}
	s.cur = evt
	s.curOK = true
}

func (s *sseScanner) event() (codexEvent, bool) { return s.cur, s.curOK }
func (s *sseScanner) err() error                { return s.rerr }
