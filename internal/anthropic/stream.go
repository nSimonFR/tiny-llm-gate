package anthropic

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

// Translator converts an Anthropic Messages SSE stream into OpenAI
// chat-completion output. Single-use: create one per upstream response.
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

// flusher is the subset of http.ResponseWriter (or an io.Pipe adapter) we need.
type flusher interface {
	io.Writer
	Flush()
}

type aggregate struct {
	content      strings.Builder
	toolCalls    []*toolAccum
	byBlockIndex map[int]*toolAccum
	inputTokens  int
	outputTokens int
	stopReason   string
	errEvent     error
}

type toolAccum struct {
	index int // OpenAI tool_call index
	id    string
	name  string
	args  strings.Builder
}

func newAggregate() *aggregate { return &aggregate{byBlockIndex: map[int]*toolAccum{}} }

func (a *aggregate) startTool(blockIndex int, id, name string) *toolAccum {
	if ta, ok := a.byBlockIndex[blockIndex]; ok {
		return ta
	}
	ta := &toolAccum{index: len(a.toolCalls), id: id, name: name}
	a.toolCalls = append(a.toolCalls, ta)
	a.byBlockIndex[blockIndex] = ta
	return ta
}

// Stream translates the Anthropic SSE in src into OpenAI chunk SSE on dst.
func (t *Translator) Stream(dst flusher, src io.Reader) (*openAIUsage, error) {
	agg := newAggregate()

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
		case "message_start":
			if evt.Message != nil && evt.Message.Usage != nil {
				agg.inputTokens = evt.Message.Usage.InputTokens
			}
		case "content_block_start":
			if evt.ContentBlock != nil && evt.ContentBlock.Type == "tool_use" {
				ta := agg.startTool(evt.Index, evt.ContentBlock.ID, evt.ContentBlock.Name)
				t.writeChunk(dst, t.toolStartChunk(ta))
			}
		case "content_block_delta":
			if evt.Delta == nil {
				continue
			}
			switch evt.Delta.Type {
			case "text_delta":
				if evt.Delta.Text != "" {
					t.writeChunk(dst, t.contentChunk(evt.Delta.Text))
				}
			case "input_json_delta":
				if ta, ok := agg.byBlockIndex[evt.Index]; ok && evt.Delta.PartialJSON != "" {
					ta.args.WriteString(evt.Delta.PartialJSON)
					t.writeChunk(dst, t.toolArgsChunk(ta, evt.Delta.PartialJSON))
				}
			}
		case "message_delta":
			if evt.Delta != nil && evt.Delta.StopReason != "" {
				agg.stopReason = evt.Delta.StopReason
			}
			if evt.Usage != nil {
				agg.outputTokens = evt.Usage.OutputTokens
			}
		case "error":
			if evt.Error != nil {
				return nil, fmt.Errorf("anthropic upstream error (%s): %s", evt.Error.Type, evt.Error.Message)
			}
		case "message_stop":
			// terminal — handled after the loop
		}
	}
	if err := sc.err(); err != nil {
		return nil, err
	}

	finish := finishReason(agg.stopReason, len(agg.toolCalls) > 0)
	usage := t.usage(agg)
	final := openAIChunk{
		ID: t.id, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
		Choices: []chunkChoice{{Index: 0, Delta: chunkDelta{}, FinishReason: &finish}},
		Usage:   usage,
	}
	t.writeChunk(dst, final)
	_, _ = io.WriteString(dst, "data: [DONE]\n\n")
	dst.Flush()
	return usage, nil
}

// Collect consumes the whole Anthropic SSE and returns a single OpenAI
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
		case "message_start":
			if evt.Message != nil && evt.Message.Usage != nil {
				agg.inputTokens = evt.Message.Usage.InputTokens
			}
		case "content_block_start":
			if evt.ContentBlock != nil && evt.ContentBlock.Type == "tool_use" {
				agg.startTool(evt.Index, evt.ContentBlock.ID, evt.ContentBlock.Name)
			}
		case "content_block_delta":
			if evt.Delta == nil {
				continue
			}
			switch evt.Delta.Type {
			case "text_delta":
				agg.content.WriteString(evt.Delta.Text)
			case "input_json_delta":
				if ta, ok := agg.byBlockIndex[evt.Index]; ok {
					ta.args.WriteString(evt.Delta.PartialJSON)
				}
			}
		case "message_delta":
			if evt.Delta != nil && evt.Delta.StopReason != "" {
				agg.stopReason = evt.Delta.StopReason
			}
			if evt.Usage != nil {
				agg.outputTokens = evt.Usage.OutputTokens
			}
		case "error":
			if evt.Error != nil {
				return nil, fmt.Errorf("anthropic upstream error (%s): %s", evt.Error.Type, evt.Error.Message)
			}
		}
	}
	if err := sc.err(); err != nil {
		return nil, err
	}

	msg := responseMessage{Role: "assistant"}
	for _, ta := range agg.toolCalls {
		var tc responseToolCall
		tc.ID = ta.id
		tc.Type = "function"
		tc.Function.Name = ta.name
		tc.Function.Arguments = ta.args.String()
		msg.ToolCalls = append(msg.ToolCalls, tc)
	}
	if c := agg.content.String(); c != "" || len(msg.ToolCalls) == 0 {
		msg.Content = &c
	}

	resp := openAIResponse{
		ID: t.id, Object: "chat.completion", Created: t.created, Model: t.model,
		Choices: []responseChoice{{Index: 0, Message: msg, FinishReason: finishReason(agg.stopReason, len(agg.toolCalls) > 0)}},
		Usage:   t.usage(agg),
	}
	return json.Marshal(resp)
}

func (t *Translator) usage(a *aggregate) *openAIUsage {
	if a.inputTokens == 0 && a.outputTokens == 0 {
		return nil
	}
	return &openAIUsage{
		PromptTokens:     a.inputTokens,
		CompletionTokens: a.outputTokens,
		TotalTokens:      a.inputTokens + a.outputTokens,
	}
}

// finishReason maps an Anthropic stop_reason to an OpenAI finish_reason.
func finishReason(stop string, hasTools bool) string {
	switch stop {
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "refusal":
		return "content_filter"
	case "end_turn", "stop_sequence", "pause_turn":
		return "stop"
	default:
		if hasTools {
			return "tool_calls"
		}
		return "stop"
	}
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

func randomID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// ── SSE scanner (mirrors internal/codex) ─────────────────────────

type sseScanner struct {
	r     *bufio.Reader
	block strings.Builder
	cur   anthropicEvent
	curOK bool
	rerr  error
}

func newSSEScanner(src io.Reader) *sseScanner {
	return &sseScanner{r: bufio.NewReaderSize(src, 64*1024)}
}

func (s *sseScanner) next() bool {
	s.block.Reset()
	for {
		line, err := s.r.ReadString('\n')
		if len(line) > 0 {
			trimmed := strings.TrimRight(line, "\r\n")
			if trimmed == "" {
				if s.block.Len() > 0 {
					s.parseBlock(s.block.String())
					return true
				}
				continue
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
	var evt anthropicEvent
	if json.Unmarshal([]byte(raw), &evt) != nil {
		return
	}
	s.cur = evt
	s.curOK = true
}

func (s *sseScanner) event() (anthropicEvent, bool) { return s.cur, s.curOK }
func (s *sseScanner) err() error                    { return s.rerr }
