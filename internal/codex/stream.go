package codex

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
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

	// Reopen, when set, starts a fresh upstream attempt after the current one
	// fails with ErrStalled, as long as no content has reached the client.
	Reopen func() (io.ReadCloser, error)
}

// ErrStalled marks an upstream body that went silent past its watchdog. The
// ChatGPT backend sometimes answers 200 then sends nothing, while an identical
// request completes in seconds (openai/codex#32987).
var ErrStalled = errors.New("codex upstream stalled")

// maxReopens bounds the fresh attempts one response may make after stalls.
const maxReopens = 2

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
	// emitted is set once a content or tool chunk has been produced.
	emitted  bool
	errEvent *struct {
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
	// Opening role chunk.
	t.writeChunk(dst, openAIChunk{
		ID: t.id, Object: "chat.completion.chunk", Created: t.created, Model: t.model,
		Choices: []chunkChoice{{Index: 0, Delta: chunkDelta{Role: "assistant"}}},
	})

	emit := func(c openAIChunk) { t.writeChunk(dst, c) }
	agg, err := t.consume(src, emit)
	// Only the role chunk has reached the client, so a fresh attempt is invisible.
	for n := 0; errors.Is(err, ErrStalled) && !agg.emitted && n < maxReopens && t.Reopen != nil; n++ {
		agg, err = t.reopenAndConsume(emit)
	}
	if err != nil {
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
	agg, err := t.consume(src, nil)
	for n := 0; errors.Is(err, ErrStalled) && n < maxReopens && t.Reopen != nil; n++ {
		agg, err = t.reopenAndConsume(nil)
	}
	if err != nil {
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
		ID: t.id, Object: "chat.completion",
		Created: t.created, Model: t.model,
		Choices: []responseChoice{{Index: 0, Message: msg, FinishReason: finish}},
		Usage:   toOpenAIUsage(agg.usage),
	}
	return json.Marshal(resp)
}

func (t *Translator) reopenAndConsume(emit func(openAIChunk)) (*aggregate, error) {
	src, err := t.Reopen()
	if err != nil {
		return newAggregate(), err
	}
	defer src.Close()
	return t.consume(src, emit)
}

// consume parses one upstream attempt. With emit set (streaming) each content
// and tool chunk goes out as it arrives; without it the text is accumulated
// for Collect.
func (t *Translator) consume(src io.Reader, emit func(openAIChunk)) (*aggregate, error) {
	agg := newAggregate()
	out := func(c openAIChunk) {
		agg.emitted = true
		if emit != nil {
			emit(c)
		}
	}
	sc := newSSEScanner(src)
	for sc.next() {
		evt, ok := sc.event()
		if !ok {
			continue
		}
		switch evt.Type {
		case "response.output_text.delta":
			if evt.Delta != "" {
				if emit == nil {
					agg.content.WriteString(evt.Delta)
				}
				out(t.contentChunk(evt.Delta))
			}
		case "response.output_item.added":
			if evt.Item != nil && evt.Item.Type == "function_call" && evt.Item.CallID != "" {
				ta := agg.startTool(evt.Item.CallID, evt.Item.Name)
				agg.itemIDToCall[evt.Item.ID] = ta
				out(t.toolStartChunk(ta))
			}
		case "response.function_call_arguments.delta":
			ta := agg.resolveTool(evt.argCallID(), "")
			if ta != nil && evt.Delta != "" {
				ta.args.WriteString(evt.Delta)
				out(t.toolArgsChunk(ta, evt.Delta))
			}
		case "response.function_call_arguments.done":
			// Parallel calls can arrive with no deltas, arguments only here.
			ta := agg.resolveTool(evt.argCallID(), evt.Name)
			if ta != nil && ta.args.Len() == 0 && evt.Args != "" {
				ta.args.WriteString(evt.Args)
				out(t.toolArgsChunk(ta, evt.Args))
			}
		case "response.completed", "response.incomplete":
			if evt.Response != nil && evt.Response.Usage != nil {
				agg.usage = evt.Response.Usage
			}
		case "error", "response.failed":
			if evt.Error != nil {
				return agg, fmt.Errorf("codex upstream error (%s): %s", evt.Error.Code, evt.Error.Message)
			}
		}
	}
	return agg, sc.err()
}

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
	raw   string
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
	s.raw = raw
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
