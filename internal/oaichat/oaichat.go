// Package oaichat decodes an OpenAI chat.completion(.chunk) response — the
// shape every provider type yields from the gate's chat core — into a flat
// event sequence. The Responses and Messages frontends re-encode from it.
package oaichat

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// Usage is the OpenAI usage block (only what the frontends re-emit).
type Usage struct {
	PromptTokens        int `json:"prompt_tokens"`
	CompletionTokens    int `json:"completion_tokens"`
	TotalTokens         int `json:"total_tokens"`
	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *struct {
		ReasoningTokens int `json:"reasoning_tokens"`
	} `json:"completion_tokens_details,omitempty"`
}

func (u *Usage) Cached() int {
	if u == nil || u.PromptTokensDetails == nil {
		return 0
	}
	return u.PromptTokensDetails.CachedTokens
}

func (u *Usage) Reasoning() int {
	if u == nil || u.CompletionTokensDetails == nil {
		return 0
	}
	return u.CompletionTokensDetails.ReasoningTokens
}

type toolCallDelta struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type message struct {
	Content          *string         `json:"content"`
	Reasoning        string          `json:"reasoning"`
	ReasoningContent string          `json:"reasoning_content"`
	ToolCalls        []toolCallDelta `json:"tool_calls"`
}

type choice struct {
	Delta        *message `json:"delta"`
	Message      *message `json:"message"`
	FinishReason *string  `json:"finish_reason"`
}

type payload struct {
	ID      string   `json:"id"`
	Choices []choice `json:"choices"`
	Usage   *Usage   `json:"usage"`
	Error   *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// Handler receives decoded events in order. A tool call's first event has
// ID/Name set; later ones carry only argument fragments. Index is the
// upstream tool_calls index. Returning an error stops decoding.
type Handler interface {
	Text(delta string) error
	Reasoning(delta string) error
	ToolCall(index int, id, name, argsDelta string) error
	Finish(reason string) error
	Usage(u *Usage) error
}

// Decode reads either an SSE chunk stream or a single chat.completion JSON
// document from r, by sniffing the first non-space byte.
func Decode(r io.Reader, h Handler) error {
	br := bufio.NewReaderSize(r, 4096)
	for {
		b, err := br.Peek(1)
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		if b[0] == ' ' || b[0] == '\n' || b[0] == '\r' || b[0] == '\t' {
			_, _ = br.ReadByte()
			continue
		}
		if b[0] == '{' {
			return decodeJSON(br, h)
		}
		return decodeSSE(br, h)
	}
}

func decodeJSON(r io.Reader, h Handler) error {
	var p payload
	if err := json.NewDecoder(r).Decode(&p); err != nil {
		return fmt.Errorf("decode chat completion: %w", err)
	}
	return dispatch(&p, h, false)
}

func decodeSSE(br *bufio.Reader, h Handler) error {
	for {
		line, err := br.ReadBytes('\n')
		line = bytes.TrimRight(line, "\r\n")
		if bytes.HasPrefix(line, []byte("data:")) {
			data := bytes.TrimSpace(line[len("data:"):])
			if len(data) > 0 && !bytes.Equal(data, []byte("[DONE]")) {
				var p payload
				if jerr := json.Unmarshal(data, &p); jerr == nil {
					if derr := dispatch(&p, h, true); derr != nil {
						return derr
					}
				}
			}
		}
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
	}
}

func dispatch(p *payload, h Handler, stream bool) error {
	if p.Error != nil {
		return fmt.Errorf("upstream error: %s", p.Error.Message)
	}
	for _, c := range p.Choices {
		m := c.Delta
		if !stream || m == nil {
			if c.Message != nil {
				m = c.Message
			}
		}
		if m != nil {
			if r := m.Reasoning + m.ReasoningContent; r != "" {
				if err := h.Reasoning(r); err != nil {
					return err
				}
			}
			if m.Content != nil && *m.Content != "" {
				if err := h.Text(*m.Content); err != nil {
					return err
				}
			}
			for i, tc := range m.ToolCalls {
				idx := tc.Index
				if !stream {
					idx = i // a complete message's tool_calls carry no index
				}
				if err := h.ToolCall(idx, tc.ID, tc.Function.Name, tc.Function.Arguments); err != nil {
					return err
				}
			}
		}
		if c.FinishReason != nil && *c.FinishReason != "" {
			if err := h.Finish(*c.FinishReason); err != nil {
				return err
			}
		}
		break // n>1 choices are not re-encoded
	}
	if p.Usage != nil {
		return h.Usage(p.Usage)
	}
	return nil
}

// ErrorMessage extracts a human message from an upstream error body.
func ErrorMessage(body []byte) string {
	var env struct {
		Error json.RawMessage `json:"error"`
	}
	if json.Unmarshal(body, &env) == nil && len(env.Error) > 0 {
		var obj struct {
			Message string `json:"message"`
		}
		if json.Unmarshal(env.Error, &obj) == nil && obj.Message != "" {
			return obj.Message
		}
		var s string
		if json.Unmarshal(env.Error, &s) == nil && s != "" {
			return s
		}
	}
	return strings.TrimSpace(string(body))
}
