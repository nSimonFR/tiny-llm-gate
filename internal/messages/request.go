// Package messages serves the Anthropic Messages API (POST /v1/messages) for
// models that are NOT Anthropic: requests become chat/completions bodies and
// the chat response is re-encoded as a Message or its SSE events. This is
// the reverse of internal/anthropic.
package messages

import (
	"encoding/json"
	"fmt"
	"strings"
)

type Request struct {
	Model         string          `json:"model"`
	MaxTokens     *int            `json:"max_tokens,omitempty"`
	System        json.RawMessage `json:"system,omitempty"`
	Messages      []Message       `json:"messages"`
	Tools         []Tool          `json:"tools,omitempty"`
	ToolChoice    *ToolChoice     `json:"tool_choice,omitempty"`
	StopSequences []string        `json:"stop_sequences,omitempty"`
	Temperature   *float64        `json:"temperature,omitempty"`
	TopP          *float64        `json:"top_p,omitempty"`
	Stream        bool            `json:"stream,omitempty"`
	Thinking      *struct {
		Type         string `json:"type"`
		BudgetTokens int    `json:"budget_tokens"`
	} `json:"thinking,omitempty"`
}

type Message struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type Tool struct {
	Type        string          `json:"type,omitempty"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
}

type ToolChoice struct {
	Type                   string `json:"type"`
	Name                   string `json:"name,omitempty"`
	DisableParallelToolUse bool   `json:"disable_parallel_tool_use,omitempty"`
}

type block struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	Source    *imageSource    `json:"source"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"`
	IsError   bool            `json:"is_error"`
}

type imageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
	URL       string `json:"url"`
}

type chatMessage struct {
	Role       string     `json:"role"`
	Content    any        `json:"content"`
	ToolCalls  []chatCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type chatCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type chatPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL *struct {
		URL string `json:"url"`
	} `json:"image_url,omitempty"`
}

type chatTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters,omitempty"`
	} `json:"function"`
}

type chatRequest struct {
	Model             string          `json:"model"`
	Messages          []chatMessage   `json:"messages"`
	Stream            bool            `json:"stream"`
	StreamOptions     map[string]bool `json:"stream_options,omitempty"`
	Tools             []chatTool      `json:"tools,omitempty"`
	ToolChoice        any             `json:"tool_choice,omitempty"`
	ParallelToolCalls *bool           `json:"parallel_tool_calls,omitempty"`
	ReasoningEffort   string          `json:"reasoning_effort,omitempty"`
	MaxTokens         *int            `json:"max_tokens,omitempty"`
	Stop              []string        `json:"stop,omitempty"`
	Temperature       *float64        `json:"temperature,omitempty"`
	TopP              *float64        `json:"top_p,omitempty"`
}

func Parse(body []byte) (*Request, error) {
	var req Request
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("invalid JSON body: %w", err)
	}
	return &req, nil
}

// ToChat builds the chat/completions body for one hop. The upstream is always
// asked to stream (with usage) so one decoder serves both client modes.
func (req *Request) ToChat(upstreamModel string) ([]byte, error) {
	var msgs []chatMessage
	if sys := systemText(req.System); sys != "" {
		msgs = append(msgs, chatMessage{Role: "system", Content: sys})
	}
	for _, m := range req.Messages {
		out, err := convertMessage(m)
		if err != nil {
			return nil, err
		}
		msgs = append(msgs, out...)
	}
	out := chatRequest{
		Model:         upstreamModel,
		Messages:      msgs,
		Stream:        true,
		StreamOptions: map[string]bool{"include_usage": true},
		MaxTokens:     req.MaxTokens,
		Stop:          req.StopSequences,
		Temperature:   req.Temperature,
		TopP:          req.TopP,
	}
	for _, t := range req.Tools {
		if t.Type != "" && t.Type != "custom" {
			continue // server tools (web_search, bash, …) have no chat equivalent
		}
		var ct chatTool
		ct.Type = "function"
		ct.Function.Name = t.Name
		ct.Function.Description = t.Description
		ct.Function.Parameters = t.InputSchema
		out.Tools = append(out.Tools, ct)
	}
	if tc := req.ToolChoice; tc != nil && len(out.Tools) > 0 {
		switch tc.Type {
		case "auto":
			out.ToolChoice = "auto"
		case "any":
			out.ToolChoice = "required"
		case "none":
			out.ToolChoice = "none"
		case "tool":
			out.ToolChoice = map[string]any{"type": "function", "function": map[string]string{"name": tc.Name}}
		}
		if tc.DisableParallelToolUse {
			f := false
			out.ParallelToolCalls = &f
		}
	}
	if t := req.Thinking; t != nil && t.Type == "enabled" {
		switch {
		case t.BudgetTokens <= 4096:
			out.ReasoningEffort = "low"
		case t.BudgetTokens <= 16384:
			out.ReasoningEffort = "medium"
		default:
			out.ReasoningEffort = "high"
		}
	}
	return json.Marshal(out)
}

func systemText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var blocks []block
	_ = json.Unmarshal(raw, &blocks)
	var parts []string
	for _, b := range blocks {
		if b.Type == "text" && b.Text != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n\n")
}

func blocksOf(raw json.RawMessage) ([]block, error) {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return []block{{Type: "text", Text: s}}, nil
	}
	var blocks []block
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, fmt.Errorf("invalid message content: %w", err)
	}
	return blocks, nil
}

// convertMessage maps one Anthropic turn to chat messages. A user turn's
// tool_result blocks become separate role:"tool" messages ahead of its text.
func convertMessage(m Message) ([]chatMessage, error) {
	blocks, err := blocksOf(m.Content)
	if err != nil {
		return nil, err
	}
	if m.Role == "assistant" {
		var text strings.Builder
		var calls []chatCall
		for _, b := range blocks {
			switch b.Type {
			case "text":
				text.WriteString(b.Text)
			case "tool_use":
				var c chatCall
				c.ID, c.Type = b.ID, "function"
				c.Function.Name = b.Name
				c.Function.Arguments = string(b.Input)
				if c.Function.Arguments == "" {
					c.Function.Arguments = "{}"
				}
				calls = append(calls, c)
			}
		}
		msg := chatMessage{Role: "assistant", ToolCalls: calls}
		if text.Len() > 0 || len(calls) == 0 {
			msg.Content = text.String()
		}
		return []chatMessage{msg}, nil
	}

	var out []chatMessage
	var parts []chatPart
	hasImage := false
	for _, b := range blocks {
		switch b.Type {
		case "tool_result":
			content := resultText(b.Content)
			if b.IsError {
				content = "Error: " + content
			}
			out = append(out, chatMessage{Role: "tool", ToolCallID: b.ToolUseID, Content: content})
		case "text":
			parts = append(parts, chatPart{Type: "text", Text: b.Text})
		case "image":
			url := imageURL(b.Source)
			if url == "" {
				return nil, fmt.Errorf("unsupported image source")
			}
			p := chatPart{Type: "image_url"}
			p.ImageURL = &struct {
				URL string `json:"url"`
			}{URL: url}
			parts = append(parts, p)
			hasImage = true
		}
	}
	if len(parts) > 0 {
		var content any
		if hasImage {
			content = parts
		} else {
			var texts []string
			for _, p := range parts {
				texts = append(texts, p.Text)
			}
			content = strings.Join(texts, "\n")
		}
		out = append(out, chatMessage{Role: "user", Content: content})
	}
	return out, nil
}

func imageURL(s *imageSource) string {
	if s == nil {
		return ""
	}
	switch s.Type {
	case "base64":
		return "data:" + s.MediaType + ";base64," + s.Data
	case "url":
		return s.URL
	}
	return ""
}

func resultText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var blocks []block
	if json.Unmarshal(raw, &blocks) != nil {
		return string(raw)
	}
	var parts []string
	for _, b := range blocks {
		if b.Type == "text" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n")
}
