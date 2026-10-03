// Package responses serves the OpenAI Responses API (POST /v1/responses) on
// top of the gate's chat core: requests become chat/completions bodies, and
// the chat response is re-encoded as a Response object or its SSE events.
// Stateless: previous_response_id and stored responses are not supported.
package responses

import (
	"encoding/json"
	"errors"
	"fmt"
)

// Request is the subset of a Responses API body the gate understands.
type Request struct {
	Model        string          `json:"model"`
	Input        json.RawMessage `json:"input"`
	Instructions string          `json:"instructions,omitempty"`
	Tools        []Tool          `json:"tools,omitempty"`
	ToolChoice   json.RawMessage `json:"tool_choice,omitempty"`
	Stream       bool            `json:"stream,omitempty"`
	Reasoning    *struct {
		Effort string `json:"effort,omitempty"`
	} `json:"reasoning,omitempty"`
	Text *struct {
		Format json.RawMessage `json:"format,omitempty"`
	} `json:"text,omitempty"`
	MaxOutputTokens    *int            `json:"max_output_tokens,omitempty"`
	Temperature        *float64        `json:"temperature,omitempty"`
	TopP               *float64        `json:"top_p,omitempty"`
	ParallelToolCalls  *bool           `json:"parallel_tool_calls,omitempty"`
	PreviousResponseID string          `json:"previous_response_id,omitempty"`
	Metadata           json.RawMessage `json:"metadata,omitempty"`
	User               string          `json:"user,omitempty"`
}

// Tool is a Responses function tool (flat: name/parameters at top level).
type Tool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

type inputItem struct {
	Type      string          `json:"type"`
	Role      string          `json:"role"`
	Content   json.RawMessage `json:"content"`
	CallID    string          `json:"call_id"`
	Name      string          `json:"name"`
	Arguments string          `json:"arguments"`
	Output    json.RawMessage `json:"output"`
}

type inputPart struct {
	Type     string          `json:"type"`
	Text     string          `json:"text"`
	ImageURL json.RawMessage `json:"image_url"`
	FileData string          `json:"file_data"`
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
		Strict      *bool           `json:"strict,omitempty"`
	} `json:"function"`
}

type chatRequest struct {
	Model             string          `json:"model"`
	Messages          []chatMessage   `json:"messages"`
	Stream            bool            `json:"stream"`
	StreamOptions     *streamOptions  `json:"stream_options,omitempty"`
	Tools             []chatTool      `json:"tools,omitempty"`
	ToolChoice        json.RawMessage `json:"tool_choice,omitempty"`
	ReasoningEffort   string          `json:"reasoning_effort,omitempty"`
	ResponseFormat    json.RawMessage `json:"response_format,omitempty"`
	MaxTokens         *int            `json:"max_tokens,omitempty"`
	Temperature       *float64        `json:"temperature,omitempty"`
	TopP              *float64        `json:"top_p,omitempty"`
	ParallelToolCalls *bool           `json:"parallel_tool_calls,omitempty"`
	User              string          `json:"user,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// ErrUnsupported marks a request the stateless gate cannot serve (→ 400).
var ErrUnsupported = errors.New("unsupported")

// Parse decodes a Responses request body.
func Parse(body []byte) (*Request, error) {
	var req Request
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("invalid JSON body: %w", err)
	}
	if req.PreviousResponseID != "" {
		return nil, fmt.Errorf("%w: previous_response_id (the gate stores no responses; resend the full input)", ErrUnsupported)
	}
	return &req, nil
}

// ToChat builds the chat/completions body for one hop. The upstream is always
// asked to stream (with usage) so one decoder serves both client modes.
func (req *Request) ToChat(upstreamModel string) ([]byte, error) {
	msgs := make([]chatMessage, 0, 8)
	if req.Instructions != "" {
		msgs = append(msgs, chatMessage{Role: "system", Content: req.Instructions})
	}
	in, err := req.inputMessages()
	if err != nil {
		return nil, err
	}
	msgs = append(msgs, in...)

	out := chatRequest{
		Model:             upstreamModel,
		Messages:          msgs,
		Stream:            true,
		StreamOptions:     &streamOptions{IncludeUsage: true},
		ToolChoice:        toolChoice(req.ToolChoice),
		MaxTokens:         req.MaxOutputTokens,
		Temperature:       req.Temperature,
		TopP:              req.TopP,
		ParallelToolCalls: req.ParallelToolCalls,
		User:              req.User,
	}
	for _, t := range req.Tools {
		if t.Type != "function" {
			return nil, fmt.Errorf("%w: tool type %q (only function tools)", ErrUnsupported, t.Type)
		}
		var ct chatTool
		ct.Type = "function"
		ct.Function.Name = t.Name
		ct.Function.Description = t.Description
		ct.Function.Parameters = t.Parameters
		ct.Function.Strict = t.Strict
		out.Tools = append(out.Tools, ct)
	}
	if req.Reasoning != nil {
		out.ReasoningEffort = req.Reasoning.Effort
	}
	if req.Text != nil && len(req.Text.Format) > 0 {
		out.ResponseFormat = responseFormat(req.Text.Format)
	}
	return json.Marshal(out)
}

func (req *Request) inputMessages() ([]chatMessage, error) {
	if len(req.Input) == 0 {
		return nil, errors.New("missing 'input'")
	}
	var s string
	if json.Unmarshal(req.Input, &s) == nil {
		return []chatMessage{{Role: "user", Content: s}}, nil
	}
	var items []inputItem
	if err := json.Unmarshal(req.Input, &items); err != nil {
		return nil, fmt.Errorf("'input' must be a string or an array of items: %w", err)
	}
	var out []chatMessage
	for _, it := range items {
		switch it.Type {
		case "", "message":
			role := it.Role
			if role == "developer" {
				role = "system"
			}
			content, err := messageContent(it.Content)
			if err != nil {
				return nil, err
			}
			out = append(out, chatMessage{Role: role, Content: content})
		case "function_call":
			var call chatCall
			call.ID, call.Type = it.CallID, "function"
			call.Function.Name, call.Function.Arguments = it.Name, it.Arguments
			// A turn's text item and its calls fold into one assistant message.
			if n := len(out); n > 0 && out[n-1].Role == "assistant" {
				out[n-1].ToolCalls = append(out[n-1].ToolCalls, call)
			} else {
				out = append(out, chatMessage{Role: "assistant", ToolCalls: []chatCall{call}})
			}
		case "function_call_output":
			out = append(out, chatMessage{Role: "tool", ToolCallID: it.CallID, Content: outputText(it.Output)})
		case "reasoning":
			// Opaque reasoning state from a prior turn — chat has no slot for it.
		default:
			return nil, fmt.Errorf("%w: input item type %q", ErrUnsupported, it.Type)
		}
	}
	return out, nil
}

// messageContent maps Responses content (string or typed parts) to chat
// content: a plain string when text-only, parts when images are present.
func messageContent(raw json.RawMessage) (any, error) {
	if len(raw) == 0 {
		return "", nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s, nil
	}
	var parts []inputPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, fmt.Errorf("invalid message content: %w", err)
	}
	var text string
	var out []chatPart
	hasImage := false
	for _, p := range parts {
		switch p.Type {
		case "input_text", "output_text", "text":
			out = append(out, chatPart{Type: "text", Text: p.Text})
			if text != "" {
				text += "\n"
			}
			text += p.Text
		case "input_image":
			url := imageURL(p.ImageURL)
			if url == "" {
				return nil, fmt.Errorf("%w: input_image without image_url (file_id is not supported)", ErrUnsupported)
			}
			cp := chatPart{Type: "image_url"}
			cp.ImageURL = &struct {
				URL string `json:"url"`
			}{URL: url}
			out = append(out, cp)
			hasImage = true
		case "refusal":
		default:
			return nil, fmt.Errorf("%w: content part type %q", ErrUnsupported, p.Type)
		}
	}
	if !hasImage {
		return text, nil
	}
	return out, nil
}

// imageURL accepts both the Responses string form and the chat object form.
func imageURL(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var obj struct {
		URL string `json:"url"`
	}
	_ = json.Unmarshal(raw, &obj)
	return obj.URL
}

func outputText(raw json.RawMessage) string {
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []inputPart
	if json.Unmarshal(raw, &parts) == nil {
		var text string
		for _, p := range parts {
			if p.Text != "" {
				if text != "" {
					text += "\n"
				}
				text += p.Text
			}
		}
		return text
	}
	return string(raw)
}

// toolChoice maps {type:function,name} to chat's nested form; strings pass.
func toolChoice(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	var obj struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if json.Unmarshal(raw, &obj) == nil && obj.Type == "function" && obj.Name != "" {
		out, _ := json.Marshal(map[string]any{"type": "function", "function": map[string]string{"name": obj.Name}})
		return out
	}
	return raw
}

// responseFormat maps text.format ({type:json_schema,name,schema,strict}) to
// chat's response_format ({type:json_schema,json_schema:{...}}).
func responseFormat(raw json.RawMessage) json.RawMessage {
	var f struct {
		Type   string          `json:"type"`
		Name   string          `json:"name"`
		Schema json.RawMessage `json:"schema"`
		Strict *bool           `json:"strict"`
	}
	if json.Unmarshal(raw, &f) != nil {
		return nil
	}
	switch f.Type {
	case "json_schema":
		js := map[string]any{"name": f.Name, "schema": f.Schema}
		if f.Strict != nil {
			js["strict"] = *f.Strict
		}
		out, _ := json.Marshal(map[string]any{"type": "json_schema", "json_schema": js})
		return out
	case "json_object":
		return json.RawMessage(`{"type":"json_object"}`)
	}
	return nil
}
