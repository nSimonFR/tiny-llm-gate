package codex

import (
	"encoding/json"
	"fmt"
)

const defaultInstructions = "You are a helpful assistant."

// TranslateRequest converts an OpenAI /v1/chat/completions body into a Codex
// Responses API body. It returns the marshalled Codex request plus the value
// that should be sent as the top-level model (already resolved upstream by the
// caller — TranslateRequest itself preserves whatever `model` is in the body).
//
// Mapping (see codex-proxy openai-to-codex.ts):
//   - system/developer messages are concatenated into `instructions`
//   - remaining messages become `input[]` items; assistant tool_calls become
//     function_call items and tool/function results become
//     function_call_output items
//   - tools/tool_choice are converted to Codex form
//   - reasoning_effort → reasoning{effort, summary:"auto"}
//   - response_format → text.format
//   - stream is always true upstream (we re-aggregate for non-streaming
//     clients); store is always false
func TranslateRequest(body []byte, upstreamModel string) ([]byte, error) {
	var req ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("codex: parse request: %w", err)
	}

	var instructions string
	input := make([]codexInput, 0, len(req.Messages))
	for _, m := range req.Messages {
		switch m.Role {
		case "system", "developer":
			text := messageText(m.Content)
			if text != "" {
				if instructions != "" {
					instructions += "\n\n"
				}
				instructions += text
			}
		case "assistant":
			text := messageText(m.Content)
			// Emit a text item when there's text, or when there are no tool
			// calls at all (an empty assistant turn still needs a placeholder).
			if text != "" || len(m.ToolCalls) == 0 {
				input = append(input, codexInput{Role: "assistant", Content: text})
			}
			for _, tc := range m.ToolCalls {
				input = append(input, codexInput{
					Type:      "function_call",
					CallID:    tc.ID,
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				})
			}
		case "tool":
			input = append(input, codexInput{
				Type:   "function_call_output",
				CallID: fallback(m.ToolCallID, "unknown"),
				Output: messageText(m.Content),
			})
		case "function":
			input = append(input, codexInput{
				Type:   "function_call_output",
				CallID: "fc_" + fallback(m.Name, "unknown"),
				Output: messageText(m.Content),
			})
		default: // "user" and anything else
			input = append(input, codexInput{Role: "user", Content: messageContent(m.Content)})
		}
	}

	if instructions == "" {
		instructions = defaultInstructions
	}
	if len(input) == 0 {
		input = append(input, codexInput{Role: "user", Content: ""})
	}

	out := codexRequest{
		Model:        upstreamModel,
		Instructions: instructions,
		Input:        input,
		Stream:       true,
		Store:        false,
		Tools:        translateTools(req.Tools),
	}
	if tc := translateToolChoice(req.ToolChoice); tc != nil {
		out.ToolChoice = tc
	}
	if req.ReasoningEffort != "" {
		out.Reasoning = &codexReasoning{Effort: req.ReasoningEffort, Summary: "auto"}
	}
	if f := translateResponseFormat(req.ResponseFormat); f != nil {
		out.Text = &codexText{Format: f}
	}

	return json.Marshal(out)
}

// messageText extracts plain text from an OpenAI content field (string or
// array of parts), dropping non-text parts.
func messageText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []contentPart
	if json.Unmarshal(raw, &parts) != nil {
		return ""
	}
	var text string
	for _, p := range parts {
		if p.Type == "text" && p.Text != "" {
			if text != "" {
				text += "\n"
			}
			text += p.Text
		}
	}
	return text
}

// messageContent returns either a plain string (text-only) or a []codexPart
// (when images are present), matching codex-proxy's extractContent.
func messageContent(raw json.RawMessage) any {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []contentPart
	if json.Unmarshal(raw, &parts) != nil {
		return ""
	}
	hasImage := false
	for _, p := range parts {
		if p.Type == "image_url" {
			hasImage = true
			break
		}
	}
	if !hasImage {
		var text string
		for _, p := range parts {
			if p.Type == "text" && p.Text != "" {
				if text != "" {
					text += "\n"
				}
				text += p.Text
			}
		}
		return text
	}
	out := make([]codexPart, 0, len(parts))
	for _, p := range parts {
		switch p.Type {
		case "text":
			if p.Text != "" {
				out = append(out, codexPart{Type: "input_text", Text: p.Text})
			}
		case "image_url":
			if p.ImageURL != nil && p.ImageURL.URL != "" {
				out = append(out, codexPart{Type: "input_image", ImageURL: p.ImageURL.URL})
			}
		}
	}
	if len(out) == 0 {
		return ""
	}
	return out
}

func translateTools(tools []OpenAITool) []codexTool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]codexTool, 0, len(tools))
	for _, t := range tools {
		if t.Type != "function" {
			// Pass hosted tools (web_search, image_generation) through by type.
			if t.Type != "" {
				out = append(out, codexTool{Type: t.Type})
			}
			continue
		}
		ct := codexTool{
			Type:       "function",
			Name:       t.Function.Name,
			Parameters: normalizeSchema(t.Function.Parameters),
		}
		if t.Function.Description != "" {
			ct.Description = t.Function.Description
		}
		if t.Function.Strict != nil {
			ct.Strict = *t.Function.Strict
		}
		out = append(out, ct)
	}
	return out
}

// normalizeSchema ensures an object schema carries a `properties` key, which
// the Codex backend requires (mirrors codex-proxy's normalizeSchema).
func normalizeSchema(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return raw
	}
	var m map[string]json.RawMessage
	if json.Unmarshal(raw, &m) != nil {
		return raw
	}
	typ, _ := m["type"]
	var typStr string
	_ = json.Unmarshal(typ, &typStr)
	if typStr == "object" {
		if _, ok := m["properties"]; !ok {
			m["properties"] = json.RawMessage(`{}`)
			if out, err := json.Marshal(m); err == nil {
				return out
			}
		}
	}
	return raw
}

// translateToolChoice converts an OpenAI tool_choice into Codex form:
// a string is passed through; {type:"function", function:{name}} becomes
// {type:"function", name}.
func translateToolChoice(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return raw // "none"|"auto"|"required" pass through verbatim
	}
	var obj struct {
		Type     string `json:"type"`
		Function *struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if json.Unmarshal(raw, &obj) != nil {
		return nil
	}
	if obj.Type == "function" && obj.Function != nil {
		out, _ := json.Marshal(map[string]string{"type": "function", "name": obj.Function.Name})
		return out
	}
	return raw
}

// translateResponseFormat maps OpenAI response_format → Codex text.format.
func translateResponseFormat(rf *ResponseFormat) json.RawMessage {
	if rf == nil || rf.Type == "" || rf.Type == "text" {
		return nil
	}
	switch rf.Type {
	case "json_object":
		return json.RawMessage(`{"type":"json_object"}`)
	case "json_schema":
		if rf.JSONSchema == nil {
			return nil
		}
		m := map[string]any{
			"type":   "json_schema",
			"name":   rf.JSONSchema.Name,
			"schema": rf.JSONSchema.Schema,
		}
		if rf.JSONSchema.Strict != nil {
			m["strict"] = *rf.JSONSchema.Strict
		}
		out, _ := json.Marshal(m)
		return out
	}
	return nil
}

func fallback(s, def string) string {
	if s == "" {
		return def
	}
	return s
}
