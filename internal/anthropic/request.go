package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"
)

// TranslateRequest converts an OpenAI /v1/chat/completions body into an
// Anthropic Messages API body (POST /v1/messages), always streamed upstream so
// non-streaming clients can be re-aggregated with clean token counts.
//
// Mapping:
//   - system/developer messages fold into `system`, after a Claude-Code sentinel
//   - user/assistant text → content blocks; assistant tool_calls → tool_use;
//     role:tool/function → a user-turn tool_result block
//   - tools → {name, description, input_schema}; tool_choice → auto|any|tool
//   - max_tokens is required (max_tokens / max_completion_tokens / default)
//   - image_url data: URLs → base64 image source; http(s) URLs → url source
//   - stop → stop_sequences; temperature/top_p pass through
func TranslateRequest(body []byte, upstreamModel string, defaultMaxTokens int) ([]byte, error) {
	var req ChatRequest
	if err := json.Unmarshal(body, &req); err != nil {
		return nil, fmt.Errorf("anthropic: parse request: %w", err)
	}

	// System: Claude-Code sentinel first, then any system/developer text.
	system := []anthropicText{{Type: "text", Text: claudeCodeSystemSentinel}}
	var sysExtra string
	for _, m := range req.Messages {
		if m.Role == "system" || m.Role == "developer" {
			if t := messageText(m.Content); t != "" {
				if sysExtra != "" {
					sysExtra += "\n\n"
				}
				sysExtra += t
			}
		}
	}
	if sysExtra != "" {
		system = append(system, anthropicText{Type: "text", Text: sysExtra})
	}

	// Messages: build (role, blocks) turns, merging consecutive same-role turns
	// (Anthropic requires merged content per role and strict user/assistant
	// alternation for tool_use/tool_result pairing).
	var msgs []anthropicMsg
	appendBlocks := func(role string, blocks []anthropicBlock) {
		if len(blocks) == 0 {
			return
		}
		if n := len(msgs); n > 0 && msgs[n-1].Role == role {
			existing := msgs[n-1].Content.([]anthropicBlock)
			msgs[n-1].Content = append(existing, blocks...)
			return
		}
		msgs = append(msgs, anthropicMsg{Role: role, Content: blocks})
	}

	for _, m := range req.Messages {
		switch m.Role {
		case "system", "developer":
			// handled above
		case "assistant":
			var blocks []anthropicBlock
			if t := messageText(m.Content); t != "" {
				blocks = append(blocks, anthropicBlock{Type: "text", Text: t})
			}
			for _, tc := range m.ToolCalls {
				input := json.RawMessage(tc.Function.Arguments)
				if len(input) == 0 || !json.Valid(input) {
					input = json.RawMessage(`{}`)
				}
				blocks = append(blocks, anthropicBlock{
					Type:  "tool_use",
					ID:    tc.ID,
					Name:  tc.Function.Name,
					Input: input,
				})
			}
			appendBlocks("assistant", blocks)
		case "tool":
			appendBlocks("user", []anthropicBlock{{
				Type:      "tool_result",
				ToolUseID: fallback(m.ToolCallID, "unknown"),
				Content:   messageText(m.Content),
			}})
		case "function":
			appendBlocks("user", []anthropicBlock{{
				Type:      "tool_result",
				ToolUseID: fallback(m.Name, "unknown"),
				Content:   messageText(m.Content),
			}})
		default: // "user"
			appendBlocks("user", userBlocks(m.Content))
		}
	}
	// Anthropic requires a non-empty messages array beginning with a user turn.
	if len(msgs) == 0 {
		msgs = append(msgs, anthropicMsg{Role: "user", Content: []anthropicBlock{{Type: "text", Text: ""}}})
	}

	maxTokens := defaultMaxTokens
	if req.MaxTokens != nil && *req.MaxTokens > 0 {
		maxTokens = *req.MaxTokens
	} else if req.MaxCompletion != nil && *req.MaxCompletion > 0 {
		maxTokens = *req.MaxCompletion
	}
	if maxTokens <= 0 {
		maxTokens = 4096
	}

	out := anthropicRequest{
		Model:         upstreamModel,
		MaxTokens:     maxTokens,
		System:        system,
		Messages:      msgs,
		Tools:         translateTools(req.Tools),
		Stream:        true,
		StopSequences: parseStop(req.Stop),
		Temperature:   req.Temperature,
		TopP:          req.TopP,
	}
	if tc := translateToolChoice(req.ToolChoice, len(out.Tools) > 0); tc != nil {
		out.ToolChoice = tc
	}
	return json.Marshal(out)
}

// messageText extracts plain text from an OpenAI content field (string or array
// of parts), dropping non-text parts.
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

// userBlocks builds Anthropic content blocks for a user message, handling
// multimodal image parts.
func userBlocks(raw json.RawMessage) []anthropicBlock {
	if len(raw) == 0 {
		return []anthropicBlock{{Type: "text", Text: ""}}
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return []anthropicBlock{{Type: "text", Text: s}}
	}
	var parts []contentPart
	if json.Unmarshal(raw, &parts) != nil {
		return []anthropicBlock{{Type: "text", Text: ""}}
	}
	var blocks []anthropicBlock
	for _, p := range parts {
		switch p.Type {
		case "text":
			if p.Text != "" {
				blocks = append(blocks, anthropicBlock{Type: "text", Text: p.Text})
			}
		case "image_url":
			if p.ImageURL != nil && p.ImageURL.URL != "" {
				blocks = append(blocks, imageBlock(p.ImageURL.URL))
			}
		}
	}
	if len(blocks) == 0 {
		return []anthropicBlock{{Type: "text", Text: ""}}
	}
	return blocks
}

// imageBlock converts an OpenAI image_url (a data: URL or an http(s) URL) into
// an Anthropic image content block.
func imageBlock(url string) anthropicBlock {
	if strings.HasPrefix(url, "data:") {
		// data:<media_type>;base64,<data>
		if i := strings.Index(url, ","); i >= 0 {
			meta := url[len("data:"):i]
			data := url[i+1:]
			mediaType := meta
			if semi := strings.Index(meta, ";"); semi >= 0 {
				mediaType = meta[:semi]
			}
			return anthropicBlock{Type: "image", Source: &anthropicImageSource{
				Type: "base64", MediaType: mediaType, Data: data,
			}}
		}
	}
	return anthropicBlock{Type: "image", Source: &anthropicImageSource{Type: "url", URL: url}}
}

func translateTools(tools []OpenAITool) []anthropicTool {
	if len(tools) == 0 {
		return nil
	}
	out := make([]anthropicTool, 0, len(tools))
	for _, t := range tools {
		if t.Type != "function" {
			continue
		}
		schema := t.Function.Parameters
		if len(schema) == 0 || !json.Valid(schema) {
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		out = append(out, anthropicTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: schema,
		})
	}
	return out
}

// translateToolChoice maps OpenAI tool_choice → Anthropic tool_choice. Returns
// nil (Anthropic default: auto when tools present) for "auto" and unknown forms.
func translateToolChoice(raw json.RawMessage, haveTools bool) json.RawMessage {
	if len(raw) == 0 || !haveTools {
		return nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		switch s {
		case "required":
			return json.RawMessage(`{"type":"any"}`)
		case "none":
			return json.RawMessage(`{"type":"none"}`)
		default: // "auto"
			return json.RawMessage(`{"type":"auto"}`)
		}
	}
	var obj struct {
		Type     string `json:"type"`
		Function *struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if json.Unmarshal(raw, &obj) == nil && obj.Type == "function" && obj.Function != nil {
		out, _ := json.Marshal(map[string]string{"type": "tool", "name": obj.Function.Name})
		return out
	}
	return nil
}

// parseStop maps an OpenAI stop (string or []string) to stop_sequences.
func parseStop(raw json.RawMessage) []string {
	if len(raw) == 0 {
		return nil
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		if s == "" {
			return nil
		}
		return []string{s}
	}
	var arr []string
	if json.Unmarshal(raw, &arr) == nil {
		return arr
	}
	return nil
}

func fallback(s, def string) string {
	if s == "" {
		return def
	}
	return s
}
