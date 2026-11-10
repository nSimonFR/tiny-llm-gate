// Package router picks a target model per user turn by asking a decision
// model (TypeSafe Jev) yes/no questions about the latest user message.
package router

import (
	"encoding/json"
	"strings"
)

// Msg is one human-readable message of a conversation.
type Msg struct {
	Role string // "user" or "assistant"
	Text string
}

// Turn is a conversation reduced to its system prompt and user/assistant
// text. Tool calls and tool results are dropped: a request that only carries
// a tool result continues the current turn, it does not start a new one.
type Turn struct {
	System string
	Msgs   []Msg
}

// lastUser is the index of the last user message, or -1.
func (t Turn) lastUser() int {
	for i := len(t.Msgs) - 1; i >= 0; i-- {
		if t.Msgs[i].Role == "user" {
			return i
		}
	}
	return -1
}

// textOf flattens a string or an array of content parts, keeping parts whose
// type is in kinds (any part with a "text" field when kinds is empty).
func textOf(raw json.RawMessage, kinds ...string) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return ""
	}
	var b strings.Builder
	for _, p := range parts {
		if p.Text == "" || (len(kinds) > 0 && !contains(kinds, p.Type)) {
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(p.Text)
	}
	return b.String()
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// hasPart reports whether a content array holds a part of one of these types.
func hasPart(raw json.RawMessage, kinds ...string) bool {
	var parts []struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return false
	}
	for _, p := range parts {
		if contains(kinds, p.Type) {
			return true
		}
	}
	return false
}

func (t *Turn) add(role, text string) {
	if text = strings.TrimSpace(text); text != "" {
		t.Msgs = append(t.Msgs, Msg{Role: role, Text: text})
	}
}

func (t *Turn) addSystem(text string) {
	if text = strings.TrimSpace(text); text != "" {
		if t.System != "" {
			t.System += "\n"
		}
		t.System += text
	}
}

// FromChat reads an OpenAI chat/completions body.
func FromChat(body []byte) (Turn, error) {
	var in struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	var t Turn
	if err := json.Unmarshal(body, &in); err != nil {
		return t, err
	}
	for _, m := range in.Messages {
		switch m.Role {
		case "system", "developer":
			t.addSystem(textOf(m.Content))
		case "user", "assistant":
			t.add(m.Role, textOf(m.Content))
		}
	}
	return t, nil
}

// FromResponses reads an OpenAI /v1/responses body.
func FromResponses(body []byte) (Turn, error) {
	var in struct {
		Instructions string          `json:"instructions"`
		Input        json.RawMessage `json:"input"`
	}
	var t Turn
	if err := json.Unmarshal(body, &in); err != nil {
		return t, err
	}
	t.addSystem(in.Instructions)
	var s string
	if json.Unmarshal(in.Input, &s) == nil {
		t.add("user", s)
		return t, nil
	}
	var items []struct {
		Type    string          `json:"type"`
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if len(in.Input) > 0 {
		if err := json.Unmarshal(in.Input, &items); err != nil {
			return t, err
		}
	}
	for _, it := range items {
		if it.Type != "" && it.Type != "message" {
			continue // function_call, function_call_output, reasoning, …
		}
		switch it.Role {
		case "system", "developer":
			t.addSystem(textOf(it.Content))
		case "user", "assistant":
			t.add(it.Role, textOf(it.Content))
		}
	}
	return t, nil
}

// FromGemini reads a Gemini generateContent body.
func FromGemini(body []byte) (Turn, error) {
	type part struct {
		Text             string          `json:"text"`
		FunctionResponse json.RawMessage `json:"functionResponse"`
	}
	var in struct {
		SystemInstruction *struct {
			Parts []part `json:"parts"`
		} `json:"systemInstruction"`
		Contents []struct {
			Role  string `json:"role"`
			Parts []part `json:"parts"`
		} `json:"contents"`
	}
	var t Turn
	if err := json.Unmarshal(body, &in); err != nil {
		return t, err
	}
	if in.SystemInstruction != nil {
		for _, p := range in.SystemInstruction.Parts {
			t.addSystem(p.Text)
		}
	}
	for _, c := range in.Contents {
		role := "user"
		if c.Role == "model" {
			role = "assistant"
		}
		var b strings.Builder
		toolResult := false
		for _, p := range c.Parts {
			if len(p.FunctionResponse) > 0 {
				toolResult = true
			}
			if p.Text != "" {
				if b.Len() > 0 {
					b.WriteByte('\n')
				}
				b.WriteString(p.Text)
			}
		}
		if role == "user" && toolResult {
			continue
		}
		t.add(role, b.String())
	}
	return t, nil
}

// FromMessages reads an Anthropic /v1/messages body. A user message carrying
// a tool_result is skipped even when it also has text: Claude Code appends
// reminders to tool results, and those must not open a new turn.
func FromMessages(body []byte) (Turn, error) {
	var in struct {
		System   json.RawMessage `json:"system"`
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	var t Turn
	if err := json.Unmarshal(body, &in); err != nil {
		return t, err
	}
	t.addSystem(textOf(in.System, "text"))
	for _, m := range in.Messages {
		if m.Role == "user" && hasPart(m.Content, "tool_result") {
			continue
		}
		if m.Role == "user" || m.Role == "assistant" {
			t.add(m.Role, textOf(m.Content, "text"))
		}
	}
	return t, nil
}
