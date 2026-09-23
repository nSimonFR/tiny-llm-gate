// Package codex translates between the OpenAI Chat Completions wire format and
// the ChatGPT/Codex Responses API (POST /backend-api/codex/responses).
//
// It exists so tiny-llm-gate can talk to a ChatGPT subscription directly,
// replacing the standalone codex-proxy service. Only the surface actually used
// by our clients is modelled: chat with function/tool calls, reasoning effort,
// structured output (response_format), and multimodal text+image input. Hosted
// tools (web_search, image_generation) are passed through opaquely.
//
// Ground truth for the mapping is icebear0828/codex-proxy's src/translation/
// {openai-to-codex,codex-to-openai}.ts and src/types/codex-events.ts.
package codex

import "encoding/json"

// ── OpenAI request (inbound to the gate) ─────────────────────────

// ChatRequest is the subset of an OpenAI /v1/chat/completions body we read.
// Unmodelled fields are ignored (the Codex Responses API doesn't accept most
// OpenAI sampling params anyway).
type ChatRequest struct {
	Model           string          `json:"model"`
	Messages        []ChatMessage   `json:"messages"`
	Stream          bool            `json:"stream"`
	Tools           []OpenAITool    `json:"tools,omitempty"`
	ToolChoice      json.RawMessage `json:"tool_choice,omitempty"`
	ReasoningEffort string          `json:"reasoning_effort,omitempty"`
	ResponseFormat  *ResponseFormat `json:"response_format,omitempty"`
}

// ChatMessage is one OpenAI message. Content is a raw message because it may be
// a plain string or an array of typed parts (multimodal).
type ChatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content,omitempty"`
	ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Name       string          `json:"name,omitempty"`
}

// ToolCall is an assistant's request to call a function.
type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// OpenAITool is a tool definition (function calling).
type OpenAITool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters,omitempty"`
		Strict      *bool           `json:"strict,omitempty"`
	} `json:"function"`
}

// ResponseFormat is the structured-output selector.
type ResponseFormat struct {
	Type       string `json:"type"`
	JSONSchema *struct {
		Name   string          `json:"name"`
		Schema json.RawMessage `json:"schema"`
		Strict *bool           `json:"strict,omitempty"`
	} `json:"json_schema,omitempty"`
}

// contentPart is one element of a multimodal content array.
type contentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL *struct {
		URL string `json:"url"`
	} `json:"image_url,omitempty"`
}

// ── Codex Responses request (outbound to the backend) ────────────

type codexRequest struct {
	Model        string          `json:"model"`
	Instructions string          `json:"instructions"`
	Input        []codexInput    `json:"input"`
	Stream       bool            `json:"stream"`
	Store        bool            `json:"store"`
	Tools        []codexTool     `json:"tools,omitempty"`
	ToolChoice   json.RawMessage `json:"tool_choice,omitempty"`
	Reasoning    *codexReasoning `json:"reasoning,omitempty"`
	Text         *codexText      `json:"text,omitempty"`
}

// codexInput is one item in the Codex `input` array. It is a union: a plain
// message ({role, content}); a function call ({type:"function_call", call_id,
// name, arguments}); or a tool result ({type:"function_call_output", call_id,
// output}).
type codexInput struct {
	Type      string `json:"type,omitempty"`
	Role      string `json:"role,omitempty"`
	Content   any    `json:"content,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Output    string `json:"output,omitempty"`
}

// codexPart is a multimodal content part in Codex form.
type codexPart struct {
	Type     string `json:"type"` // input_text | input_image
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

type codexTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name,omitempty"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
	Strict      bool            `json:"strict,omitempty"`
}

type codexReasoning struct {
	Effort  string `json:"effort"`
	Summary string `json:"summary"`
}

type codexText struct {
	Format json.RawMessage `json:"format"`
}

// ── Codex SSE events (inbound from the backend) ──────────────────

// codexEvent is a decoded SSE data payload. Only the fields we act on are
// modelled; the discriminant is `type`.
type codexEvent struct {
	Type   string `json:"type"`
	Delta  string `json:"delta"`
	CallID string `json:"call_id"`
	ItemID string `json:"item_id"`
	Name   string `json:"name"`
	Args   string `json:"arguments"`
	Item   *struct {
		Type   string `json:"type"`
		ID     string `json:"id"`
		CallID string `json:"call_id"`
		Name   string `json:"name"`
	} `json:"item"`
	Response *struct {
		ID    string      `json:"id"`
		Usage *codexUsage `json:"usage"`
	} `json:"response"`
	Error *struct {
		Type    string `json:"type"`
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

type codexUsage struct {
	InputTokens     int `json:"input_tokens"`
	OutputTokens    int `json:"output_tokens"`
	CachedTokens    int `json:"cached_tokens"`
	ReasoningTokens int `json:"reasoning_tokens"`
}

// argCallID returns the identifier tying a response.function_call_arguments.*
// event to its tool accumulator. The live Codex backend keys these events by
// item_id (the output item's id), while response.output_item.added carries
// both that item id and the real call_id. Prefer call_id (as
// codex-proxy/synthetic streams may send it), fall back to item_id — the
// aggregate maps item ids to their tool via itemIDToCall.
func (e codexEvent) argCallID() string {
	if e.CallID != "" {
		return e.CallID
	}
	return e.ItemID
}

// ── OpenAI response/chunk (outbound to the client) ───────────────

type openAIChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []chunkChoice `json:"choices"`
	Usage   *openAIUsage  `json:"usage,omitempty"`
}

type chunkChoice struct {
	Index        int        `json:"index"`
	Delta        chunkDelta `json:"delta"`
	FinishReason *string    `json:"finish_reason"`
}

type chunkDelta struct {
	Role      string          `json:"role,omitempty"`
	Content   string          `json:"content,omitempty"`
	ToolCalls []chunkToolCall `json:"tool_calls,omitempty"`
}

type chunkToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// non-streaming completion response
type openAIResponse struct {
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []responseChoice `json:"choices"`
	Usage   *openAIUsage     `json:"usage,omitempty"`
}

type responseChoice struct {
	Index        int             `json:"index"`
	Message      responseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

type responseMessage struct {
	Role      string             `json:"role"`
	Content   *string            `json:"content"`
	ToolCalls []responseToolCall `json:"tool_calls,omitempty"`
}

type responseToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
