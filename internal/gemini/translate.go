package gemini

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// OpenAIChatRequest is the minimum shape we produce for upstream.
type OpenAIChatRequest struct {
	Model       string           `json:"model"`
	Messages    []OpenAIMessage  `json:"messages"`
	Stream      bool             `json:"stream,omitempty"`
	Temperature *float64         `json:"temperature,omitempty"`
	TopP        *float64         `json:"top_p,omitempty"`
	MaxTokens   *int             `json:"max_tokens,omitempty"`
	Stop        []string         `json:"stop,omitempty"`
	Tools       []OpenAITool     `json:"tools,omitempty"`
}

// OpenAITool is a function-calling tool definition.
type OpenAITool struct {
	Type     string         `json:"type"`
	Function OpenAIFunction `json:"function"`
}

// OpenAIFunction describes a callable function in OpenAI format.
type OpenAIFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// OpenAIMessage is a single message.
type OpenAIMessage struct {
	Role       string           `json:"role"`
	Content    string           `json:"content"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	ToolCalls  []OpenAIToolCall `json:"tool_calls,omitempty"`
}

// OpenAIToolCall is a tool invocation in a response.
type OpenAIToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// OpenAIChatResponse is the non-streaming response shape.
type OpenAIChatResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Model   string         `json:"model"`
	Created int64          `json:"created"`
	Choices []OpenAIChoice `json:"choices"`
	Usage   *OpenAIUsage   `json:"usage,omitempty"`
}

// OpenAIChoice is one of the Choices in a chat response.
type OpenAIChoice struct {
	Index        int              `json:"index"`
	Message      OpenAIChoiceMsg  `json:"message"`
	FinishReason string           `json:"finish_reason,omitempty"`
}

// OpenAIChoiceMsg extends OpenAIMessage with tool_calls for responses.
type OpenAIChoiceMsg struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	ToolCalls []OpenAIToolCall  `json:"tool_calls,omitempty"`
}

// OpenAIUsage mirrors the usage object we care about.
type OpenAIUsage struct {
	PromptTokens     int `json:"prompt_tokens,omitempty"`
	CompletionTokens int `json:"completion_tokens,omitempty"`
	TotalTokens      int `json:"total_tokens,omitempty"`
}

// OpenAIEmbedRequest is the minimum shape for the embeddings endpoint.
type OpenAIEmbedRequest struct {
	Model      string   `json:"model"`
	Input      []string `json:"input"`
	Dimensions *int     `json:"dimensions,omitempty"`
}

// OpenAIEmbedResponse is the embeddings response.
type OpenAIEmbedResponse struct {
	Data []OpenAIEmbedding `json:"data"`
}

// OpenAIEmbedding holds one vector.
type OpenAIEmbedding struct {
	Index     int       `json:"index"`
	Object    string    `json:"object,omitempty"`
	Embedding []float64 `json:"embedding"`
}

// OpenAIStreamChunk is one SSE chunk from /v1/chat/completions.
type OpenAIStreamChunk struct {
	ID      string                `json:"id,omitempty"`
	Model   string                `json:"model,omitempty"`
	Choices []OpenAIStreamChoice  `json:"choices"`
	Usage   *OpenAIUsage          `json:"usage,omitempty"`
}

// OpenAIStreamChoice is the delta portion of a streaming chunk.
type OpenAIStreamChoice struct {
	Index        int            `json:"index"`
	Delta        OpenAIDelta    `json:"delta"`
	FinishReason string         `json:"finish_reason,omitempty"`
}

// OpenAIDelta is a streaming token delta.
type OpenAIDelta struct {
	Role      string           `json:"role,omitempty"`
	Content   string           `json:"content,omitempty"`
	ToolCalls []OpenAIToolCall  `json:"tool_calls,omitempty"`
}

// ChatRequestToOpenAI converts a Gemini chat request into the OpenAI wire
// format. Handles text, function calls (model→assistant with tool_calls),
// and function responses (user→tool with tool_call_id).
func ChatRequestToOpenAI(in *ChatRequest, model string, stream bool) (*OpenAIChatRequest, error) {
	out := &OpenAIChatRequest{
		Model:  model,
		Stream: stream,
	}
	if in.SystemInstruction != nil {
		text := joinParts(in.SystemInstruction.Parts)
		if text != "" {
			out.Messages = append(out.Messages, OpenAIMessage{
				Role:    "system",
				Content: text,
			})
		}
	}

	// Track generated tool_call_ids so we can match functionResponse to
	// the correct prior assistant tool_call. Key: function name, value:
	// queue of IDs (multiple calls to the same function are possible).
	pendingIDs := make(map[string][]string) // name → []id
	callSeq := 0

	for _, c := range in.Contents {
		// Classify parts in this Content.
		var textParts []Part
		var fcParts []Part       // functionCall (model)
		var frParts []Part       // functionResponse (user)
		for _, p := range c.Parts {
			switch {
			case p.FunctionCall != nil:
				fcParts = append(fcParts, p)
			case p.FunctionResponse != nil:
				frParts = append(frParts, p)
			default:
				textParts = append(textParts, p)
			}
		}

		// Model message with function calls → assistant with tool_calls.
		if len(fcParts) > 0 {
			msg := OpenAIMessage{Role: "assistant"}
			text := joinParts(textParts)
			msg.Content = text
			for _, p := range fcParts {
				id := fmt.Sprintf("gemini_call_%d", callSeq)
				callSeq++
				pendingIDs[p.FunctionCall.Name] = append(pendingIDs[p.FunctionCall.Name], id)
				msg.ToolCalls = append(msg.ToolCalls, OpenAIToolCall{
					Index: len(msg.ToolCalls),
					ID:    id,
					Type:  "function",
					Function: struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					}{
						Name:      p.FunctionCall.Name,
						Arguments: string(p.FunctionCall.Args),
					},
				})
			}
			out.Messages = append(out.Messages, msg)
			continue
		}

		// Function responses → one "tool" message per response.
		if len(frParts) > 0 {
			for _, p := range frParts {
				fr := p.FunctionResponse
				// Find matching tool_call_id.
				toolCallID := ""
				if ids, ok := pendingIDs[fr.Name]; ok && len(ids) > 0 {
					toolCallID = ids[0]
					pendingIDs[fr.Name] = ids[1:]
				} else {
					// No matching call found; generate a synthetic ID.
					toolCallID = fmt.Sprintf("gemini_call_%d", callSeq)
					callSeq++
				}
				content := string(fr.Response)
				if content == "" {
					content = "{}"
				}
				out.Messages = append(out.Messages, OpenAIMessage{
					Role:       "tool",
					Content:    content,
					ToolCallID: toolCallID,
				})
			}
			// Also emit any text in the same Content as a user message.
			if text := joinParts(textParts); text != "" {
				out.Messages = append(out.Messages, OpenAIMessage{
					Role:    mapRoleToOpenAI(c.Role),
					Content: text,
				})
			}
			continue
		}

		// Plain text message.
		text := joinParts(textParts)
		if text == "" {
			continue
		}
		out.Messages = append(out.Messages, OpenAIMessage{
			Role:    mapRoleToOpenAI(c.Role),
			Content: text,
		})
	}
	if len(out.Messages) == 0 {
		return nil, errors.New("gemini: request has no text content")
	}
	if cfg := in.GenerationConfig; cfg != nil {
		out.Temperature = cfg.Temperature
		out.TopP = cfg.TopP
		out.MaxTokens = cfg.MaxOutputTokens
		out.Stop = cfg.StopSequences
	}
	// Forward tool (function calling) definitions.
	for _, t := range in.Tools {
		for _, fd := range t.FunctionDeclarations {
			out.Tools = append(out.Tools, OpenAITool{
				Type: "function",
				Function: OpenAIFunction{
					Name:        fd.Name,
					Description: fd.Description,
					Parameters:  fd.Parameters,
				},
			})
		}
	}
	return out, nil
}

// ChatResponseFromOpenAI converts a non-streaming OpenAI response back to
// Gemini's shape.
func ChatResponseFromOpenAI(resp *OpenAIChatResponse) *ChatResponse {
	out := &ChatResponse{Candidates: make([]Candidate, 0, len(resp.Choices))}
	for _, ch := range resp.Choices {
		parts := toolCallsToParts(ch.Message.ToolCalls)
		if ch.Message.Content != "" {
			parts = append([]Part{{Text: ch.Message.Content}}, parts...)
		}
		out.Candidates = append(out.Candidates, Candidate{
			Index: ch.Index,
			Content: Content{
				Role:  "model",
				Parts: parts,
			},
			FinishReason: mapFinishReasonToGemini(ch.FinishReason),
		})
	}
	if resp.Usage != nil {
		out.UsageMetadata = &UsageMetadata{
			PromptTokenCount:     resp.Usage.PromptTokens,
			CandidatesTokenCount: resp.Usage.CompletionTokens,
			TotalTokenCount:      resp.Usage.TotalTokens,
		}
	}
	return out
}

// ToolCallAccumulator buffers incremental OpenAI streaming tool-call deltas
// and produces complete Gemini FunctionCall parts once finished.
//
// OpenAI streams tool calls incrementally: the first chunk carries the id,
// type, and function name with empty arguments; subsequent chunks carry
// argument fragments (with no name). Gemini expects complete function calls
// in a single chunk, so we must buffer until finish_reason signals completion.
type ToolCallAccumulator struct {
	// pending maps tool-call index → accumulated state.
	pending map[int]*pendingToolCall
}

type pendingToolCall struct {
	Name string
	Args []byte // accumulated argument fragments
}

// Add merges a streaming tool-call delta into the accumulator.
func (a *ToolCallAccumulator) Add(tc OpenAIToolCall) {
	if a.pending == nil {
		a.pending = make(map[int]*pendingToolCall)
	}
	idx := tc.Index
	p, ok := a.pending[idx]
	if !ok {
		p = &pendingToolCall{}
		a.pending[idx] = p
	}
	if tc.Function.Name != "" {
		p.Name = tc.Function.Name
	}
	if tc.Function.Arguments != "" {
		p.Args = append(p.Args, tc.Function.Arguments...)
	}
}

// Flush returns complete Gemini FunctionCall parts for all buffered tool
// calls and resets the accumulator.
func (a *ToolCallAccumulator) Flush() []Part {
	if len(a.pending) == 0 {
		return nil
	}
	parts := make([]Part, 0, len(a.pending))
	for _, p := range a.pending {
		if p.Name == "" {
			continue
		}
		part := Part{
			FunctionCall: &FunctionCall{
				Name: p.Name,
			},
		}
		if len(p.Args) > 0 {
			part.FunctionCall.Args = json.RawMessage(p.Args)
		}
		parts = append(parts, part)
	}
	a.pending = nil
	return parts
}

// HasPending reports whether any tool calls are being accumulated.
func (a *ToolCallAccumulator) HasPending() bool {
	return len(a.pending) > 0
}

// StreamChunkResult holds the parsed result of a single OpenAI SSE chunk,
// split into text content and tool-call deltas so the caller can handle
// buffering of incremental tool calls.
type StreamChunkResult struct {
	// TextResponse is non-nil when the chunk carries text content or a
	// finish reason (without tool calls). Ready to send immediately.
	TextResponse *ChatResponse
	// ToolCallDeltas are incremental tool-call fragments to feed into a
	// ToolCallAccumulator.
	ToolCallDeltas []OpenAIToolCall
	// FinishReason is set when the chunk carries a finish reason, signalling
	// that any buffered tool calls should be flushed.
	FinishReason string
	// Usage, if present.
	Usage *OpenAIUsage
}

// ParseStreamChunk parses a single OpenAI SSE chunk and separates text
// content from tool-call deltas. The caller is responsible for buffering
// tool-call deltas via ToolCallAccumulator and flushing on FinishReason.
func ParseStreamChunk(raw []byte) (*StreamChunkResult, error) {
	var chunk OpenAIStreamChunk
	if err := json.Unmarshal(raw, &chunk); err != nil {
		return nil, fmt.Errorf("parse openai chunk: %w", err)
	}
	if len(chunk.Choices) == 0 {
		return nil, nil
	}

	result := &StreamChunkResult{Usage: chunk.Usage}
	hasText := false

	out := &ChatResponse{Candidates: make([]Candidate, 0, len(chunk.Choices))}

	for _, ch := range chunk.Choices {
		if ch.FinishReason != "" {
			result.FinishReason = ch.FinishReason
		}
		// Collect tool-call deltas separately.
		for _, tc := range ch.Delta.ToolCalls {
			result.ToolCallDeltas = append(result.ToolCallDeltas, tc)
		}
		// Build text-only candidate.
		content := Content{Role: "model"}
		if ch.Delta.Content != "" {
			content.Parts = append(content.Parts, Part{Text: ch.Delta.Content})
			hasText = true
		}
		candidate := Candidate{
			Index:   ch.Index,
			Content: content,
		}
		// Attach finish reason to text/empty candidates. For tool-call
		// streams the finish reason is applied when flushing the accumulator.
		if len(ch.Delta.ToolCalls) == 0 {
			candidate.FinishReason = mapFinishReasonToGemini(ch.FinishReason)
		}
		out.Candidates = append(out.Candidates, candidate)
	}

	// Emit TextResponse when there's text OR a finish reason (without
	// pending tool calls). The finish-reason-only chunk signals stream
	// completion — dropping it causes the client to hang.
	hasFinish := result.FinishReason != "" && len(result.ToolCallDeltas) == 0
	if hasText || hasFinish {
		if chunk.Usage != nil {
			out.UsageMetadata = &UsageMetadata{
				PromptTokenCount:     chunk.Usage.PromptTokens,
				CandidatesTokenCount: chunk.Usage.CompletionTokens,
				TotalTokenCount:      chunk.Usage.TotalTokens,
			}
		}
		result.TextResponse = out
	}

	return result, nil
}

// BuildToolCallResponse constructs a Gemini ChatResponse containing the
// flushed FunctionCall parts. Called when finish_reason signals completion.
func BuildToolCallResponse(parts []Part, finishReason string, usage *OpenAIUsage) *ChatResponse {
	if len(parts) == 0 {
		return nil
	}
	out := &ChatResponse{
		Candidates: []Candidate{{
			Content: Content{
				Role:  "model",
				Parts: parts,
			},
			FinishReason: mapFinishReasonToGemini(finishReason),
		}},
	}
	if usage != nil {
		out.UsageMetadata = &UsageMetadata{
			PromptTokenCount:     usage.PromptTokens,
			CandidatesTokenCount: usage.CompletionTokens,
			TotalTokenCount:      usage.TotalTokens,
		}
	}
	return out
}

// EmbedContentToOpenAI converts a single Gemini embed request. Forwards the
// Gemini `outputDimensionality` as OpenAI's `dimensions`, which Ollama's
// OpenAI-compat endpoint respects for Matryoshka-capable embedding models.
func EmbedContentToOpenAI(in *EmbedContentRequest, model string) (*OpenAIEmbedRequest, error) {
	text := joinParts(in.Content.Parts)
	if text == "" {
		return nil, errors.New("gemini embed: content has no text")
	}
	return &OpenAIEmbedRequest{
		Model:      model,
		Input:      []string{text},
		Dimensions: in.OutputDimensionality,
	}, nil
}

// EmbedContentResponseFromOpenAI extracts the first embedding back into
// Gemini's single-embed shape.
func EmbedContentResponseFromOpenAI(resp *OpenAIEmbedResponse) (*EmbedContentResponse, error) {
	if len(resp.Data) == 0 {
		return nil, errors.New("openai embed: no data")
	}
	return &EmbedContentResponse{Embedding: Embedding{Values: resp.Data[0].Embedding}}, nil
}

// BatchEmbedRequestToOpenAI converts a batch embed request. All sub-requests
// are grouped into a single OpenAI call with an Input array. If any
// sub-request sets `outputDimensionality`, it's propagated as the
// OpenAI `dimensions`; when sub-requests disagree we take the first value
// since OpenAI's endpoint is single-valued.
func BatchEmbedRequestToOpenAI(in *BatchEmbedContentsRequest, model string) (*OpenAIEmbedRequest, error) {
	inputs := make([]string, 0, len(in.Requests))
	var dims *int
	for i, r := range in.Requests {
		text := joinParts(r.Content.Parts)
		if text == "" {
			return nil, fmt.Errorf("batch embed: request #%d has no text", i)
		}
		inputs = append(inputs, text)
		if dims == nil && r.OutputDimensionality != nil {
			dims = r.OutputDimensionality
		}
	}
	if len(inputs) == 0 {
		return nil, errors.New("batch embed: empty request list")
	}
	return &OpenAIEmbedRequest{Model: model, Input: inputs, Dimensions: dims}, nil
}

// BatchEmbedResponseFromOpenAI converts an OpenAI embed response back into
// Gemini's batch shape. Preserves order via the Index field.
func BatchEmbedResponseFromOpenAI(resp *OpenAIEmbedResponse) *BatchEmbedContentsResponse {
	embeddings := make([]Embedding, len(resp.Data))
	// OpenAI returns them in request order, but sort by Index just in case.
	for _, d := range resp.Data {
		if d.Index < 0 || d.Index >= len(embeddings) {
			continue
		}
		embeddings[d.Index] = Embedding{Values: d.Embedding}
	}
	return &BatchEmbedContentsResponse{Embeddings: embeddings}
}

// toolCallsToParts converts OpenAI tool_calls into Gemini FunctionCall parts.
func toolCallsToParts(calls []OpenAIToolCall) []Part {
	if len(calls) == 0 {
		return nil
	}
	parts := make([]Part, 0, len(calls))
	for _, tc := range calls {
		if tc.Function.Name == "" {
			continue
		}
		parts = append(parts, Part{
			FunctionCall: &FunctionCall{
				Name: tc.Function.Name,
				Args: json.RawMessage(tc.Function.Arguments),
			},
		})
	}
	return parts
}

func joinParts(parts []Part) string {
	if len(parts) == 0 {
		return ""
	}
	if len(parts) == 1 {
		return parts[0].Text
	}
	var b strings.Builder
	for i, p := range parts {
		if i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(p.Text)
	}
	return b.String()
}

// mapRoleToOpenAI normalizes Gemini roles. Gemini uses "user" and "model";
// OpenAI uses "user" and "assistant". "system" is Gemini's own convention in
// some SDKs even though the native field is systemInstruction.
func mapRoleToOpenAI(role string) string {
	switch role {
	case "model":
		return "assistant"
	case "", "user":
		return "user"
	case "system":
		return "system"
	default:
		return role
	}
}

// mapFinishReasonToGemini translates OpenAI finish reasons to Gemini names.
// Unknown values pass through — clients rarely branch on unrecognized
// finish reasons so preserving them is safer than dropping them.
func mapFinishReasonToGemini(r string) string {
	switch strings.ToLower(r) {
	case "stop":
		return "STOP"
	case "tool_calls":
		return "STOP"
	case "length":
		return "MAX_TOKENS"
	case "content_filter":
		return "SAFETY"
	case "":
		return ""
	default:
		return strings.ToUpper(r)
	}
}
