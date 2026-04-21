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
}

// OpenAIMessage is a single message.
type OpenAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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
	Index        int                    `json:"index"`
	Message      OpenAIMessage          `json:"message"`
	FinishReason string                 `json:"finish_reason,omitempty"`
	Delta        map[string]any         `json:"delta,omitempty"`
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
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// ChatRequestToOpenAI converts a Gemini chat request into the OpenAI wire
// format. Text-only — any non-text parts are dropped with an error.
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
	for _, c := range in.Contents {
		text := joinParts(c.Parts)
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
	return out, nil
}

// ChatResponseFromOpenAI converts a non-streaming OpenAI response back to
// Gemini's shape.
func ChatResponseFromOpenAI(resp *OpenAIChatResponse) *ChatResponse {
	out := &ChatResponse{Candidates: make([]Candidate, 0, len(resp.Choices))}
	for _, ch := range resp.Choices {
		out.Candidates = append(out.Candidates, Candidate{
			Index: ch.Index,
			Content: Content{
				Role:  "model",
				Parts: []Part{{Text: ch.Message.Content}},
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

// StreamChunkFromOpenAI converts a single OpenAI SSE chunk to the Gemini
// streaming JSON shape that Gemini emits (one JSON object per line, no
// `data:` prefix). Returns nil if the chunk carries no deltable content.
func StreamChunkFromOpenAI(raw []byte) (*ChatResponse, error) {
	var chunk OpenAIStreamChunk
	if err := json.Unmarshal(raw, &chunk); err != nil {
		return nil, fmt.Errorf("parse openai chunk: %w", err)
	}
	if len(chunk.Choices) == 0 {
		return nil, nil
	}
	out := &ChatResponse{Candidates: make([]Candidate, 0, len(chunk.Choices))}
	hasContent := false
	for _, ch := range chunk.Choices {
		content := Content{Role: "model"}
		if ch.Delta.Content != "" {
			content.Parts = []Part{{Text: ch.Delta.Content}}
			hasContent = true
		}
		out.Candidates = append(out.Candidates, Candidate{
			Index:        ch.Index,
			Content:      content,
			FinishReason: mapFinishReasonToGemini(ch.FinishReason),
		})
	}
	if !hasContent && allFinishReasonsEmpty(out.Candidates) {
		return nil, nil
	}
	if chunk.Usage != nil {
		out.UsageMetadata = &UsageMetadata{
			PromptTokenCount:     chunk.Usage.PromptTokens,
			CandidatesTokenCount: chunk.Usage.CompletionTokens,
			TotalTokenCount:      chunk.Usage.TotalTokens,
		}
	}
	return out, nil
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

func allFinishReasonsEmpty(c []Candidate) bool {
	for _, x := range c {
		if x.FinishReason != "" {
			return false
		}
	}
	return true
}
