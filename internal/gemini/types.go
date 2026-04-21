// Package gemini implements the Google Gemini native wire format.
//
// We only model the fields actually used by AFFiNE today: text-only chat
// (generateContent / streamGenerateContent) and text embeddings
// (embedContent / batchEmbedContents). Multimodal parts (images, audio),
// function calling, and safety settings are deliberately out of scope — add
// them when a real client needs them.
package gemini

import "encoding/json"

// ChatRequest is the body of POST /v1beta/models/{model}:generateContent.
type ChatRequest struct {
	Contents          []Content          `json:"contents,omitempty"`
	SystemInstruction *Content           `json:"systemInstruction,omitempty"`
	GenerationConfig  *GenerationConfig  `json:"generationConfig,omitempty"`
	Tools             []Tool             `json:"tools,omitempty"`
}

// Tool wraps a list of function declarations (Gemini format).
type Tool struct {
	FunctionDeclarations []FunctionDeclaration `json:"functionDeclarations,omitempty"`
}

// FunctionDeclaration describes a callable function.
type FunctionDeclaration struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// Content is a single turn in the conversation.
type Content struct {
	// Role is "user" or "model". System prompts use SystemInstruction.
	Role  string `json:"role,omitempty"`
	Parts []Part `json:"parts"`
}

// Part is one chunk of a message.
type Part struct {
	Text         string        `json:"text,omitempty"`
	FunctionCall *FunctionCall `json:"functionCall,omitempty"`
}

// FunctionCall is emitted by the model when it wants to invoke a tool.
type FunctionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

// GenerationConfig is the subset of sampling parameters we translate.
type GenerationConfig struct {
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"topP,omitempty"`
	TopK            *int     `json:"topK,omitempty"`
	MaxOutputTokens *int     `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

// ChatResponse is the non-streaming body of generateContent.
type ChatResponse struct {
	Candidates    []Candidate    `json:"candidates"`
	UsageMetadata *UsageMetadata `json:"usageMetadata,omitempty"`
}

// Candidate is one possible response.
type Candidate struct {
	Content      Content `json:"content"`
	FinishReason string  `json:"finishReason,omitempty"`
	Index        int     `json:"index"`
}

// UsageMetadata is the Gemini equivalent of OpenAI's usage object.
type UsageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount,omitempty"`
	CandidatesTokenCount int `json:"candidatesTokenCount,omitempty"`
	TotalTokenCount      int `json:"totalTokenCount,omitempty"`
}

// EmbedContentRequest is POST /v1beta/models/{model}:embedContent.
//
// OutputDimensionality and TaskType are sent by the Vercel `@ai-sdk/google`
// library on every embed call (via providerOptions). AFFiNE for instance
// sets OutputDimensionality=1024 so the vector matches its pgvector column
// width — dropping it on the floor returns the model's native dimension
// (4096 for qwen3-embedding:8b) and the caller's INSERT fails.
type EmbedContentRequest struct {
	Content              Content `json:"content"`
	OutputDimensionality *int    `json:"outputDimensionality,omitempty"`
	TaskType             string  `json:"taskType,omitempty"`
	Title                string  `json:"title,omitempty"`
}

// EmbedContentResponse is its response body.
type EmbedContentResponse struct {
	Embedding Embedding `json:"embedding"`
}

// Embedding carries the vector.
type Embedding struct {
	Values []float64 `json:"values"`
}

// BatchEmbedContentsRequest is POST /v1beta/models/{model}:batchEmbedContents.
type BatchEmbedContentsRequest struct {
	Requests []BatchEmbedRequest `json:"requests"`
}

// BatchEmbedRequest wraps a single content to embed in a batch.
type BatchEmbedRequest struct {
	// Model may be set but we ignore it — routing happens at the URL level.
	Model                string  `json:"model,omitempty"`
	Content              Content `json:"content"`
	OutputDimensionality *int    `json:"outputDimensionality,omitempty"`
	TaskType             string  `json:"taskType,omitempty"`
	Title                string  `json:"title,omitempty"`
}

// BatchEmbedContentsResponse carries all resulting vectors.
type BatchEmbedContentsResponse struct {
	Embeddings []Embedding `json:"embeddings"`
}
