package gemini

import (
	"encoding/json"
	"testing"
)

func TestChatRequestToOpenAIMessagesAndSystem(t *testing.T) {
	temp := 0.7
	in := &ChatRequest{
		SystemInstruction: &Content{Parts: []Part{{Text: "You are helpful."}}},
		Contents: []Content{
			{Role: "user", Parts: []Part{{Text: "Hello"}}},
			{Role: "model", Parts: []Part{{Text: "Hi!"}}},
			{Role: "user", Parts: []Part{{Text: "Count"}}},
		},
		GenerationConfig: &GenerationConfig{Temperature: &temp},
	}
	out, err := ChatRequestToOpenAI(in, "target-model", true)
	if err != nil {
		t.Fatal(err)
	}
	if out.Model != "target-model" || !out.Stream {
		t.Errorf("meta wrong: model=%s stream=%v", out.Model, out.Stream)
	}
	if len(out.Messages) != 4 {
		t.Fatalf("messages=%d want 4", len(out.Messages))
	}
	if out.Messages[0].Role != "system" || out.Messages[0].Content != "You are helpful." {
		t.Errorf("system msg wrong: %+v", out.Messages[0])
	}
	if out.Messages[1].Role != "user" || out.Messages[2].Role != "assistant" || out.Messages[3].Role != "user" {
		t.Errorf("role mapping wrong: %+v", out.Messages)
	}
	if out.Temperature == nil || *out.Temperature != 0.7 {
		t.Errorf("temperature not propagated")
	}
}

func TestChatRequestToOpenAIMultipartText(t *testing.T) {
	in := &ChatRequest{
		Contents: []Content{
			{Role: "user", Parts: []Part{{Text: "line 1"}, {Text: "line 2"}}},
		},
	}
	out, err := ChatRequestToOpenAI(in, "m", false)
	if err != nil {
		t.Fatal(err)
	}
	if out.Messages[0].Content != "line 1\nline 2" {
		t.Errorf("multipart join wrong: %q", out.Messages[0].Content)
	}
}

func TestChatRequestNoContentErrors(t *testing.T) {
	in := &ChatRequest{Contents: []Content{{Role: "user", Parts: []Part{{Text: ""}}}}}
	if _, err := ChatRequestToOpenAI(in, "m", false); err == nil {
		t.Error("expected error for empty content")
	}
}

func TestChatResponseFromOpenAIRoundTrip(t *testing.T) {
	src := &OpenAIChatResponse{
		ID:      "chat-1",
		Model:   "gpt-test",
		Choices: []OpenAIChoice{{Index: 0, Message: OpenAIMessage{Role: "assistant", Content: "Hello there."}, FinishReason: "stop"}},
		Usage:   &OpenAIUsage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
	}
	out := ChatResponseFromOpenAI(src)
	if len(out.Candidates) != 1 {
		t.Fatalf("candidates=%d", len(out.Candidates))
	}
	c := out.Candidates[0]
	if c.Content.Role != "model" || len(c.Content.Parts) != 1 || c.Content.Parts[0].Text != "Hello there." {
		t.Errorf("candidate content wrong: %+v", c)
	}
	if c.FinishReason != "STOP" {
		t.Errorf("finish reason not uppercased: %q", c.FinishReason)
	}
	if out.UsageMetadata == nil || out.UsageMetadata.TotalTokenCount != 8 {
		t.Errorf("usage metadata wrong: %+v", out.UsageMetadata)
	}
}

func TestStreamChunkFromOpenAIEmitsGeminiShape(t *testing.T) {
	in := `{"id":"abc","model":"m","choices":[{"index":0,"delta":{"content":"foo"}}]}`
	out, err := StreamChunkFromOpenAI([]byte(in))
	if err != nil {
		t.Fatal(err)
	}
	if out == nil || len(out.Candidates) != 1 {
		t.Fatalf("unexpected: %+v", out)
	}
	if out.Candidates[0].Content.Parts[0].Text != "foo" {
		t.Errorf("text lost: %+v", out.Candidates[0])
	}
}

func TestStreamChunkEmptyDeltaReturnsNil(t *testing.T) {
	in := `{"choices":[{"index":0,"delta":{"role":"assistant"}}]}`
	out, err := StreamChunkFromOpenAI([]byte(in))
	if err != nil {
		t.Fatal(err)
	}
	if out != nil {
		t.Errorf("expected nil for role-only delta, got %+v", out)
	}
}

func TestStreamChunkFinishOnlyPreserved(t *testing.T) {
	in := `{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`
	out, err := StreamChunkFromOpenAI([]byte(in))
	if err != nil {
		t.Fatal(err)
	}
	if out == nil {
		t.Fatal("expected non-nil for finish")
	}
	if out.Candidates[0].FinishReason != "STOP" {
		t.Errorf("finish not mapped: %+v", out.Candidates[0])
	}
}

func TestEmbedContentForwardsOutputDimensionality(t *testing.T) {
	dims := 1024
	in := &EmbedContentRequest{
		Content:              Content{Parts: []Part{{Text: "hello"}}},
		OutputDimensionality: &dims,
		TaskType:             "RETRIEVAL_DOCUMENT",
	}
	out, err := EmbedContentToOpenAI(in, "qwen3-embedding:8b")
	if err != nil {
		t.Fatal(err)
	}
	if out.Dimensions == nil || *out.Dimensions != 1024 {
		t.Errorf("dimensions not forwarded: %+v", out.Dimensions)
	}
}

func TestBatchEmbedForwardsFirstNonNilDimensionality(t *testing.T) {
	d := 768
	in := &BatchEmbedContentsRequest{
		Requests: []BatchEmbedRequest{
			{Content: Content{Parts: []Part{{Text: "a"}}}},
			{Content: Content{Parts: []Part{{Text: "b"}}}, OutputDimensionality: &d},
		},
	}
	out, err := BatchEmbedRequestToOpenAI(in, "m")
	if err != nil {
		t.Fatal(err)
	}
	if out.Dimensions == nil || *out.Dimensions != 768 {
		t.Errorf("expected dims=768, got %+v", out.Dimensions)
	}
}

func TestEmbedContentRoundTrip(t *testing.T) {
	in := &EmbedContentRequest{Content: Content{Parts: []Part{{Text: "hello"}}}}
	toOA, err := EmbedContentToOpenAI(in, "emb-model")
	if err != nil {
		t.Fatal(err)
	}
	if toOA.Model != "emb-model" || len(toOA.Input) != 1 || toOA.Input[0] != "hello" {
		t.Errorf("embed request wrong: %+v", toOA)
	}
	oaResp := &OpenAIEmbedResponse{Data: []OpenAIEmbedding{{Index: 0, Embedding: []float64{0.1, 0.2}}}}
	back, err := EmbedContentResponseFromOpenAI(oaResp)
	if err != nil {
		t.Fatal(err)
	}
	if len(back.Embedding.Values) != 2 || back.Embedding.Values[0] != 0.1 {
		t.Errorf("embedding response wrong: %+v", back)
	}
}

func TestBatchEmbedRoundTrip(t *testing.T) {
	in := &BatchEmbedContentsRequest{
		Requests: []BatchEmbedRequest{
			{Content: Content{Parts: []Part{{Text: "a"}}}},
			{Content: Content{Parts: []Part{{Text: "b"}}}},
			{Content: Content{Parts: []Part{{Text: "c"}}}},
		},
	}
	toOA, err := BatchEmbedRequestToOpenAI(in, "emb")
	if err != nil {
		t.Fatal(err)
	}
	if len(toOA.Input) != 3 || toOA.Input[1] != "b" {
		t.Errorf("batch inputs wrong: %v", toOA.Input)
	}

	oaResp := &OpenAIEmbedResponse{Data: []OpenAIEmbedding{
		{Index: 1, Embedding: []float64{1.0}},
		{Index: 0, Embedding: []float64{0.0}},
		{Index: 2, Embedding: []float64{2.0}},
	}}
	back := BatchEmbedResponseFromOpenAI(oaResp)
	if len(back.Embeddings) != 3 {
		t.Fatalf("expected 3 embeddings, got %d", len(back.Embeddings))
	}
	if back.Embeddings[0].Values[0] != 0.0 || back.Embeddings[1].Values[0] != 1.0 || back.Embeddings[2].Values[0] != 2.0 {
		t.Errorf("embeddings out of order: %+v", back.Embeddings)
	}
}

// Sanity: the produced OpenAI request JSON-serializes to the shape LiteLLM
// consumes. Mostly guards against silent struct-tag drift.
func TestOpenAIRequestJSONShape(t *testing.T) {
	req := &OpenAIChatRequest{
		Model:    "m",
		Stream:   true,
		Messages: []OpenAIMessage{{Role: "user", Content: "hi"}},
	}
	b, _ := json.Marshal(req)
	got := string(b)
	want := `{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true}`
	if got != want {
		t.Errorf("json shape drift:\n got:  %s\n want: %s", got, want)
	}
}
