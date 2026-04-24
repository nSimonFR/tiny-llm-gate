package server

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"
)

// anthropicUsage holds token counts extracted from Anthropic SSE events.
type anthropicUsage struct {
	Model                string
	InputTokens          int
	OutputTokens         int
	CacheReadInputTokens int
	CacheCreateTokens    int
}

// handleAnthropicMessages proxies POST /v1/messages to the Anthropic API
// with full pass-through of client auth and headers. After the response
// completes, it fires an async shadow request to Aperture for observability.
func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	started := time.Now()
	reqID := requestID(r.Context())

	body, err := readBoundedBody(r)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		s.logger.Warn("anthropic: read body", "request_id", reqID, "err", err)
		return
	}

	var peek struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	if err := json.Unmarshal(body, &peek); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if peek.Model == "" {
		writeJSONError(w, http.StatusBadRequest, "missing 'model' field")
		return
	}

	// Build upstream URL preserving query string (?beta=true etc.)
	upstream := strings.TrimRight(s.cfg.Anthropic.Upstream, "/") + r.URL.Path
	if r.URL.RawQuery != "" {
		upstream += "?" + r.URL.RawQuery
	}

	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, upstream, bytes.NewReader(body))
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, "build upstream request: "+err.Error())
		return
	}

	// Pass through all client headers (the whole point of this handler).
	copyHeaders(req.Header, r.Header)
	// Override Accept-Encoding to avoid gzip complications — we need to
	// parse SSE events from the response for usage extraction.
	req.Header.Set("Accept-Encoding", "identity")

	resp, err := s.client.Do(req)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return // client disconnected
		}
		writeJSONError(w, http.StatusBadGateway, "upstream transport: "+err.Error())
		s.logger.Error("anthropic: upstream", "request_id", reqID, "err", err)
		return
	}
	defer resp.Body.Close()

	// Copy response headers to client.
	copyHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)

	// Only extract usage and fire shadow on success.
	if resp.StatusCode != http.StatusOK {
		_, _ = io.Copy(w, resp.Body)
		s.logger.Info("anthropic: upstream error",
			"request_id", reqID, "status", resp.StatusCode)
		return
	}

	var usage anthropicUsage
	usage.Model = peek.Model

	if peek.Stream {
		usage = anthropicStreamTee(w, resp.Body, peek.Model)
	} else {
		usage = anthropicCopyExtractUsage(w, resp.Body, peek.Model)
	}

	s.logger.Info("served",
		"request_id", reqID,
		"frontend", "anthropic",
		"model", peek.Model,
		"stream", peek.Stream,
		"input_tokens", usage.InputTokens,
		"output_tokens", usage.OutputTokens,
		"cache_read", usage.CacheReadInputTokens,
		"latency_ms", time.Since(started).Milliseconds(),
	)

	// Fire shadow request async — never blocks the real response.
	if s.cfg.Anthropic.ShadowURL != "" {
		go s.fireShadow(usage)
	}
}

// anthropicStreamTee streams the upstream response to the client while
// extracting usage data from Anthropic SSE events. Only parses message_start
// and message_delta events; all other data lines are forwarded without parsing.
func anthropicStreamTee(w http.ResponseWriter, body io.Reader, model string) anthropicUsage {
	flusher, _ := w.(http.Flusher)
	usage := anthropicUsage{Model: model}
	reader := bufio.NewReaderSize(body, 4096)

	var currentEvent string

	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			// Forward every byte to the client immediately.
			if _, werr := w.Write(line); werr != nil {
				return usage
			}
			if flusher != nil {
				flusher.Flush()
			}

			trimmed := bytes.TrimRight(line, "\r\n")
			if len(trimmed) == 0 {
				currentEvent = ""
				continue
			}

			// Track SSE event type.
			if bytes.HasPrefix(trimmed, []byte("event:")) {
				currentEvent = string(bytes.TrimSpace(trimmed[len("event:"):]))
				continue
			}

			// Only parse data lines for usage-bearing events.
			if bytes.HasPrefix(trimmed, []byte("data:")) {
				payload := bytes.TrimSpace(trimmed[len("data:"):])
				switch currentEvent {
				case "message_start":
					extractMessageStartUsage(payload, &usage)
				case "message_delta":
					extractMessageDeltaUsage(payload, &usage)
				}
			}
		}
		if err != nil {
			return usage
		}
	}
}

// extractMessageStartUsage extracts input_tokens and cache_read_input_tokens
// from a message_start event's data payload.
//
//	{"type":"message_start","message":{"usage":{"input_tokens":3,"cache_read_input_tokens":68976,...}}}
func extractMessageStartUsage(data []byte, usage *anthropicUsage) {
	var envelope struct {
		Message struct {
			Usage struct {
				InputTokens              int `json:"input_tokens"`
				CacheReadInputTokens     int `json:"cache_read_input_tokens"`
				CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			} `json:"usage"`
		} `json:"message"`
	}
	if json.Unmarshal(data, &envelope) == nil {
		usage.InputTokens = envelope.Message.Usage.InputTokens
		usage.CacheReadInputTokens = envelope.Message.Usage.CacheReadInputTokens
		usage.CacheCreateTokens = envelope.Message.Usage.CacheCreationInputTokens
	}
}

// extractMessageDeltaUsage extracts output_tokens from a message_delta event.
//
//	{"type":"message_delta","usage":{"output_tokens":2}}
func extractMessageDeltaUsage(data []byte, usage *anthropicUsage) {
	var envelope struct {
		Usage struct {
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if json.Unmarshal(data, &envelope) == nil {
		usage.OutputTokens = envelope.Usage.OutputTokens
	}
}

// anthropicCopyExtractUsage handles non-streaming responses: copies the body
// to the client and extracts usage from the top-level JSON.
func anthropicCopyExtractUsage(w http.ResponseWriter, body io.Reader, model string) anthropicUsage {
	usage := anthropicUsage{Model: model}

	data, err := io.ReadAll(io.LimitReader(body, maxRequestBytes))
	if err != nil {
		return usage
	}
	_, _ = w.Write(data)

	var envelope struct {
		Usage struct {
			InputTokens              int `json:"input_tokens"`
			OutputTokens             int `json:"output_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
		} `json:"usage"`
	}
	if json.Unmarshal(data, &envelope) == nil {
		usage.InputTokens = envelope.Usage.InputTokens
		usage.OutputTokens = envelope.Usage.OutputTokens
		usage.CacheReadInputTokens = envelope.Usage.CacheReadInputTokens
		usage.CacheCreateTokens = envelope.Usage.CacheCreationInputTokens
	}
	return usage
}

// fireShadow sends a fire-and-forget shadow request to the configured
// ShadowURL (typically Aperture) for observability logging. The request
// uses OpenAI /v1/chat/completions format with a cc/ prefixed model name.
// Real token usage from the Anthropic response is embedded in the request
// body via x_usage so the echo handler can return it in the response —
// this is what Aperture reads for cost tracking.
func (s *Server) fireShadow(usage anthropicUsage) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	shadowModel := shadowModelPrefix + usage.Model
	totalInput := usage.InputTokens + usage.CacheReadInputTokens + usage.CacheCreateTokens
	payload, _ := json.Marshal(map[string]any{
		"model":      shadowModel,
		"messages":   []map[string]string{{"role": "user", "content": "shadow"}},
		"max_tokens": 1,
		// Embed real usage for the echo handler to return in the response.
		"x_usage": map[string]int{
			"prompt_tokens":     totalInput,
			"completion_tokens": usage.OutputTokens,
			"total_tokens":      totalInput + usage.OutputTokens,
		},
	})

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.Anthropic.ShadowURL, bytes.NewReader(payload))
	if err != nil {
		s.logger.Warn("shadow: build request", "err", err)
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		s.logger.Warn("shadow: transport", "err", err)
		return
	}
	drain(resp.Body)
	resp.Body.Close()

	s.logger.Info("shadow",
		"model", shadowModel,
		"status", resp.StatusCode,
		"input_tokens", usage.InputTokens,
		"output_tokens", usage.OutputTokens,
		"cache_read", usage.CacheReadInputTokens,
		"cache_create", usage.CacheCreateTokens,
	)
}
