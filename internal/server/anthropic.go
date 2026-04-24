package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// handleAnthropicMessages proxies POST /v1/messages to the configured
// Anthropic upstream. The client's request body is forwarded as-is; non-auth
// headers pass through (so anthropic-version, anthropic-beta, etc. reach
// Anthropic unchanged), but any incoming Authorization header is stripped
// and replaced with the gate's configured auth.
//
// Intended to sit behind an observability layer (e.g. Aperture) so the full
// request and response body are logged there — tiny-llm-gate does not
// inspect or rewrite the payload.
func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	started := time.Now()
	reqID := requestID(r.Context())

	body, err := readBoundedBody(r)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		s.logger.Warn("anthropic: read body", "request_id", reqID, "err", err)
		return
	}

	// Peek model + stream for logging only — don't mutate the body.
	var peek struct {
		Model  string `json:"model"`
		Stream bool   `json:"stream"`
	}
	if err := json.Unmarshal(body, &peek); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}

	// Build upstream URL preserving query string (e.g. ?beta=true).
	upstream := strings.TrimRight(s.cfg.Anthropic.Upstream, "/") + r.URL.Path
	if r.URL.RawQuery != "" {
		upstream += "?" + r.URL.RawQuery
	}

	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, upstream, bytes.NewReader(body))
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, fmt.Sprintf("build upstream request: %v", err))
		return
	}

	// Forward non-auth client headers (anthropic-version, anthropic-beta,
	// content-type, X-Stainless-*, etc.) then overwrite Authorization with
	// our configured credential. Accept-Encoding is forced to identity so
	// we forward plaintext to the client (the upstream might otherwise
	// choose gzip and we'd relay a compressed body).
	copyHeaders(req.Header, r.Header)
	req.Header.Del("Authorization")
	req.Header.Set("Accept-Encoding", "identity")
	if s.anthropicAuth != nil {
		if err := s.anthropicAuth.Apply(r.Context(), req); err != nil {
			writeJSONError(w, http.StatusBadGateway, fmt.Sprintf("apply anthropic auth: %v", err))
			return
		}
	}

	resp, err := s.client.Do(req)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return // client disconnected; no response to write.
		}
		writeJSONError(w, http.StatusBadGateway, fmt.Sprintf("upstream transport: %v", err))
		s.logger.Error("anthropic: upstream", "request_id", reqID, "err", err)
		return
	}
	defer resp.Body.Close()

	copyHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)

	// Stream or copy the body. Streaming (stream=true or SSE content-type)
	// uses the same 4KB-flush loop as the OpenAI handler.
	if peek.Stream || strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		streamCopy(w, resp.Body)
	} else {
		_, _ = io.Copy(w, resp.Body)
	}

	s.logger.Info("served",
		"request_id", reqID,
		"frontend", "anthropic",
		"model", peek.Model,
		"stream", peek.Stream,
		"status", resp.StatusCode,
		"latency_ms", time.Since(started).Milliseconds(),
	)
}
