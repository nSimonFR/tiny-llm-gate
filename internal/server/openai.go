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

	"github.com/nSimonFR/tiny-llm-gate/internal/resolve"
)

// Maximum request body size. Generous for batch embeddings but small enough
// to prevent a single request from blowing the memory budget.
const maxRequestBytes = 8 * 1024 * 1024 // 8 MiB

// chatCompletionsPath is appended to the provider's base_url.
const (
	chatPath  = "/chat/completions"
	embedPath = "/embeddings"
)

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	s.proxyOpenAI(w, r, chatPath)
}

func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	s.proxyOpenAI(w, r, embedPath)
}

// proxyOpenAI is the shared body of chat/completions and embeddings routes.
// It:
//   1. Reads the body (bounded by maxRequestBytes)
//   2. Peeks the "model" and "stream" fields
//   3. Resolves the model through aliases
//   4. Iterates the model + its fallbacks, sending to each upstream until one
//      succeeds. Fallbacks only fire when no bytes have been written to the
//      client yet.
func (s *Server) proxyOpenAI(w http.ResponseWriter, r *http.Request, upstreamPath string) {
	started := time.Now()
	reqID := requestID(r.Context())

	body, err := readBoundedBody(r)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		s.logger.Warn("read body", "request_id", reqID, "err", err)
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

	res, err := s.resolver.Resolve(peek.Model)
	if err != nil {
		writeJSONError(w, http.StatusNotFound, err.Error())
		s.logger.Info("unknown model",
			"request_id", reqID, "model", peek.Model, "err", err)
		return
	}

	// Build the chain of model names to try: primary then fallbacks.
	chain := make([]string, 0, len(res.Fallback)+1)
	chain = append(chain, res.ModelName)
	chain = append(chain, res.Fallback...)

	var lastErr error
	for i, name := range chain {
		hop, err := s.resolver.Resolve(name)
		if err != nil {
			lastErr = err
			continue
		}
		newBody, err := rewriteModelField(body, hop.UpstreamModel)
		if err != nil {
			lastErr = err
			continue
		}
		done, err := s.sendUpstream(w, r, hop, upstreamPath, newBody, peek.Stream, i < len(chain)-1)
		if done {
			s.logger.Info("served",
				"request_id", reqID,
				"client_model", peek.Model,
				"resolved_model", hop.ModelName,
				"provider", hop.ProviderName,
				"stream", peek.Stream,
				"fallback_index", i,
				"latency_ms", time.Since(started).Milliseconds(),
			)
			return
		}
		lastErr = err
	}

	writeJSONError(w, http.StatusBadGateway,
		fmt.Sprintf("all upstreams failed: %v", lastErr))
	s.logger.Error("all upstreams failed",
		"request_id", reqID, "client_model", peek.Model, "err", lastErr)
}

// sendUpstream executes one attempt. Returns (done=true) iff bytes have
// already been committed to the client — in which case the caller must NOT
// try another fallback. When done=false, err explains the failure and the
// caller can continue to the next fallback.
//
// canRetry indicates whether the caller would attempt another fallback on
// failure. When canRetry is true we withhold writing ANY response header on
// upstream-level errors, so the caller has a chance to try again.
func (s *Server) sendUpstream(
	w http.ResponseWriter,
	r *http.Request,
	hop *resolve.Resolved,
	upstreamPath string,
	body []byte,
	isStream bool,
	canRetry bool,
) (done bool, err error) {
	url := strings.TrimRight(hop.Provider.BaseURL, "/") + upstreamPath

	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return false, fmt.Errorf("build upstream request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if hop.Provider.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+hop.Provider.APIKey)
	}

	resp, err := s.client.Do(req)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			// client disconnected; nothing else to do.
			return true, nil
		}
		return false, fmt.Errorf("upstream transport: %w", err)
	}
	defer resp.Body.Close()

	// Retryable upstream error: consume body to free the connection and
	// bubble up so the caller tries the next fallback.
	if resp.StatusCode >= 500 && canRetry {
		drain(resp.Body)
		return false, fmt.Errorf("upstream status %d", resp.StatusCode)
	}

	// Commit response to client.
	copyHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)

	if isStream && resp.StatusCode == http.StatusOK {
		streamCopy(w, resp.Body)
	} else {
		_, _ = io.Copy(w, resp.Body)
	}
	return true, nil
}

// streamCopy pipes upstream SSE chunks to the client with Flush after each
// read, so the client receives tokens as they arrive instead of waiting for
// buffer fills.
func streamCopy(w http.ResponseWriter, r io.Reader) {
	flusher, _ := w.(http.Flusher)
	buf := make([]byte, 4096)
	for {
		n, err := r.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			return
		}
	}
}

// copyHeaders copies src into dst, skipping hop-by-hop headers.
func copyHeaders(dst, src http.Header) {
	for k, vs := range src {
		if isHopByHop(k) {
			continue
		}
		for _, v := range vs {
			dst.Add(k, v)
		}
	}
}

func isHopByHop(name string) bool {
	switch http.CanonicalHeaderKey(name) {
	case "Connection", "Keep-Alive", "Proxy-Authenticate", "Proxy-Authorization",
		"Te", "Trailer", "Transfer-Encoding", "Upgrade":
		return true
	}
	return false
}

func drain(r io.Reader) {
	_, _ = io.Copy(io.Discard, r)
}

func readBoundedBody(r *http.Request) ([]byte, error) {
	r.Body = http.MaxBytesReader(nil, r.Body, maxRequestBytes)
	body, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, fmt.Errorf("read body: %w", err)
	}
	return body, nil
}

// rewriteModelField produces a new JSON body with the "model" field replaced.
// Uses json.RawMessage decoding to preserve field order and unknown fields.
func rewriteModelField(body []byte, newModel string) ([]byte, error) {
	// Decode into an ordered list of key/value pairs to preserve insertion
	// order. json.Unmarshal into map[string]any is fine for correctness but
	// scrambles key order, which can surprise observability tools sniffing
	// request bodies. We use a lightweight stream decoder to rewrite in
	// place.
	out := &bytes.Buffer{}
	out.Grow(len(body) + len(newModel))

	dec := json.NewDecoder(bytes.NewReader(body))
	enc := json.NewEncoder(out)
	enc.SetEscapeHTML(false)

	// Parse top-level object.
	tok, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("json: %w", err)
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return nil, errors.New("json body must be an object")
	}
	out.WriteByte('{')
	first := true
	replaced := false
	for dec.More() {
		if !first {
			out.WriteByte(',')
		}
		first = false
		kTok, err := dec.Token()
		if err != nil {
			return nil, fmt.Errorf("json key: %w", err)
		}
		key, ok := kTok.(string)
		if !ok {
			return nil, fmt.Errorf("expected string key, got %T", kTok)
		}
		if err := enc.Encode(key); err != nil {
			return nil, err
		}
		// json.Encoder appends a trailing newline; trim and add ':'
		trimTrailingNewline(out)
		out.WriteByte(':')
		if key == "model" {
			// consume and drop the original value
			var drop json.RawMessage
			if err := dec.Decode(&drop); err != nil {
				return nil, fmt.Errorf("json model value: %w", err)
			}
			if err := enc.Encode(newModel); err != nil {
				return nil, err
			}
			trimTrailingNewline(out)
			replaced = true
		} else {
			var raw json.RawMessage
			if err := dec.Decode(&raw); err != nil {
				return nil, fmt.Errorf("json value for %q: %w", key, err)
			}
			out.Write(raw)
		}
	}
	out.WriteByte('}')
	if !replaced {
		return nil, errors.New("no 'model' field to rewrite")
	}
	return out.Bytes(), nil
}

func trimTrailingNewline(b *bytes.Buffer) {
	if b.Len() == 0 {
		return
	}
	if b.Bytes()[b.Len()-1] == '\n' {
		b.Truncate(b.Len() - 1)
	}
}

func writeJSONError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}{
		Error: struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		}{
			Message: message,
			Type:    "tiny_llm_gate_error",
		},
	})
}

// ModelList response for GET /v1/models.
type modelListResponse struct {
	Object string          `json:"object"`
	Data   []modelListItem `json:"data"`
}

type modelListItem struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	models := s.resolver.ListModels()
	resp := modelListResponse{
		Object: "list",
		Data:   make([]modelListItem, 0, len(models)),
	}
	for _, m := range models {
		resp.Data = append(resp.Data, modelListItem{
			ID:      m,
			Object:  "model",
			Created: 0,
			OwnedBy: "tiny-llm-gate",
		})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

