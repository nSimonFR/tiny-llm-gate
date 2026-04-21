package server

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/gemini"
	"github.com/nSimonFR/tiny-llm-gate/internal/resolve"
)

// handleGenerateContent handles POST /v1beta/models/{model}:generateContent
// (non-streaming) and POST /v1beta/models/{model}:streamGenerateContent.
//
// Flow: parse Gemini request → translate to OpenAI chat → proxy through
// fallback chain → translate response back to Gemini shape (or SSE to
// newline-delimited JSON for the streaming variant).
func (s *Server) handleGenerateContent(w http.ResponseWriter, r *http.Request) {
	s.serveGeminiChat(w, r, false)
}

func (s *Server) handleStreamGenerateContent(w http.ResponseWriter, r *http.Request) {
	s.serveGeminiChat(w, r, true)
}

func (s *Server) serveGeminiChat(w http.ResponseWriter, r *http.Request, stream bool) {
	started := time.Now()
	reqID := requestID(r.Context())

	clientModel, ok := extractGeminiModel(r.URL.Path)
	if !ok {
		writeJSONError(w, http.StatusBadRequest, "could not parse model from URL")
		return
	}

	body, err := readBoundedBody(r)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	var in gemini.ChatRequest
	if err := json.Unmarshal(body, &in); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid Gemini JSON body")
		return
	}

	res, err := s.resolver.Resolve(clientModel)
	if err != nil {
		writeJSONError(w, http.StatusNotFound, err.Error())
		return
	}
	chain := append([]string{res.ModelName}, res.Fallback...)

	var lastErr error
	for i, name := range chain {
		hop, err := s.resolver.Resolve(name)
		if err != nil {
			lastErr = err
			continue
		}
		oaReq, err := gemini.ChatRequestToOpenAI(&in, hop.UpstreamModel, stream)
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, err.Error())
			return
		}
		upstreamBody, err := json.Marshal(oaReq)
		if err != nil {
			lastErr = err
			continue
		}

		resp, retryable, err := s.sendGeminiChatRequest(r, hop, upstreamBody, i < len(chain)-1)
		if err != nil {
			lastErr = err
			if retryable {
				continue
			}
			writeJSONError(w, http.StatusBadGateway, err.Error())
			return
		}

		if stream {
			s.writeGeminiStream(w, resp)
		} else {
			s.writeGeminiNonStream(w, resp)
		}
		s.logger.Info("served",
			"request_id", reqID,
			"frontend", "gemini",
			"client_model", clientModel,
			"resolved_model", hop.ModelName,
			"provider", hop.ProviderName,
			"stream", stream,
			"fallback_index", i,
			"latency_ms", time.Since(started).Milliseconds(),
		)
		return
	}
	writeJSONError(w, http.StatusBadGateway, fmt.Sprintf("all upstreams failed: %v", lastErr))
}

// sendGeminiChatRequest issues one attempt, returning the upstream
// response, whether the caller should retry on error, and the error.
func (s *Server) sendGeminiChatRequest(
	r *http.Request,
	hop *resolve.Resolved,
	body []byte,
	canRetry bool,
) (*http.Response, bool, error) {
	url := strings.TrimRight(hop.Provider.BaseURL, "/") + chatPath
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, false, fmt.Errorf("build upstream request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if authz, ok := s.auths[hop.ProviderName]; ok {
		if err := authz.Apply(r.Context(), req); err != nil {
			return nil, false, fmt.Errorf("auth: %w", err)
		}
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, true, fmt.Errorf("upstream transport: %w", err)
	}
	if resp.StatusCode >= 500 && canRetry {
		drain(resp.Body)
		resp.Body.Close()
		return nil, true, fmt.Errorf("upstream status %d", resp.StatusCode)
	}
	if resp.StatusCode != http.StatusOK {
		// Propagate non-retryable error as-is to the client.
		defer resp.Body.Close()
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
		return nil, false, fmt.Errorf("upstream status %d: %s", resp.StatusCode, truncateBytes(b, 400))
	}
	return resp, false, nil
}

func (s *Server) writeGeminiNonStream(w http.ResponseWriter, resp *http.Response) {
	defer resp.Body.Close()
	var oa gemini.OpenAIChatResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4*1024*1024)).Decode(&oa); err != nil {
		writeJSONError(w, http.StatusBadGateway, "parse upstream response: "+err.Error())
		return
	}
	out := gemini.ChatResponseFromOpenAI(&oa)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

// writeGeminiStream pipes OpenAI SSE chunks into Gemini's newline-delimited
// JSON streaming format. Each emitted line is one JSON ChatResponse chunk.
func (s *Server) writeGeminiStream(w http.ResponseWriter, resp *http.Response) {
	defer resp.Body.Close()

	// Gemini's streamGenerateContent responds with application/json and
	// sends JSON objects separated by commas inside a top-level array — but
	// the commonly-used google-genai libraries accept newline-delimited
	// JSON too, which is what the npm `@google/genai` uses via text/plain.
	// We use newline-delimited JSON here (Content-Type: application/json)
	// because it is simpler to produce token-by-token and widely consumed.
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)

	reader := bufio.NewReaderSize(resp.Body, 4096)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)

	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			line = bytes.TrimRight(line, "\r\n")
			if len(line) == 0 {
				continue
			}
			// SSE lines look like `data: {...}` or `data: [DONE]`.
			if !bytes.HasPrefix(line, []byte("data:")) {
				continue
			}
			payload := bytes.TrimSpace(line[len("data:"):])
			if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
				continue
			}
			chunk, cerr := gemini.StreamChunkFromOpenAI(payload)
			if cerr != nil || chunk == nil {
				continue
			}
			if werr := enc.Encode(chunk); werr != nil {
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

// handleEmbedContent handles POST /v1beta/models/{model}:embedContent.
func (s *Server) handleEmbedContent(w http.ResponseWriter, r *http.Request) {
	clientModel, ok := extractGeminiModel(r.URL.Path)
	if !ok {
		writeJSONError(w, http.StatusBadRequest, "could not parse model from URL")
		return
	}

	body, err := readBoundedBody(r)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	var in gemini.EmbedContentRequest
	if err := json.Unmarshal(body, &in); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid Gemini JSON body")
		return
	}

	res, err := s.resolver.Resolve(clientModel)
	if err != nil {
		writeJSONError(w, http.StatusNotFound, err.Error())
		return
	}

	oaReq, err := gemini.EmbedContentToOpenAI(&in, res.UpstreamModel)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	oaResp, err := s.sendGeminiEmbedRequest(r, res, oaReq)
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, err.Error())
		return
	}
	out, err := gemini.EmbedContentResponseFromOpenAI(oaResp)
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

// handleBatchEmbedContents handles POST /v1beta/models/{model}:batchEmbedContents.
func (s *Server) handleBatchEmbedContents(w http.ResponseWriter, r *http.Request) {
	clientModel, ok := extractGeminiModel(r.URL.Path)
	if !ok {
		writeJSONError(w, http.StatusBadRequest, "could not parse model from URL")
		return
	}

	body, err := readBoundedBody(r)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	var in gemini.BatchEmbedContentsRequest
	if err := json.Unmarshal(body, &in); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid Gemini JSON body")
		return
	}

	res, err := s.resolver.Resolve(clientModel)
	if err != nil {
		writeJSONError(w, http.StatusNotFound, err.Error())
		return
	}

	oaReq, err := gemini.BatchEmbedRequestToOpenAI(&in, res.UpstreamModel)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}

	oaResp, err := s.sendGeminiEmbedRequest(r, res, oaReq)
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, err.Error())
		return
	}
	out := gemini.BatchEmbedResponseFromOpenAI(oaResp)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(out)
}

// sendGeminiEmbedRequest POSTs an OpenAI-shaped embed request and parses
// the response.
func (s *Server) sendGeminiEmbedRequest(
	r *http.Request,
	hop *resolve.Resolved,
	oaReq *gemini.OpenAIEmbedRequest,
) (*gemini.OpenAIEmbedResponse, error) {
	body, err := json.Marshal(oaReq)
	if err != nil {
		return nil, err
	}
	url := strings.TrimRight(hop.Provider.BaseURL, "/") + embedPath
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build upstream request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if authz, ok := s.auths[hop.ProviderName]; ok {
		if err := authz.Apply(r.Context(), req); err != nil {
			return nil, fmt.Errorf("auth: %w", err)
		}
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("upstream transport: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
		return nil, fmt.Errorf("upstream status %d: %s", resp.StatusCode, truncateBytes(b, 400))
	}
	var out gemini.OpenAIEmbedResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, 8*1024*1024)).Decode(&out); err != nil {
		return nil, fmt.Errorf("parse upstream: %w", err)
	}
	return &out, nil
}

// extractGeminiModel pulls {model} out of /v1beta/models/{model}:action.
// Uses the last `:` to split so model names with colons in them (e.g.
// `gemma4:e4b`) round-trip correctly. Returns ("", false) on parse failure.
func extractGeminiModel(path string) (string, bool) {
	const prefix = "/v1beta/models/"
	if !strings.HasPrefix(path, prefix) {
		return "", false
	}
	rest := path[len(prefix):]
	colon := strings.LastIndexByte(rest, ':')
	if colon <= 0 {
		return "", false
	}
	return rest[:colon], true
}

func truncateBytes(b []byte, max int) string {
	if len(b) <= max {
		return string(b)
	}
	return string(b[:max]) + "…"
}
