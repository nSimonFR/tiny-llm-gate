package server

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/messages"
	"github.com/nSimonFR/tiny-llm-gate/internal/oaichat"
	"github.com/nSimonFR/tiny-llm-gate/internal/resolve"
	"github.com/nSimonFR/tiny-llm-gate/internal/responses"
)

// errBuild marks a request the frontend could not translate: identical on
// every hop, so it is a 400 rather than a fallback.
var errBuild = errors.New("bad request")

// chatChain runs a translated frontend's request through the chat core,
// building the chat body per hop, and returns the first non-retryable
// upstream response (2xx or not). The upstream is always asked to stream.
func (s *Server) chatChain(
	r *http.Request,
	chain []string,
	build func(hop *resolve.Resolved) ([]byte, error),
) (*http.Response, *resolve.Resolved, int, error) {
	var lastErr error
	for i, name := range chain {
		hop, err := s.resolver.Resolve(name)
		if err != nil {
			lastErr = err
			continue
		}
		body, err := build(hop)
		if err != nil {
			return nil, nil, i, fmt.Errorf("%w: %v", errBuild, err)
		}
		if hop.ReasoningEffort != nil {
			body = injectReasoningEffort(body, *hop.ReasoningEffort)
		}
		resp, retryable, err := s.chatUpstream(r, hop, body, true, i < len(chain)-1)
		if err != nil {
			if errors.Is(err, context.Canceled) || !retryable {
				return nil, hop, i, err
			}
			lastErr = err
			continue
		}
		return resp, hop, i, nil
	}
	return nil, nil, len(chain), fmt.Errorf("all upstreams failed: %v", lastErr)
}

type flushWriter struct{ http.ResponseWriter }

func (f flushWriter) Flush() {
	if fl, ok := f.ResponseWriter.(http.Flusher); ok {
		fl.Flush()
	}
}

// chatEncoder is what both translated frontends' encoders provide.
type chatEncoder interface {
	oaichat.Handler
	Start() error
	Complete() ([]byte, error)
	Fail(msg string)
}

// relay decodes a 2xx chat response through enc and writes the frontend's
// shape: SSE when stream, else one JSON document.
func relay(w http.ResponseWriter, resp *http.Response, stream bool, enc chatEncoder, writeErr func(int, string)) {
	defer resp.Body.Close()
	if stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		if err := enc.Start(); err != nil {
			return
		}
		if err := oaichat.Decode(resp.Body, enc); err != nil {
			enc.Fail(err.Error())
			return
		}
		_, _ = enc.Complete()
		return
	}
	if err := oaichat.Decode(resp.Body, enc); err != nil {
		writeErr(http.StatusBadGateway, err.Error())
		return
	}
	out, err := enc.Complete()
	if err != nil {
		writeErr(http.StatusBadGateway, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(out)
}

func upstreamError(resp *http.Response) string {
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	resp.Body.Close()
	return oaichat.ErrorMessage(b)
}

// handleResponses serves POST /v1/responses (OpenAI Responses API) for every
// provider type, via the chat core.
func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	started := time.Now()
	body, err := readBoundedBody(r)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	req, err := responses.Parse(body)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.Model == "" {
		writeJSONError(w, http.StatusBadRequest, "missing 'model' field")
		return
	}
	res, err := s.resolver.Resolve(req.Model)
	if err != nil {
		writeJSONError(w, http.StatusNotFound, err.Error())
		return
	}
	if !s.modelAllowed(w, r, req.Model, res) {
		return
	}
	chain := append([]string{res.ModelName}, res.Fallback...)
	resp, hop, idx, err := s.chatChain(r, chain, func(hop *resolve.Resolved) ([]byte, error) {
		return req.ToChat(hop.UpstreamModel)
	})
	if err != nil {
		switch {
		case errors.Is(err, context.Canceled):
		case errors.Is(err, errBuild):
			writeJSONError(w, http.StatusBadRequest, err.Error())
		default:
			writeJSONError(w, http.StatusBadGateway, err.Error())
		}
		return
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		writeJSONError(w, resp.StatusCode, upstreamError(resp))
		return
	}
	var sink responses.Flusher
	if req.Stream {
		sink = flushWriter{w}
	}
	relay(w, resp, req.Stream, responses.NewEncoder(sink, req, req.Model),
		func(code int, msg string) { writeJSONError(w, code, msg) })
	s.logServed(r, "responses", req.Model, hop, req.Stream, idx, started)
}

// serveMessagesTranslated answers /v1/messages for a non-Anthropic model.
func (s *Server) serveMessagesTranslated(w http.ResponseWriter, r *http.Request, body []byte, res *resolve.Resolved) {
	started := time.Now()
	writeErr := func(code int, msg string) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(code)
		_, _ = w.Write(messages.ErrorBody(messages.ErrorType(code), msg))
	}
	req, err := messages.Parse(body)
	if err != nil {
		writeErr(http.StatusBadRequest, err.Error())
		return
	}
	chain := append([]string{res.ModelName}, res.Fallback...)
	resp, hop, idx, err := s.chatChain(r, chain, func(hop *resolve.Resolved) ([]byte, error) {
		return req.ToChat(hop.UpstreamModel)
	})
	if err != nil {
		switch {
		case errors.Is(err, context.Canceled):
		case errors.Is(err, errBuild):
			writeErr(http.StatusBadRequest, err.Error())
		default:
			writeErr(http.StatusBadGateway, err.Error())
		}
		return
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		writeErr(resp.StatusCode, upstreamError(resp))
		return
	}
	var sink messages.Flusher
	if req.Stream {
		sink = flushWriter{w}
	}
	relay(w, resp, req.Stream, messages.NewEncoder(sink, req.Model), writeErr)
	s.logServed(r, "anthropic", req.Model, hop, req.Stream, idx, started)
}

func (s *Server) logServed(r *http.Request, frontend, clientModel string, hop *resolve.Resolved, stream bool, idx int, started time.Time) {
	s.logger.Info("served",
		"request_id", requestID(r.Context()),
		"frontend", frontend,
		"client", clientName(r),
		"app", appName(r),
		"client_model", clientModel,
		"resolved_model", hop.ModelName,
		"provider", hop.ProviderName,
		"stream", stream,
		"fallback_index", idx,
		"latency_ms", time.Since(started).Milliseconds(),
	)
}
