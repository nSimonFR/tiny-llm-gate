package server

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/nSimonFR/tiny-llm-gate/internal/resolve"
)

// chatUpstream performs ONE hop for an OpenAI /chat/completions request and
// returns an OpenAI-shaped *http.Response, dispatching on the provider type.
// This is the provider-agnostic "chat core": both the OpenAI and Gemini
// frontends call it, so any provider type works behind either frontend.
//
//   - "openai":    the REAL upstream response, verbatim (status/headers/body).
//   - "codex":     a SYNTHETIC 200 whose Body carries translated OpenAI
//     chat.completion(.chunk) bytes (JSON for non-stream, SSE for stream).
//   - "anthropic": same as codex, translating the Anthropic Messages API via
//     the shared account pool.
//
// Return contract (mirrors the old sendGeminiChatRequest so callers stay simple):
//
//	(resp, false, nil)          : upstream responded — resp carries the real
//	                              status. 2xx ⇒ consume Body; non-2xx ⇒ each
//	                              frontend decides (OpenAI forwards verbatim,
//	                              Gemini emits its own 502). Caller MUST Close Body.
//	(nil,  true,  err)          : retryable failure — caller may try the next hop.
//	(nil,  false, err)          : terminal failure (translate/auth/last-hop 5xx
//	                              body error). errors.Is(err, context.Canceled)
//	                              means the client disconnected.
//
// Streaming codex translation runs in a goroutine feeding an io.Pipe; the
// returned Body is the pipe reader. The upstream fetch is bound to r.Context(),
// so a client disconnect cancels it, the goroutine's upstream Read errors, and
// it closes both the pipe (CloseWithError) and the upstream Body — no
// goroutine/connection leak. Callers Close the returned Body on their own path
// too, which also unblocks the goroutine.
func (s *Server) chatUpstream(
	r *http.Request,
	hop *resolve.Resolved,
	body []byte,
	isStream bool,
	canRetry bool,
) (*http.Response, bool, error) {
	// A provider whose authenticator failed to build at startup is disabled,
	// not fatal: fail this hop (retryable) so the caller falls back.
	if reason, bad := s.disabled[hop.ProviderName]; bad {
		return nil, true, fmt.Errorf("provider %q disabled at startup: %s", hop.ProviderName, reason)
	}

	switch hop.Provider.Type {
	case "codex":
		return s.codexUpstream(r, hop, body, isStream, canRetry)
	case "anthropic":
		return s.anthropicUpstream(r, hop, body, isStream, canRetry)
	default: // "openai" and any OpenAI-compatible upstream
		return s.forwardOpenAIChat(r, hop, body, canRetry)
	}
}

// forwardOpenAIChat is the byte-forward path for openai-type providers. It
// returns the REAL *http.Response (verbatim upstream status/headers/body — no
// re-serialization), so downstream observability headers survive. Non-2xx is
// NOT treated as an error here: it comes back as (resp, false, nil) and each
// frontend forwards or rejects it per its own semantics. Only 5xx while a
// fallback remains is turned into a retryable error.
func (s *Server) forwardOpenAIChat(
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
		if errors.Is(err, context.Canceled) {
			return nil, false, err // client disconnected
		}
		return nil, true, fmt.Errorf("upstream transport: %w", err)
	}

	// Retryable upstream error: consume body to free the connection and bubble
	// up so the caller tries the next fallback. Exception: some proxies wrap
	// client errors as 5xx — isWrappedClientError rewrites those to 400 and
	// forwards them instead of falling back.
	if resp.StatusCode >= 500 && canRetry {
		if isWrappedClientError(resp) {
			resp.StatusCode = http.StatusBadRequest
		} else {
			drain(resp.Body)
			resp.Body.Close()
			return nil, true, fmt.Errorf("upstream status %d", resp.StatusCode)
		}
	}
	return resp, false, nil
}

// synthOpenAIResponse wraps translated bytes/SSE as a 200 *http.Response so the
// frontends' existing writers (which only touch resp.Body) can consume it
// uniformly, exactly as they consume a real OpenAI-compatible upstream.
func synthOpenAIResponse(isStream bool, body io.ReadCloser) *http.Response {
	h := http.Header{}
	if isStream {
		h.Set("Content-Type", "text/event-stream")
		h.Set("Cache-Control", "no-cache")
	} else {
		h.Set("Content-Type", "application/json")
	}
	return &http.Response{StatusCode: http.StatusOK, Header: h, Body: body}
}

// pipeFlusher adapts an *io.PipeWriter to the translators' flusher interface
// (io.Writer + Flush). Flush is a no-op: io.Pipe is unbuffered — each Write
// blocks until the consumer reads — so the real client-side flush is owned by
// streamCopy / writeGeminiStream downstream.
type pipeFlusher struct{ w *io.PipeWriter }

func (p pipeFlusher) Write(b []byte) (int, error) { return p.w.Write(b) }
func (p pipeFlusher) Flush()                      {}
