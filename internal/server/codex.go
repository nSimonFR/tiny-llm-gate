package server

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/codex"
	"github.com/nSimonFR/tiny-llm-gate/internal/resolve"
)

// Codex Desktop fingerprint. These mirror icebear0828/codex-proxy's default
// config/fingerprint.yaml + the createResponseViaHttp headers, presenting the
// gate as the genuine Codex desktop client to the ChatGPT backend.
const (
	codexOriginator = "Codex Desktop"
	codexUserAgent  = "Codex Desktop/260202.0859 (darwin; arm64)"
	codexBeta       = "responses=experimental"
	codexResidency  = "us"
)

// codexInstallationID is a stable per-process UUID sent as
// x-codex-installation-id (the real client persists one per install; a stable
// value per gate process is close enough and avoids looking like a fresh
// client on every request).
var codexInstallationID = randomUUID()

// codexUpstream performs one hop for a "codex" provider: it translates the
// OpenAI chat/completions body to a Codex Responses request, POSTs it with the
// desktop fingerprint + OAuth auth, then translates the SSE back into the
// OpenAI chat wire format — buffered for non-stream, or piped through a
// goroutine for stream. It returns an OpenAI-shaped *http.Response per the
// chatUpstream contract (see chat.go). Codex is chat-only; embeddings never
// reach here.
//
// Disconnection: the upstream request is bound to r.Context(); a client drop
// cancels it, the streaming goroutine's upstream Read errors, and the goroutine
// closes both the pipe (CloseWithError) and the upstream Body — no leak.
func (s *Server) codexUpstream(
	r *http.Request,
	hop *resolve.Resolved,
	body []byte,
	isStream bool,
	canRetry bool,
) (*http.Response, bool, error) {
	codexBody, err := codex.TranslateRequest(body, hop.UpstreamModel)
	if err != nil {
		return nil, canRetry, fmt.Errorf("codex translate: %w", err)
	}

	up, retryable, err := s.codexPost(r, hop, codexBody, canRetry)
	if up == nil || up.StatusCode < 200 || up.StatusCode >= 300 {
		return up, retryable, err
	}

	tr := codex.NewTranslator(hop.ModelName)
	reqID := requestID(r.Context())
	tr.Reopen = func() (io.ReadCloser, error) {
		s.logger.Warn("codex: upstream stalled, retrying", "request_id", reqID, "model", hop.ModelName)
		up, _, err := s.codexPost(r, hop, codexBody, false)
		if err != nil {
			return nil, err
		}
		if up.StatusCode < 200 || up.StatusCode >= 300 {
			drain(up.Body)
			up.Body.Close()
			return nil, fmt.Errorf("codex upstream status %d on retry", up.StatusCode)
		}
		return up.Body, nil
	}

	if !isStream {
		out, cerr := tr.Collect(up.Body)
		up.Body.Close()
		if cerr != nil {
			return nil, canRetry, fmt.Errorf("codex translate: %w", cerr)
		}
		return synthOpenAIResponse(false, io.NopCloser(bytes.NewReader(out))), false, nil
	}

	pr, pw := io.Pipe()
	go func() {
		defer up.Body.Close()
		_, serr := tr.Stream(pipeFlusher{pw}, up.Body)
		if errors.Is(serr, codex.ErrStalled) {
			s.logger.Warn("codex: upstream stalled after output, ending stream", "request_id", reqID, "model", hop.ModelName)
		}
		pw.CloseWithError(serr)
	}()
	return synthOpenAIResponse(true, pr), false, nil
}

// codexPost POSTs a Codex Responses body to the ChatGPT backend with the
// desktop fingerprint + OAuth auth. Same return contract as chatUpstream; a
// non-2xx response comes back unread for the caller to forward.
func (s *Server) codexPost(r *http.Request, hop *resolve.Resolved, codexBody []byte, canRetry bool) (*http.Response, bool, error) {
	url := strings.TrimRight(hop.Provider.BaseURL, "/") + "/responses"
	// Per-attempt context: the stall guard cancels it to unblock a silent body.
	ctx, cancel := context.WithCancel(r.Context())
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(codexBody))
	if err != nil {
		cancel()
		return nil, false, fmt.Errorf("build codex request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("originator", codexOriginator)
	req.Header.Set("User-Agent", codexUserAgent)
	req.Header.Set("OpenAI-Beta", codexBeta)
	req.Header.Set("x-openai-internal-codex-residency", codexResidency)
	req.Header.Set("x-codex-installation-id", codexInstallationID)
	req.Header.Set("x-client-request-id", randomUUID())
	// Auth (oauth_chatgpt) sets Authorization + ChatGPT-Account-Id.
	if authz, ok := s.auths[hop.ProviderName]; ok {
		if err := authz.Apply(r.Context(), req); err != nil {
			cancel()
			return nil, false, fmt.Errorf("codex auth: %w", err)
		}
	}

	up, err := s.client.Do(req)
	if err != nil {
		cancel()
		if errors.Is(err, context.Canceled) {
			return nil, false, err // client disconnected
		}
		return nil, true, fmt.Errorf("codex transport: %w", err)
	}

	// Retry the next hop only on 5xx when a fallback remains (same policy as
	// the OpenAI path); any other non-2xx is returned for the frontend to
	// forward verbatim (OpenAI) or reject with 502 (Gemini).
	if shouldFallbackStatus(hop, up.StatusCode, canRetry) {
		drain(up.Body)
		up.Body.Close()
		cancel()
		return nil, true, fmt.Errorf("codex upstream status %d", up.StatusCode)
	}
	up.Body = guardStall(up.Body, cancel, codexFirstByteTimeout, codexIdleTimeout)
	return up, false, nil
}

// A healthy ChatGPT stream sends response.created within a second or two;
// 300s between bytes is Codex CLI's own stream_idle_timeout default.
var (
	codexFirstByteTimeout = 60 * time.Second
	codexIdleTimeout      = 300 * time.Second
)

// stallGuard cancels an upstream attempt that sends no byte for too long, so a
// silent stream fails with codex.ErrStalled instead of hanging until the client
// gives up. Close also cancels the attempt.
type stallGuard struct {
	body    io.ReadCloser
	cancel  context.CancelFunc
	timer   *time.Timer
	idle    time.Duration
	stalled atomic.Bool
}

func guardStall(body io.ReadCloser, cancel context.CancelFunc, first, idle time.Duration) *stallGuard {
	g := &stallGuard{body: body, cancel: cancel, idle: idle}
	g.timer = time.AfterFunc(first, func() {
		g.stalled.Store(true)
		cancel()
	})
	return g
}

func (g *stallGuard) Read(p []byte) (int, error) {
	n, err := g.body.Read(p)
	if n > 0 {
		g.timer.Reset(g.idle)
	}
	if err != nil && g.stalled.Load() {
		return n, codex.ErrStalled
	}
	return n, err
}

func (g *stallGuard) Close() error {
	g.timer.Stop()
	g.cancel()
	return g.body.Close()
}

func randomUUID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	h := hex.EncodeToString(b)
	return h[0:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:32]
}
