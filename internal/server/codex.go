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

// sendCodex handles one hop for a provider of type "codex". It translates the
// OpenAI chat/completions body to a Codex Responses request, posts it to the
// Codex backend with the desktop fingerprint + OAuth auth, then translates the
// SSE response back to OpenAI shape (streaming or aggregated).
//
// Return semantics match sendUpstream: done=true means bytes were committed to
// the client (no fallback), done=false means the caller may try the next hop.
// Only embeddings are unsupported (codex is chat-only) — that is a config
// error and surfaces as done=false so a fallback can still serve.
func (s *Server) sendCodex(
	w http.ResponseWriter,
	r *http.Request,
	hop *resolve.Resolved,
	upstreamPath string,
	body []byte,
	isStream bool,
	canRetry bool,
) (done bool, err error) {
	if upstreamPath != chatPath {
		return false, fmt.Errorf("codex provider does not support %s", upstreamPath)
	}

	codexBody, err := codex.TranslateRequest(body, hop.UpstreamModel)
	if err != nil {
		return false, err
	}

	url := strings.TrimRight(hop.Provider.BaseURL, "/") + "/responses"
	req, err := http.NewRequestWithContext(r.Context(), http.MethodPost, url, bytes.NewReader(codexBody))
	if err != nil {
		return false, fmt.Errorf("build codex request: %w", err)
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
			return false, fmt.Errorf("codex auth: %w", err)
		}
	}

	resp, err := s.client.Do(req)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			return true, nil
		}
		return false, fmt.Errorf("codex transport: %w", err)
	}
	defer resp.Body.Close()

	// Non-2xx: surface upstream status. Retry the next hop only on 5xx when a
	// fallback remains (same policy as the OpenAI path); 4xx is terminal.
	if resp.StatusCode >= 500 && canRetry {
		drain(resp.Body)
		return false, fmt.Errorf("codex upstream status %d", resp.StatusCode)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		// Forward the upstream error body verbatim to the client.
		copyHeaders(w.Header(), resp.Header)
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
		return true, nil
	}

	tr := codex.NewTranslator(hop.ModelName)
	if isStream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		fw := &flushWriter{w: w}
		if _, err := tr.Stream(fw, resp.Body); err != nil {
			// Bytes already partially sent; can't fall back. Best effort: the
			// translator emits a terminal error only before writing, so a
			// mid-stream failure just truncates. Log via caller.
			return true, err
		}
		return true, nil
	}

	// Non-streaming: aggregate the SSE stream into one chat.completion JSON.
	out, err := tr.Collect(resp.Body)
	if err != nil {
		if canRetry {
			return false, err
		}
		writeJSONError(w, http.StatusBadGateway, err.Error())
		return true, err
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(out)
	return true, nil
}

// flushWriter adapts an http.ResponseWriter to codex.flusher, flushing after
// every write so SSE chunks reach the client promptly.
type flushWriter struct {
	w http.ResponseWriter
}

func (f *flushWriter) Write(p []byte) (int, error) { return f.w.Write(p) }

func (f *flushWriter) Flush() {
	if fl, ok := f.w.(http.Flusher); ok {
		fl.Flush()
	}
}

func randomUUID() string {
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	h := hex.EncodeToString(b)
	return h[0:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:32]
}
