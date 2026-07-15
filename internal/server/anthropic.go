package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/auth"
)

// defaultAnthropicCooldown is used when a 429 response carries no usable
// Retry-After header.
const defaultAnthropicCooldown = 60 * time.Second

// authFailureCooldown is applied when an account returns 401/403 — an
// expired/invalid token doesn't self-heal on a Retry-After timescale, so cool
// it down long enough that the token-refresh sidecar can rotate it, rather
// than hammering the dead account (and failing every request) in the interim.
const authFailureCooldown = 15 * time.Minute

// usagePath is Anthropic's OAuth subscription usage endpoint, relative to the
// API root. A GET costs no tokens and answers even for exhausted accounts.
const usagePath = "/api/oauth/usage"

// exhaustedUtilization is the five-hour-window utilization at or above which
// a Retry-After-less 429 is treated as window exhaustion (bench until the
// window resets) rather than a transient burst throttle (keep the short
// default cooldown). The endpoint reports whole percents; a 429 plus >=95%
// means the window is spent.
const exhaustedUtilization = 95.0

// maxUsageBench caps how long a usage-derived bench can last, so a
// misparsed or weekly-scale resets_at can't silently remove an account from
// the pool for days — worst case we re-verify with one wasted 429 per cap.
const maxUsageBench = 6 * time.Hour

// anthropicAccount is one credential in the Anthropic account pool, with
// cooldown state for the sticky-until-429 failover strategy: the gate stays
// on one account until it 429s, then moves to the next available account and
// stays there (see pickAnthropicAccount / nextAnthropicAccount). Naive
// round-robin was rejected — alternating every request risks rate-limiting
// multiple accounts at once and more closely resembles ToS-gaming.
type anthropicAccount struct {
	name string
	auth auth.Authenticator
	// coolingDownUntil is a unix-seconds deadline; zero (or past) means
	// available. Set on a 429 (rate limit) or 401/403 (auth failure) from
	// this account's upstream request.
	coolingDownUntil atomic.Int64
	// usageCheckInFlight serializes benchIfExhausted lookups so a burst of
	// concurrent 429s doesn't stampede the usage endpoint.
	usageCheckInFlight atomic.Bool
}

func (a *anthropicAccount) available(now time.Time) bool {
	return a.coolingDownUntil.Load() <= now.Unix()
}

// cooldownUntil extends the bench monotonically: a later deadline always
// wins, an earlier one never shortens an existing bench. Without this, the
// all-cooling fallback path (pickAnthropicAccount returning a benched
// account) would get a fresh Retry-After-less 429 whose 60s default clobbers
// a correct bench-until-window-reset.
func (a *anthropicAccount) cooldownUntil(deadline int64) {
	for {
		cur := a.coolingDownUntil.Load()
		if deadline <= cur {
			return
		}
		if a.coolingDownUntil.CompareAndSwap(cur, deadline) {
			return
		}
	}
}

func (a *anthropicAccount) cooldown(d time.Duration) {
	a.cooldownUntil(time.Now().Add(d).Unix())
}

// pickAnthropicAccount returns the current sticky account, skipping ahead to
// the next available one if the current account is still cooling down.
// Returns nil when no accounts are configured (unauthenticated passthrough).
// If every account is cooling down, returns the current one anyway so its
// 429 can surface to the client rather than inventing an error.
func (s *Server) pickAnthropicAccount() *anthropicAccount {
	n := len(s.anthropicAccounts)
	if n == 0 {
		return nil
	}
	now := time.Now()
	start := int(s.currentAnthropic.Load())
	for i := 0; i < n; i++ {
		idx := (start + i) % n
		if s.anthropicAccounts[idx].available(now) {
			if idx != start {
				s.currentAnthropic.Store(int32(idx))
			}
			return s.anthropicAccounts[idx]
		}
	}
	return s.anthropicAccounts[start]
}

// nextAnthropicAccount advances the sticky index past the current account
// (which just 429'd) and returns the newly selected account's name, for
// logging.
func (s *Server) nextAnthropicAccount() string {
	n := len(s.anthropicAccounts)
	if n == 0 {
		return ""
	}
	next := (int(s.currentAnthropic.Load()) + 1) % n
	s.currentAnthropic.Store(int32(next))
	return s.anthropicAccounts[next].name
}

// parseRetryAfter reads a Retry-After header value (seconds form only —
// Anthropic sends seconds, not an HTTP-date) and falls back to
// defaultAnthropicCooldown when absent or unparseable.
func parseRetryAfter(v string) time.Duration {
	if secs, err := strconv.Atoi(v); err == nil && secs > 0 {
		return time.Duration(secs) * time.Second
	}
	return defaultAnthropicCooldown
}

// failoverTrigger classifies an upstream response into a failover decision.
// A 429 (rate limited) or a 401/403 (expired/invalid token) cools the current
// account down and moves to the next one; any other status (including 2xx) is
// returned to the client unchanged. Returns the cooldown to apply and a short
// reason for logging; ok=false means "not a failover trigger".
func failoverTrigger(resp *http.Response) (cooldown time.Duration, reason string, ok bool) {
	switch resp.StatusCode {
	case http.StatusTooManyRequests:
		return parseRetryAfter(resp.Header.Get("Retry-After")), "429", true
	case http.StatusUnauthorized, http.StatusForbidden:
		return authFailureCooldown, "auth_error", true
	default:
		return 0, "", false
	}
}

// benchIfExhausted disambiguates a Retry-After-less 429. Subscription (OAuth)
// accounts get no rate-limit headers, so such a 429 is either a transient
// burst throttle (recovers in seconds — the 60s default is right) or an
// exhausted usage window (recovers at resets_at, hours away — 60s means we
// re-select a dead account every minute). The free usage endpoint tells the
// two apart: at >= exhaustedUtilization the window is spent, so extend the
// bench to resets_at (capped at maxUsageBench). Best-effort: any failure
// leaves the short default in place.
func (s *Server) benchIfExhausted(ctx context.Context, acct *anthropicAccount, reqID string) {
	// Already hard-benched (e.g. by a previous check or a real Retry-After)
	// — nothing to learn.
	if acct.coolingDownUntil.Load() > time.Now().Add(2*time.Minute).Unix() {
		return
	}
	if !acct.usageCheckInFlight.CompareAndSwap(false, true) {
		return // another request is already checking this account
	}
	defer acct.usageCheckInFlight.Store(false)

	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		strings.TrimRight(s.cfg.Anthropic.Upstream, "/")+usagePath, nil)
	if err != nil {
		return
	}
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("anthropic-beta", "oauth-2025-04-20")
	if err := acct.auth.Apply(ctx, req); err != nil {
		return
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return
	}
	var usage struct {
		FiveHour struct {
			Utilization float64   `json:"utilization"`
			ResetsAt    time.Time `json:"resets_at"`
		} `json:"five_hour"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&usage); err != nil {
		return
	}
	if usage.FiveHour.Utilization < exhaustedUtilization || usage.FiveHour.ResetsAt.IsZero() {
		return // burst throttle, not exhaustion — the short default stands
	}
	deadline := usage.FiveHour.ResetsAt
	if capAt := time.Now().Add(maxUsageBench); deadline.After(capAt) {
		deadline = capAt
	}
	acct.cooldownUntil(deadline.Unix())
	s.logger.Warn("anthropic: account window exhausted, benched until reset",
		"request_id", reqID,
		"account", acct.name,
		"utilization", usage.FiveHour.Utilization,
		"resets_at", usage.FiveHour.ResetsAt.Format(time.RFC3339),
		"benched_until", deadline.Format(time.RFC3339),
	)
}

// handleAnthropicMessages proxies POST /v1/messages to the configured
// Anthropic upstream. The client's request body is forwarded as-is; non-auth
// headers pass through (so anthropic-version, anthropic-beta, etc. reach
// Anthropic unchanged), but any incoming Authorization header is stripped
// and replaced with the gate's configured auth.
//
// With multiple accounts configured, a 429 (rate limit) or 401/403 (expired/
// invalid token) from the current account cools it down — 429 for Retry-After
// (or defaultAnthropicCooldown), auth errors for authFailureCooldown — and
// transparently retries the same request on the next available account,
// bounded to one pass through the pool so a request can't loop forever. The
// retry happens before any bytes are written to the client, so it's invisible
// to callers. A single-account gate has no failover target, so its 401/429
// passes straight through.
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

	attempts := len(s.anthropicAccounts)
	if attempts == 0 {
		attempts = 1 // unauthenticated passthrough — single attempt, no failover.
	}

	var resp *http.Response
	var accountName string
	for attempt := 0; attempt < attempts; attempt++ {
		req, buildErr := http.NewRequestWithContext(r.Context(), http.MethodPost, upstream, bytes.NewReader(body))
		if buildErr != nil {
			writeJSONError(w, http.StatusBadGateway, fmt.Sprintf("build upstream request: %v", buildErr))
			return
		}

		// Forward non-auth client headers (anthropic-version, anthropic-beta,
		// content-type, X-Stainless-*, etc.) then overwrite Authorization with
		// our configured credential. Accept-Encoding is forced to identity so
		// we forward plaintext to the client (the upstream might otherwise
		// choose gzip and we'd relay a compressed body).
		//
		// Also strip x-api-key: Anthropic prefers it over Authorization when
		// both are present, so leaving a client-supplied x-api-key would defeat
		// the auth replacement. Clients like pi-coding-agent send x-api-key by
		// default; Claude Code uses Authorization Bearer, so this is a no-op
		// for it.
		copyHeaders(req.Header, r.Header)
		req.Header.Del("Authorization")
		req.Header.Del("x-api-key")
		req.Header.Set("Accept-Encoding", "identity")

		acct := s.pickAnthropicAccount()
		if acct != nil {
			accountName = acct.name
			if err := acct.auth.Apply(r.Context(), req); err != nil {
				writeJSONError(w, http.StatusBadGateway, fmt.Sprintf("apply anthropic auth: %v", err))
				return
			}
		}

		var doErr error
		resp, doErr = s.client.Do(req)
		if doErr != nil {
			if errors.Is(doErr, context.Canceled) {
				return // client disconnected; no response to write.
			}
			writeJSONError(w, http.StatusBadGateway, fmt.Sprintf("upstream transport: %v", doErr))
			s.logger.Error("anthropic: upstream", "request_id", reqID, "err", doErr)
			return
		}

		cooldown, reason, failover := failoverTrigger(resp)
		if !failover || acct == nil || len(s.anthropicAccounts) < 2 {
			break
		}

		// Cool the failing account regardless. If the pool is now exhausted
		// (this was the last attempt), keep this response so its status and
		// body reach the client instead of being discarded into an empty body.
		acct.cooldown(cooldown)
		// A 429 with no Retry-After got the short default above — check
		// whether it's actually window exhaustion and extend the bench to
		// the window reset if so. Runs on the last attempt too: the client
		// gets this 429 either way, but future requests shouldn't re-select
		// a dead account every minute.
		if reason == "429" && resp.Header.Get("Retry-After") == "" {
			s.benchIfExhausted(r.Context(), acct, reqID)
		}
		if attempt == attempts-1 {
			break
		}
		next := s.nextAnthropicAccount()
		s.logger.Warn("anthropic: account switch",
			"request_id", reqID,
			"from", acct.name,
			"to", next,
			"reason", reason,
			"status", resp.StatusCode,
			"cooldown_s", cooldown.Seconds(),
		)
		resp.Body.Close()
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
		"account", accountName,
		"stream", peek.Stream,
		"status", resp.StatusCode,
		"latency_ms", time.Since(started).Milliseconds(),
	)
}
