package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
)

// buildAnthropicServer builds a Server with an Anthropic config pointing at
// the given upstream URL and (optionally) a bearer token. Minimum providers
// + models are included to satisfy config validation.
func buildAnthropicServer(t *testing.T, upstreamURL, bearerToken string) *Server {
	t.Helper()
	cfg := &config.Config{
		Listen: "127.0.0.1:0",
		Providers: map[string]config.Provider{
			"stub": {Type: "openai", BaseURL: "http://localhost:1/v1"},
		},
		Models: map[string]config.Model{
			"stub": {Provider: "stub", UpstreamModel: "stub"},
		},
		Anthropic: &config.Anthropic{
			Upstream: upstreamURL,
		},
	}
	if bearerToken != "" {
		cfg.Anthropic.Auth = &config.Auth{Type: "bearer", Token: bearerToken}
	}
	s, err := New(cfg, discardLogger())
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	return s
}

// postMessages sends a POST /v1/messages request with a fake OAuth
// Authorization header and the Anthropic-specific metadata headers.
func postMessages(t *testing.T, h http.Handler, body map[string]any) (*httptest.ResponseRecorder, *http.Request) {
	t.Helper()
	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages?beta=true", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer sk-ant-oat01-incoming-token")
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("anthropic-beta", "oauth-2025-04-20")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec, req
}

func TestAnthropicPassthroughHeaders(t *testing.T) {
	var gotHeaders http.Header
	var gotPath string
	var gotBody []byte

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeaders = r.Header.Clone()
		gotPath = r.URL.Path + "?" + r.URL.RawQuery
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"type":"message","id":"msg_1","model":"claude-opus-4-6","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":2}}`))
	}))
	defer upstream.Close()

	s := buildAnthropicServer(t, upstream.URL, "")
	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 100,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})

	if rec.Code != 200 {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	// Non-auth headers pass through.
	if got := gotHeaders.Get("Anthropic-Version"); got != "2023-06-01" {
		t.Errorf("anthropic-version = %q", got)
	}
	if got := gotHeaders.Get("Anthropic-Beta"); got != "oauth-2025-04-20" {
		t.Errorf("anthropic-beta = %q", got)
	}
	// Query string preserved.
	if gotPath != "/v1/messages?beta=true" {
		t.Errorf("path = %q", gotPath)
	}
	// Body passes through unchanged.
	if !bytes.Contains(gotBody, []byte(`"claude-opus-4-6"`)) {
		t.Errorf("body missing model: %s", gotBody)
	}
}

func TestAnthropicStripsIncomingAuthWhenNoneConfigured(t *testing.T) {
	// Without configured auth, the handler must STRIP the incoming
	// Authorization header so we don't leak Aperture's apikey upstream.
	var gotAuth string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"type":"message","id":"msg_1","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer upstream.Close()

	s := buildAnthropicServer(t, upstream.URL, "")
	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})

	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}
	if gotAuth != "" {
		t.Errorf("expected Authorization stripped, got %q", gotAuth)
	}
}

func TestAnthropicAppliesConfiguredAuth(t *testing.T) {
	// With configured bearer token, the handler must replace the incoming
	// Authorization with the configured one (not pass the incoming one through).
	var gotAuth string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"type":"message","id":"msg_1","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer upstream.Close()

	s := buildAnthropicServer(t, upstream.URL, "sk-ant-oat01-configured-bot-token")
	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})

	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}
	if gotAuth != "Bearer sk-ant-oat01-configured-bot-token" {
		t.Errorf("upstream Authorization = %q; expected configured bot token", gotAuth)
	}
}

func TestAnthropicStreamingRelayedUnchanged(t *testing.T) {
	sseChunks := []string{
		"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-opus-4-6\"}}\n\n",
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n",
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
	}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(200)
		flusher, _ := w.(http.Flusher)
		for _, chunk := range sseChunks {
			_, _ = w.Write([]byte(chunk))
			if flusher != nil {
				flusher.Flush()
			}
		}
	}))
	defer upstream.Close()

	s := buildAnthropicServer(t, upstream.URL, "")
	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
		"stream":     true,
	})

	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}
	body := rec.Body.String()
	for _, want := range []string{"message_start", "Hello", "message_stop"} {
		if !strings.Contains(body, want) {
			t.Errorf("streamed body missing %q:\n%s", want, body)
		}
	}
}

func TestAnthropicUpstreamErrorPassesThrough(t *testing.T) {
	// 4xx/5xx responses from upstream are forwarded to the client as-is,
	// no rewriting.
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(401)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"authentication_error","message":"invalid token"}}`))
	}))
	defer upstream.Close()

	s := buildAnthropicServer(t, upstream.URL, "sk-ant-oat01-bad-token")
	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})

	if rec.Code != 401 {
		t.Fatalf("expected 401 pass-through, got %d", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "authentication_error") {
		t.Errorf("body missing upstream error message: %s", rec.Body.String())
	}
}

func TestAnthropicRouteNotRegisteredWithoutConfig(t *testing.T) {
	// When the anthropic config section is absent, /v1/messages must not
	// be routed (returns 404 / 405).
	s := buildServer(t,
		map[string]config.Provider{
			"ollama": {Type: "openai", BaseURL: "http://localhost:1/v1"},
		},
		map[string]config.Model{
			"stub": {Provider: "ollama", UpstreamModel: "stub"},
		},
		nil,
	)

	b, _ := json.Marshal(map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	if rec.Code == 200 {
		t.Errorf("expected non-200 when anthropic not configured, got %d", rec.Code)
	}
}

// buildAnthropicAccountsServer builds a Server with a multi-account Anthropic
// config, all pointed at the same test upstream (accounts differ only in
// which bearer token they send).
func buildAnthropicAccountsServer(t *testing.T, upstreamURL string, accounts ...config.AnthropicAccount) *Server {
	t.Helper()
	cfg := &config.Config{
		Listen: "127.0.0.1:0",
		Providers: map[string]config.Provider{
			"stub": {Type: "openai", BaseURL: "http://localhost:1/v1"},
		},
		Models: map[string]config.Model{
			"stub": {Provider: "stub", UpstreamModel: "stub"},
		},
		Anthropic: &config.Anthropic{
			Upstream: upstreamURL,
			Accounts: accounts,
		},
	}
	s, err := New(cfg, discardLogger())
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	return s
}

func TestAnthropicFailoverOnRateLimit(t *testing.T) {
	var hits []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tok := r.Header.Get("Authorization")
		hits = append(hits, tok)
		if tok == "Bearer acct1-token" {
			w.Header().Set("Retry-After", "30")
			w.WriteHeader(429)
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"type":"message","id":"msg_1","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer upstream.Close()

	s := buildAnthropicAccountsServer(t, upstream.URL,
		config.AnthropicAccount{Name: "acct1", Auth: &config.Auth{Type: "bearer", Token: "acct1-token"}},
		config.AnthropicAccount{Name: "acct2", Auth: &config.Auth{Type: "bearer", Token: "acct2-token"}},
	)

	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 200 {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if len(hits) != 2 || hits[0] != "Bearer acct1-token" || hits[1] != "Bearer acct2-token" {
		t.Fatalf("expected failover acct1->acct2, got %v", hits)
	}

	// A second request must go straight to acct2 (sticky), not back to acct1.
	rec2, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi again"}},
	})
	if rec2.Code != 200 {
		t.Fatalf("status = %d body=%s", rec2.Code, rec2.Body.String())
	}
	if len(hits) != 3 || hits[2] != "Bearer acct2-token" {
		t.Fatalf("expected second request to stick to acct2, got %v", hits)
	}
}

func TestAnthropicAllAccountsCoolingDownReturnsUpstream429(t *testing.T) {
	var hitCount int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hitCount++
		w.Header().Set("Retry-After", "30")
		w.WriteHeader(429)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}`))
	}))
	defer upstream.Close()

	s := buildAnthropicAccountsServer(t, upstream.URL,
		config.AnthropicAccount{Name: "acct1", Auth: &config.Auth{Type: "bearer", Token: "acct1-token"}},
		config.AnthropicAccount{Name: "acct2", Auth: &config.Auth{Type: "bearer", Token: "acct2-token"}},
	)

	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 429 {
		t.Fatalf("expected 429 when all accounts cooling down, got %d", rec.Code)
	}
	if hitCount != 2 {
		t.Fatalf("expected exactly one attempt per account (2), got %d", hitCount)
	}
}

func TestAnthropicFailoverOnAuthError(t *testing.T) {
	// An expired/invalid token surfaces as 401 (not 429). The gate must treat
	// it as a failover trigger and retry on the next account, otherwise a dead
	// primary token wedges the whole pool (the rpi5 outage this fixes).
	var hits []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tok := r.Header.Get("Authorization")
		hits = append(hits, tok)
		if tok == "Bearer acct1-token" {
			w.WriteHeader(401)
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"authentication_error","message":"OAuth access token has expired. Re-authenticate to continue."}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"type":"message","id":"msg_1","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer upstream.Close()

	s := buildAnthropicAccountsServer(t, upstream.URL,
		config.AnthropicAccount{Name: "acct1", Auth: &config.Auth{Type: "bearer", Token: "acct1-token"}},
		config.AnthropicAccount{Name: "acct2", Auth: &config.Auth{Type: "bearer", Token: "acct2-token"}},
	)

	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 200 {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if len(hits) != 2 || hits[0] != "Bearer acct1-token" || hits[1] != "Bearer acct2-token" {
		t.Fatalf("expected auth-error failover acct1->acct2, got %v", hits)
	}

	// The failed account is cooled down, so a second request sticks to acct2.
	rec2, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi again"}},
	})
	if rec2.Code != 200 {
		t.Fatalf("status = %d body=%s", rec2.Code, rec2.Body.String())
	}
	if len(hits) != 3 || hits[2] != "Bearer acct2-token" {
		t.Fatalf("expected second request to stick to acct2, got %v", hits)
	}
}

func TestAnthropicAllAccountsAuthErrorReturnsUpstream401(t *testing.T) {
	// When every account's token is dead, the gate makes exactly one attempt
	// per account (bounded, no infinite loop) and surfaces the last 401.
	var hitCount int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hitCount++
		w.WriteHeader(401)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"authentication_error","message":"invalid token"}}`))
	}))
	defer upstream.Close()

	s := buildAnthropicAccountsServer(t, upstream.URL,
		config.AnthropicAccount{Name: "acct1", Auth: &config.Auth{Type: "bearer", Token: "acct1-token"}},
		config.AnthropicAccount{Name: "acct2", Auth: &config.Auth{Type: "bearer", Token: "acct2-token"}},
	)

	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 401 {
		t.Fatalf("expected 401 when all accounts dead, got %d", rec.Code)
	}
	if hitCount != 2 {
		t.Fatalf("expected exactly one attempt per account (2), got %d", hitCount)
	}
	if !strings.Contains(rec.Body.String(), "authentication_error") {
		t.Errorf("body missing upstream error: %s", rec.Body.String())
	}
}

func TestAnthropicSingleAccountRateLimitPassesThroughNoRetry(t *testing.T) {
	var hitCount int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hitCount++
		w.WriteHeader(429)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}`))
	}))
	defer upstream.Close()

	s := buildAnthropicServer(t, upstream.URL, "sk-ant-oat01-only-token")
	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 429 {
		t.Fatalf("expected 429 pass-through, got %d", rec.Code)
	}
	if hitCount != 1 {
		t.Fatalf("expected exactly 1 upstream call (no failover target), got %d", hitCount)
	}
}

func TestAnthropicCooldownMonotonic(t *testing.T) {
	// A short cooldown must never shorten an existing longer bench — the
	// all-cooling fallback path re-hits benched accounts, and their fresh
	// Retry-After-less 429s would otherwise clobber a bench-until-reset.
	var a anthropicAccount
	far := time.Now().Add(2 * time.Hour).Unix()
	a.cooldownUntil(far)
	a.cooldown(60 * time.Second)
	if got := a.coolingDownUntil.Load(); got != far {
		t.Fatalf("short cooldown shortened the bench: got %d want %d", got, far)
	}
	farther := time.Now().Add(3 * time.Hour).Unix()
	a.cooldownUntil(farther)
	if got := a.coolingDownUntil.Load(); got != farther {
		t.Fatalf("cooldownUntil did not extend: got %d want %d", got, farther)
	}
}

// buildUsageUpstream serves /v1/messages (Retry-After-less 429 for acct1,
// 200 for acct2) plus the usage endpoint with the given status/utilization.
// Returns the server and a pointer to the Authorization header seen by the
// usage endpoint.
func buildUsageUpstream(t *testing.T, usageStatus int, utilization float64, resetsAt time.Time) (*httptest.Server, *string) {
	t.Helper()
	usageAuth := new(string)
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/messages", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") == "Bearer acct1-token" {
			w.WriteHeader(429) // deliberately no Retry-After — the OAuth case
			_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"rate limited"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"type":"message","id":"msg_1","usage":{"input_tokens":1,"output_tokens":1}}`))
	})
	mux.HandleFunc(usagePath, func(w http.ResponseWriter, r *http.Request) {
		*usageAuth = r.Header.Get("Authorization")
		if usageStatus != 200 {
			w.WriteHeader(usageStatus)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"five_hour":{"utilization":%g,"resets_at":%q}}`,
			utilization, resetsAt.Format(time.RFC3339))
	})
	return httptest.NewServer(mux), usageAuth
}

func rateLimit429(t *testing.T, upstreamURL string) *Server {
	t.Helper()
	return buildAnthropicAccountsServer(t, upstreamURL,
		config.AnthropicAccount{Name: "acct1", Auth: &config.Auth{Type: "bearer", Token: "acct1-token"}},
		config.AnthropicAccount{Name: "acct2", Auth: &config.Auth{Type: "bearer", Token: "acct2-token"}},
	)
}

func TestAnthropicExhausted429BenchedUntilReset(t *testing.T) {
	// A Retry-After-less 429 from an account whose usage window is spent
	// must bench that account until the window reset, not the 60s default
	// (which would re-select a dead account every minute for hours).
	resetsAt := time.Now().Add(2 * time.Hour).UTC()
	upstream, usageAuth := buildUsageUpstream(t, 200, 100.0, resetsAt)
	defer upstream.Close()

	s := rateLimit429(t, upstream.URL)
	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 200 {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if *usageAuth != "Bearer acct1-token" {
		t.Fatalf("usage check used wrong credential: %q", *usageAuth)
	}
	benched := s.anthropicAccounts[0].coolingDownUntil.Load()
	if want := resetsAt.Truncate(time.Second).Unix(); benched != want {
		t.Fatalf("acct1 benched until %d, want window reset %d", benched, want)
	}
}

func TestAnthropicTransient429KeepsShortCooldown(t *testing.T) {
	// Below the exhaustion threshold the 429 is a burst throttle — the short
	// default cooldown is correct and must not be extended to the reset.
	upstream, _ := buildUsageUpstream(t, 200, 40.0, time.Now().Add(2*time.Hour))
	defer upstream.Close()

	s := rateLimit429(t, upstream.URL)
	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 200 {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	benched := s.anthropicAccounts[0].coolingDownUntil.Load()
	if benched <= time.Now().Unix() {
		t.Fatal("expected a short cooldown to be set")
	}
	if max := time.Now().Add(2 * time.Minute).Unix(); benched > max {
		t.Fatalf("transient 429 over-benched: until %d (> now+2m %d)", benched, max)
	}
}

func TestAnthropicUsageEndpointFailureKeepsShortCooldown(t *testing.T) {
	// The usage lookup is best-effort: a failing endpoint must leave the
	// short default in place, never error the client request.
	upstream, _ := buildUsageUpstream(t, 500, 0, time.Time{})
	defer upstream.Close()

	s := rateLimit429(t, upstream.URL)
	rec, _ := postMessages(t, s.Handler(), map[string]any{
		"model":      "claude-opus-4-6",
		"max_tokens": 10,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})
	if rec.Code != 200 {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	benched := s.anthropicAccounts[0].coolingDownUntil.Load()
	if benched <= time.Now().Unix() {
		t.Fatal("expected the short default cooldown despite usage failure")
	}
	if max := time.Now().Add(2 * time.Minute).Unix(); benched > max {
		t.Fatalf("usage failure over-benched: until %d", benched)
	}
}

func TestAnthropicBodyPassesThroughUnchanged(t *testing.T) {
	// Verify we do NOT rewrite the model field or any other field in the
	// body — unlike /v1/chat/completions which rewrites client-facing
	// model names to upstream model ids.
	var gotBody []byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		_, _ = w.Write([]byte(`{"type":"message","id":"msg_1","usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer upstream.Close()

	s := buildAnthropicServer(t, upstream.URL, "")
	originalBody := map[string]any{
		"model":       "claude-opus-4-6",
		"max_tokens":  100,
		"temperature": 0.7,
		"system":      "You are helpful.",
		"messages": []map[string]any{
			{"role": "user", "content": "hi"},
		},
	}
	rec, _ := postMessages(t, s.Handler(), originalBody)
	if rec.Code != 200 {
		t.Fatalf("status = %d", rec.Code)
	}

	var parsed map[string]any
	if err := json.Unmarshal(gotBody, &parsed); err != nil {
		t.Fatalf("upstream body not valid JSON: %v", err)
	}
	if parsed["model"] != "claude-opus-4-6" {
		t.Errorf("model rewritten: %v", parsed["model"])
	}
	if parsed["system"] != "You are helpful." {
		t.Errorf("system prompt lost: %v", parsed["system"])
	}
	if parsed["temperature"] != 0.7 {
		t.Errorf("temperature lost: %v", parsed["temperature"])
	}
}
