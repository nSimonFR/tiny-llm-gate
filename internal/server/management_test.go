package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
)

const testManagementKey = "mgmt-secret"

// fakeJWT builds an unsigned JWT; the gate never verifies signatures.
func fakeJWT(t *testing.T, claims map[string]any) string {
	t.Helper()
	payload, err := json.Marshal(claims)
	if err != nil {
		t.Fatal(err)
	}
	enc := base64.RawURLEncoding.EncodeToString
	return enc([]byte(`{"alg":"none"}`)) + "." + enc(payload) + ".sig"
}

// redirectTransport sends every request to target, keeping the path, so the
// relay can be tested against the real allowlisted hosts.
type redirectTransport struct{ target *url.URL }

func (rt redirectTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	r = r.Clone(r.Context())
	r.Header.Set("X-Original-Host", r.URL.Host)
	r.URL.Scheme, r.URL.Host = rt.target.Scheme, rt.target.Host
	return http.DefaultTransport.RoundTrip(r)
}

func buildManagementServer(t *testing.T, upstream *httptest.Server) *Server {
	t.Helper()
	access := fakeJWT(t, map[string]any{
		"exp": time.Now().Add(time.Hour).Unix(),
		"https://api.openai.com/auth": map[string]any{
			"chatgpt_account_id": "acct-123",
			"chatgpt_plan_type":  "pro",
		},
	})
	idToken := fakeJWT(t, map[string]any{"email": "codex@example.com"})
	creds := filepath.Join(t.TempDir(), "codex.json")
	data, _ := json.Marshal(map[string]string{
		"access_token": access, "refresh_token": "refresh", "id_token": idToken,
	})
	if err := os.WriteFile(creds, data, 0o600); err != nil {
		t.Fatal(err)
	}
	cfg := &config.Config{
		Listen: "127.0.0.1:0",
		Providers: map[string]config.Provider{
			"chatgpt": {Type: "codex", BaseURL: "https://chatgpt.com/backend-api/codex",
				Auth: &config.Auth{Type: "oauth_chatgpt", File: creds}},
		},
		Models: map[string]config.Model{
			"gpt": {Provider: "chatgpt", UpstreamModel: "gpt-5"},
		},
		Anthropic: &config.Anthropic{
			Upstream: "https://api.anthropic.com",
			Accounts: []config.AnthropicAccount{
				{Name: "work", Email: "work@example.com", Auth: &config.Auth{Type: "bearer", Token: "claude-work-token"}},
			},
		},
		Management: &config.Management{Key: testManagementKey},
	}
	s, err := New(cfg, discardLogger())
	if err != nil {
		t.Fatalf("server.New: %v", err)
	}
	if upstream != nil {
		target, _ := url.Parse(upstream.URL)
		s.client = &http.Client{Transport: redirectTransport{target: target}}
	}
	return s
}

func managementRequest(t *testing.T, h http.Handler, method, path, key string, body any) *httptest.ResponseRecorder {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		_ = json.NewEncoder(&buf).Encode(body)
	}
	req := httptest.NewRequest(method, path, &buf)
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestManagementRequiresKey(t *testing.T) {
	h := buildManagementServer(t, nil).Handler()
	if rec := managementRequest(t, h, http.MethodGet, "/v0/management/auth-files", "wrong", nil); rec.Code != http.StatusUnauthorized {
		t.Fatalf("wrong key: status %d", rec.Code)
	}
	if rec := managementRequest(t, h, http.MethodGet, "/v0/management/auth-files", "", nil); rec.Code != http.StatusUnauthorized {
		t.Fatalf("no key: status %d", rec.Code)
	}
}

func TestManagementRoutesAbsentWithoutConfig(t *testing.T) {
	h := buildAnthropicServer(t, "https://api.anthropic.com", "tok").Handler()
	if rec := managementRequest(t, h, http.MethodGet, "/v0/management/auth-files", testManagementKey, nil); rec.Code != http.StatusNotFound {
		t.Fatalf("status %d, want 404", rec.Code)
	}
}

func TestManagementAuthFilesListsClaudeAndCodex(t *testing.T) {
	h := buildManagementServer(t, nil).Handler()
	rec := managementRequest(t, h, http.MethodGet, "/v0/management/auth-files", testManagementKey, nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var got struct{ Files []authFile }
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if len(got.Files) != 2 {
		t.Fatalf("files = %+v", got.Files)
	}
	claude, codex := got.Files[0], got.Files[1]
	if claude.AuthIndex != "claude:work" || claude.Provider != "claude" || claude.Email != "work@example.com" {
		t.Errorf("claude = %+v", claude)
	}
	if codex.AuthIndex != "codex:chatgpt" || codex.Provider != "codex" || codex.Email != "codex@example.com" ||
		codex.IDToken == nil || codex.IDToken.ChatGPTAccountID != "acct-123" || codex.IDToken.ChatGPTPlanType != "pro" {
		t.Errorf("codex = %+v (id_token %+v)", codex, codex.IDToken)
	}
}

func TestManagementAPICallAppliesAccountAuth(t *testing.T) {
	var gotAuth, gotBeta, gotHost, gotAccount string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth, gotBeta = r.Header.Get("Authorization"), r.Header.Get("anthropic-beta")
		gotHost, gotAccount = r.Header.Get("X-Original-Host"), r.Header.Get("ChatGPT-Account-Id")
		w.WriteHeader(http.StatusTeapot)
		_, _ = w.Write([]byte(`{"five_hour":null}`))
	}))
	defer upstream.Close()
	h := buildManagementServer(t, upstream).Handler()

	rec := managementRequest(t, h, http.MethodPost, "/v0/management/api-call", testManagementKey, map[string]any{
		"auth_index": "claude:work",
		"method":     "GET",
		"url":        "https://api.anthropic.com/api/oauth/usage",
		"header":     map[string]string{"Authorization": "Bearer $TOKEN$", "anthropic-beta": "oauth-2025-04-20"},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	var got struct {
		StatusCode int    `json:"status_code"`
		Body       string `json:"body"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &got)
	if got.StatusCode != http.StatusTeapot || got.Body != `{"five_hour":null}` {
		t.Errorf("relay = %+v", got)
	}
	if gotAuth != "Bearer claude-work-token" || gotBeta != "oauth-2025-04-20" || gotHost != "api.anthropic.com" {
		t.Errorf("upstream saw auth=%q beta=%q host=%q", gotAuth, gotBeta, gotHost)
	}

	rec = managementRequest(t, h, http.MethodPost, "/v0/management/api-call", testManagementKey, map[string]any{
		"auth_index": "codex:chatgpt",
		"method":     "GET",
		"url":        "https://chatgpt.com/backend-api/wham/usage",
		"header":     map[string]string{"Authorization": "Bearer $TOKEN$"},
	})
	if rec.Code != http.StatusOK || gotAccount != "acct-123" || gotAuth == "Bearer $TOKEN$" || gotAuth == "" {
		t.Errorf("codex relay: status %d, auth=%q account=%q", rec.Code, gotAuth, gotAccount)
	}
}

func TestManagementAPICallRejectsOtherHosts(t *testing.T) {
	h := buildManagementServer(t, nil).Handler()
	for _, target := range []string{
		"https://evil.example.com/steal",
		"http://api.anthropic.com/api/oauth/usage",
		"https://api.anthropic.com:8443/api/oauth/usage",
	} {
		rec := managementRequest(t, h, http.MethodPost, "/v0/management/api-call", testManagementKey, map[string]any{
			"auth_index": "claude:work", "method": "GET", "url": target,
		})
		if rec.Code != http.StatusForbidden {
			t.Errorf("%s: status %d, want 403", target, rec.Code)
		}
	}
	rec := managementRequest(t, h, http.MethodPost, "/v0/management/api-call", testManagementKey, map[string]any{
		"auth_index": "claude:nope", "method": "GET", "url": "https://api.anthropic.com/api/oauth/usage",
	})
	if rec.Code != http.StatusNotFound {
		t.Errorf("unknown account: status %d", rec.Code)
	}
}

func TestManagementResetQuotaClearsCooldown(t *testing.T) {
	s := buildManagementServer(t, nil)
	s.anthropicAccounts[0].cooldown(time.Hour)
	rec := managementRequest(t, s.Handler(), http.MethodPost, "/v0/management/reset-quota", testManagementKey,
		map[string]any{"auth_index": "claude:work"})
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}
	if !s.anthropicAccounts[0].available(time.Now()) {
		t.Error("account still cooling down")
	}
}

func TestManagementAuthFilesLooksUpClaudeEmail(t *testing.T) {
	var calls int
	status := http.StatusTooManyRequests
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != profilePath || r.Header.Get("Authorization") != "Bearer claude-work-token" {
			t.Errorf("unexpected request %s auth=%q", r.URL.Path, r.Header.Get("Authorization"))
		}
		calls++
		w.WriteHeader(status)
		_, _ = w.Write([]byte(`{"account":{"email":"profile@example.com"}}`))
	}))
	defer upstream.Close()
	s := buildManagementServer(t, upstream)
	s.anthropicAccounts[0].email = ""
	h := s.Handler()

	claudeEmail := func() string {
		rec := managementRequest(t, h, http.MethodGet, "/v0/management/auth-files", testManagementKey, nil)
		var got struct{ Files []authFile }
		_ = json.Unmarshal(rec.Body.Bytes(), &got)
		return got.Files[0].Email
	}
	if got := claudeEmail(); got != "" {
		t.Errorf("rate-limited lookup: email %q, want none", got)
	}
	status = http.StatusOK
	if got := claudeEmail(); got != "profile@example.com" {
		t.Errorf("email %q after recovery", got)
	}
	if got := claudeEmail(); got != "profile@example.com" || calls != 2 {
		t.Errorf("cached email %q, %d upstream calls, want 2", got, calls)
	}
}
