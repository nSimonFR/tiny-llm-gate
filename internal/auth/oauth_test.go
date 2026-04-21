package auth

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// makeJWT builds a token with only the fields we decode (exp) set. The
// signature is garbage — we never validate it.
func makeJWT(t *testing.T, expOffsetSec int64) string {
	t.Helper()
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none"}`))
	payload, _ := json.Marshal(map[string]int64{"exp": time.Now().Unix() + expOffsetSec})
	body := base64.RawURLEncoding.EncodeToString(payload)
	return header + "." + body + ".sig"
}

func writeAuth(t *testing.T, dir string, access, refresh string, last time.Time) string {
	t.Helper()
	p := filepath.Join(dir, "auth.json")
	data, _ := json.MarshalIndent(authJSON{
		Tokens:      authTokens{AccessToken: access, RefreshToken: refresh},
		LastRefresh: last.UTC().Format(time.RFC3339Nano),
	}, "", "  ")
	if err := os.WriteFile(p, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestBearerApply(t *testing.T) {
	req, _ := http.NewRequest(http.MethodGet, "http://x/", nil)
	if err := (Bearer{Token: "abc"}).Apply(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if got := req.Header.Get("Authorization"); got != "Bearer abc" {
		t.Errorf("Authorization = %q", got)
	}
}

func TestBearerEmptyTokenNoop(t *testing.T) {
	req, _ := http.NewRequest(http.MethodGet, "http://x/", nil)
	if err := (Bearer{Token: ""}).Apply(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if got := req.Header.Get("Authorization"); got != "" {
		t.Errorf("expected no auth header, got %q", got)
	}
}

func TestChatGPTLoadsFile(t *testing.T) {
	dir := t.TempDir()
	p := writeAuth(t, dir, makeJWT(t, 3600), "rt", time.Now())
	o, err := NewChatGPTOAuth(p, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if o.refreshTok != "rt" {
		t.Errorf("refreshTok = %q", o.refreshTok)
	}
	if o.issuer != DefaultIssuer {
		t.Errorf("default issuer not applied")
	}
	if o.clientID != DefaultClientID {
		t.Errorf("default client_id not applied")
	}
}

func TestChatGPTSetsAccountIDHeader(t *testing.T) {
	dir := t.TempDir()
	access := makeJWT(t, 3600)
	p := filepath.Join(dir, "auth.json")
	data, _ := json.MarshalIndent(authJSON{
		Tokens: authTokens{
			AccessToken:  access,
			RefreshToken: "rt",
			AccountID:    "acct-123",
		},
		LastRefresh: time.Now().UTC().Format(time.RFC3339Nano),
	}, "", "  ")
	_ = os.WriteFile(p, data, 0o600)

	o, err := NewChatGPTOAuth(p, "", "")
	if err != nil {
		t.Fatal(err)
	}
	req, _ := http.NewRequest(http.MethodGet, "http://x/", nil)
	if err := o.Apply(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if got := req.Header.Get("chatgpt-account-id"); got != "acct-123" {
		t.Errorf("chatgpt-account-id = %q; want acct-123", got)
	}
	if got := req.Header.Get("Authorization"); got != "Bearer "+access {
		t.Errorf("Authorization header missing or wrong: %q", got)
	}
	if got := req.Header.Get("OpenAI-Beta"); got != "responses=experimental" {
		t.Errorf("OpenAI-Beta header missing or wrong: %q", got)
	}
}

func TestChatGPTOmitsAccountIDWhenMissing(t *testing.T) {
	dir := t.TempDir()
	p := writeAuth(t, dir, makeJWT(t, 3600), "rt", time.Now())
	o, err := NewChatGPTOAuth(p, "", "")
	if err != nil {
		t.Fatal(err)
	}
	req, _ := http.NewRequest(http.MethodGet, "http://x/", nil)
	if err := o.Apply(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if got := req.Header.Get("chatgpt-account-id"); got != "" {
		t.Errorf("expected no chatgpt-account-id header when missing, got %q", got)
	}
}

func TestChatGPTAppliesFreshToken(t *testing.T) {
	dir := t.TempDir()
	// Token expires in an hour — no refresh should fire.
	access := makeJWT(t, 3600)
	p := writeAuth(t, dir, access, "rt", time.Now())
	o, err := NewChatGPTOAuth(p, "", "")
	if err != nil {
		t.Fatal(err)
	}

	req, _ := http.NewRequest(http.MethodGet, "http://x/", nil)
	if err := o.Apply(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if got := req.Header.Get("Authorization"); got != "Bearer "+access {
		t.Errorf("Authorization = %q; expected Bearer %s", got, access)
	}
}

func TestChatGPTRefreshesExpiredToken(t *testing.T) {
	dir := t.TempDir()
	// Token expired 5 minutes ago.
	oldAccess := makeJWT(t, -300)
	newAccess := makeJWT(t, 3600)
	newRefresh := "new-rt"
	p := writeAuth(t, dir, oldAccess, "old-rt", time.Now().Add(-2*time.Hour))

	issuer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/oauth/token" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		if err := r.ParseForm(); err != nil {
			t.Fatal(err)
		}
		if r.Form.Get("grant_type") != "refresh_token" {
			t.Errorf("grant_type = %s", r.Form.Get("grant_type"))
		}
		if r.Form.Get("refresh_token") != "old-rt" {
			t.Errorf("refresh_token = %s", r.Form.Get("refresh_token"))
		}
		if r.Form.Get("client_id") != "test-client" {
			t.Errorf("client_id = %s", r.Form.Get("client_id"))
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(refreshResponse{
			AccessToken:  newAccess,
			RefreshToken: newRefresh,
			ExpiresIn:    3600,
		})
	}))
	defer issuer.Close()

	o, err := NewChatGPTOAuth(p, issuer.URL, "test-client")
	if err != nil {
		t.Fatal(err)
	}

	req, _ := http.NewRequest(http.MethodGet, "http://x/", nil)
	if err := o.Apply(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if got := req.Header.Get("Authorization"); got != "Bearer "+newAccess {
		t.Errorf("Authorization = %q; expected Bearer %s", got, newAccess)
	}

	// Verify auth.json was rewritten.
	raw, _ := os.ReadFile(p)
	var persisted authJSON
	_ = json.Unmarshal(raw, &persisted)
	if persisted.Tokens.AccessToken != newAccess {
		t.Errorf("persisted access_token not updated; got %s", persisted.Tokens.AccessToken)
	}
	if persisted.Tokens.RefreshToken != newRefresh {
		t.Errorf("persisted refresh_token not updated; got %s", persisted.Tokens.RefreshToken)
	}
}

func TestChatGPTRefreshFailureSurfacesError(t *testing.T) {
	dir := t.TempDir()
	p := writeAuth(t, dir, makeJWT(t, -300), "rt", time.Now().Add(-2*time.Hour))

	issuer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error":"invalid_grant"}`, http.StatusUnauthorized)
	}))
	defer issuer.Close()

	o, err := NewChatGPTOAuth(p, issuer.URL, "c")
	if err != nil {
		t.Fatal(err)
	}
	req, _ := http.NewRequest(http.MethodGet, "http://x/", nil)
	if err := o.Apply(context.Background(), req); err == nil {
		t.Error("expected refresh failure to surface as Apply error")
	}
}

func TestChatGPTMissingFileFails(t *testing.T) {
	if _, err := NewChatGPTOAuth("/nonexistent/auth.json", "", ""); err == nil {
		t.Error("expected error for missing file")
	}
}

func TestNeedsRefreshTimings(t *testing.T) {
	now := time.Now()
	cases := []struct {
		name  string
		token string
		last  time.Time
		want  bool
	}{
		{"empty access", "", now, true},
		{"far-future exp", makeJWT(t, 3600), now, false},
		{"past exp", makeJWT(t, -10), now, true},
		{"near exp (inside margin)", makeJWT(t, 60), now, true},
		{"bogus token, fresh last_refresh", "not.a.jwt", now, false},
		{"bogus token, stale last_refresh", "not.a.jwt", now.Add(-2 * time.Hour), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := needsRefresh(tc.token, tc.last, now); got != tc.want {
				t.Errorf("got %v want %v", got, tc.want)
			}
		})
	}
}

// smoke test: the refresh URL is built relative to issuer without double-slash.
func TestIssuerTrailingSlash(t *testing.T) {
	var got string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = r.URL.Path
		_ = json.NewEncoder(w).Encode(refreshResponse{AccessToken: makeJWT(t, 3600)})
	}))
	defer srv.Close()

	dir := t.TempDir()
	p := writeAuth(t, dir, "", "rt", time.Now().Add(-2*time.Hour))
	o, err := NewChatGPTOAuth(p, srv.URL+"/", "c")
	if err != nil {
		t.Fatal(err)
	}
	req, _ := http.NewRequest(http.MethodGet, "http://x/", nil)
	if err := o.Apply(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if got != "/oauth/token" {
		t.Errorf("refresh path = %q", got)
	}
}

