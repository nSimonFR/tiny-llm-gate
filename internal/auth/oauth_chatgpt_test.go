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

// makeJWT builds an unsigned JWT with the given claims payload.
func makeJWT(t *testing.T, claims map[string]any) string {
	t.Helper()
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none","typ":"JWT"}`))
	pj, _ := json.Marshal(claims)
	payload := base64.RawURLEncoding.EncodeToString(pj)
	return header + "." + payload + ".sig"
}

func writeCreds(t *testing.T, dir string, c map[string]any) string {
	t.Helper()
	p := filepath.Join(dir, "creds.json")
	b, _ := json.Marshal(c)
	if err := os.WriteFile(p, b, 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestChatGPTOAuth_RefreshesExpiredAndSetsHeaders(t *testing.T) {
	dir := t.TempDir()
	// Expired access token forces a refresh on first Apply.
	expired := makeJWT(t, map[string]any{"exp": time.Now().Add(-time.Hour).Unix()})
	credsPath := writeCreds(t, dir, map[string]any{
		"access_token":  expired,
		"refresh_token": "rt_original",
	})

	// Mock token endpoint returns a fresh token carrying the account-id claim.
	fresh := makeJWT(t, map[string]any{
		"exp": time.Now().Add(time.Hour).Unix(),
		"https://api.openai.com/auth": map[string]any{
			"chatgpt_account_id": "acct_123",
		},
	})
	var gotGrant, gotRefresh string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseForm()
		gotGrant = r.Form.Get("grant_type")
		gotRefresh = r.Form.Get("refresh_token")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"access_token":  fresh,
			"refresh_token": "rt_rotated",
		})
	}))
	defer srv.Close()

	oauth, err := NewChatGPTOAuth(credsPath, srv.URL, "client_x", nil)
	if err != nil {
		t.Fatal(err)
	}
	req, _ := http.NewRequest(http.MethodPost, "http://backend/", nil)
	if err := oauth.Apply(context.Background(), req); err != nil {
		t.Fatal(err)
	}

	if gotGrant != "refresh_token" || gotRefresh != "rt_original" {
		t.Fatalf("refresh call grant=%q refresh=%q", gotGrant, gotRefresh)
	}
	if got := req.Header.Get("Authorization"); got != "Bearer "+fresh {
		t.Fatalf("Authorization = %q", got)
	}
	if got := req.Header.Get("ChatGPT-Account-Id"); got != "acct_123" {
		t.Fatalf("ChatGPT-Account-Id = %q", got)
	}

	// Rotated refresh token must be persisted back to disk.
	b, _ := os.ReadFile(credsPath)
	var persisted map[string]any
	_ = json.Unmarshal(b, &persisted)
	if persisted["refresh_token"] != "rt_rotated" {
		t.Fatalf("persisted refresh_token = %v", persisted["refresh_token"])
	}
}

func TestChatGPTOAuth_NoRefreshWhenFresh(t *testing.T) {
	dir := t.TempDir()
	fresh := makeJWT(t, map[string]any{
		"exp":                         time.Now().Add(time.Hour).Unix(),
		"https://api.openai.com/auth": map[string]any{"chatgpt_account_id": "acct_9"},
	})
	credsPath := writeCreds(t, dir, map[string]any{
		"access_token":  fresh,
		"refresh_token": "rt",
	})
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
	}))
	defer srv.Close()

	oauth, err := NewChatGPTOAuth(credsPath, srv.URL, "c", nil)
	if err != nil {
		t.Fatal(err)
	}
	req, _ := http.NewRequest(http.MethodPost, "http://x/", nil)
	if err := oauth.Apply(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if called {
		t.Fatal("should not refresh a still-valid token")
	}
	if req.Header.Get("ChatGPT-Account-Id") != "acct_9" {
		t.Fatalf("account id = %q", req.Header.Get("ChatGPT-Account-Id"))
	}
}

func TestChatGPTOAuth_NestedTokensLayout(t *testing.T) {
	dir := t.TempDir()
	fresh := makeJWT(t, map[string]any{"exp": time.Now().Add(time.Hour).Unix()})
	// Codex CLI / openai-oauth nested shape.
	p := filepath.Join(dir, "auth.json")
	b, _ := json.Marshal(map[string]any{
		"tokens": map[string]any{
			"access_token":  fresh,
			"refresh_token": "rt_nested",
		},
	})
	_ = os.WriteFile(p, b, 0o600)

	oauth, err := NewChatGPTOAuth(p, "http://unused", "c", nil)
	if err != nil {
		t.Fatalf("should load nested layout: %v", err)
	}
	if oauth.refreshTok != "rt_nested" {
		t.Fatalf("refresh token = %q", oauth.refreshTok)
	}
}

func TestChatGPTOAuth_MissingRefreshTokenFailsBuild(t *testing.T) {
	dir := t.TempDir()
	p := writeCreds(t, dir, map[string]any{"access_token": "x"})
	if _, err := NewChatGPTOAuth(p, "http://x", "c", nil); err == nil {
		t.Fatal("expected error when refresh_token is absent")
	}
}
