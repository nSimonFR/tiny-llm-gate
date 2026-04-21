// Package auth implements authentication strategies for upstream providers.
//
// The oauth_chatgpt strategy reads and refreshes the token file used by
// https://github.com/EvanZhouDev/openai-oauth, making tiny-llm-gate a drop-in
// replacement for the openai-codex-proxy service: same auth.json, same
// refresh semantics, just served in-process.
package auth

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Default OAuth parameters mirroring upstream openai-oauth constants. Can be
// overridden per-provider via config.Auth.Issuer / ClientID.
const (
	DefaultIssuer             = "https://auth.openai.com"
	DefaultClientID           = "app_EMoamEEZ73f0CkXaXp7hrann"
	refreshExpiryMarginMS     = 5 * 60 * 1000    // refresh when < 5 min to exp
	passiveRefreshIntervalSec = 55 * 60          // or every 55 min regardless
	httpTimeout               = 10 * time.Second // refresh POST timeout
)

// Authenticator is applied to every outbound request to add authentication
// headers. Implementations must be safe for concurrent use.
type Authenticator interface {
	Apply(ctx context.Context, req *http.Request) error
}

// Bearer sets `Authorization: Bearer <token>`.
type Bearer struct{ Token string }

func (b Bearer) Apply(_ context.Context, req *http.Request) error {
	if b.Token != "" {
		req.Header.Set("Authorization", "Bearer "+b.Token)
	}
	return nil
}

// ChatGPTOAuth manages an auth.json file written by openai-oauth, refreshing
// tokens as they approach expiry and persisting new tokens atomically.
type ChatGPTOAuth struct {
	filePath string
	issuer   string
	clientID string
	client   *http.Client

	mu          sync.RWMutex
	accessToken string
	refreshTok  string
	idToken     string
	accountID   string
	lastRefresh time.Time
}

// NewChatGPTOAuth reads the initial token state from filePath. Returns an
// error if the file is missing or malformed.
func NewChatGPTOAuth(filePath, issuer, clientID string) (*ChatGPTOAuth, error) {
	if issuer == "" {
		issuer = DefaultIssuer
	}
	if clientID == "" {
		clientID = DefaultClientID
	}
	c := &ChatGPTOAuth{
		filePath: filePath,
		issuer:   issuer,
		clientID: clientID,
		client:   &http.Client{Timeout: httpTimeout},
	}
	if err := c.load(); err != nil {
		return nil, err
	}
	return c, nil
}

// Apply implements Authenticator. Triggers a synchronous refresh if the
// current access token is stale or missing.
func (c *ChatGPTOAuth) Apply(ctx context.Context, req *http.Request) error {
	if err := c.ensureFresh(ctx); err != nil {
		return fmt.Errorf("oauth refresh: %w", err)
	}
	c.mu.RLock()
	tok := c.accessToken
	c.mu.RUnlock()
	if tok == "" {
		return errors.New("oauth: no access token available")
	}
	req.Header.Set("Authorization", "Bearer "+tok)
	return nil
}

// authJSON mirrors the on-disk layout written by openai-oauth.
type authJSON struct {
	Tokens      authTokens `json:"tokens"`
	LastRefresh string     `json:"last_refresh"`
}

type authTokens struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	IDToken      string `json:"id_token,omitempty"`
	AccountID    string `json:"account_id,omitempty"`
}

func (c *ChatGPTOAuth) load() error {
	data, err := os.ReadFile(c.filePath)
	if err != nil {
		return fmt.Errorf("read %s: %w", c.filePath, err)
	}
	var j authJSON
	if err := json.Unmarshal(data, &j); err != nil {
		return fmt.Errorf("parse %s: %w", c.filePath, err)
	}
	last, _ := time.Parse(time.RFC3339Nano, j.LastRefresh)
	c.mu.Lock()
	c.accessToken = j.Tokens.AccessToken
	c.refreshTok = j.Tokens.RefreshToken
	c.idToken = j.Tokens.IDToken
	c.accountID = j.Tokens.AccountID
	c.lastRefresh = last
	c.mu.Unlock()
	return nil
}

// ensureFresh refreshes the access token if it is about to expire. Safe for
// concurrent callers — only one refresh runs at a time per instance.
func (c *ChatGPTOAuth) ensureFresh(ctx context.Context) error {
	c.mu.RLock()
	access := c.accessToken
	refresh := c.refreshTok
	last := c.lastRefresh
	c.mu.RUnlock()

	if !needsRefresh(access, last, time.Now()) {
		return nil
	}
	if refresh == "" {
		return errors.New("oauth: refresh token missing")
	}

	// Serialize refreshes via a single-flight lock. Re-check under lock to
	// avoid thundering-herd refreshes.
	c.mu.Lock()
	defer c.mu.Unlock()
	if !needsRefresh(c.accessToken, c.lastRefresh, time.Now()) {
		return nil
	}
	newTokens, err := c.refresh(ctx, c.refreshTok)
	if err != nil {
		return err
	}
	c.accessToken = newTokens.AccessToken
	if newTokens.RefreshToken != "" {
		c.refreshTok = newTokens.RefreshToken
	}
	if newTokens.IDToken != "" {
		c.idToken = newTokens.IDToken
	}
	c.lastRefresh = time.Now()
	return c.persistLocked()
}

func needsRefresh(accessToken string, lastRefresh, now time.Time) bool {
	if accessToken == "" {
		return true
	}
	if exp, ok := jwtExpiry(accessToken); ok {
		if exp.UnixMilli() <= now.UnixMilli()+int64(refreshExpiryMarginMS) {
			return true
		}
	} else if !lastRefresh.IsZero() && now.Sub(lastRefresh).Seconds() > float64(passiveRefreshIntervalSec) {
		// JWT couldn't be parsed — fall back to a time-based refresh.
		return true
	}
	return false
}

// jwtExpiry extracts `exp` from an unverified JWT. We never validate the
// signature — we only care whether we have a still-usable access token; the
// upstream will reject if we guess wrong.
func jwtExpiry(token string) (time.Time, bool) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return time.Time{}, false
	}
	payload, err := base64.RawURLEncoding.DecodeString(padB64(parts[1]))
	if err != nil {
		return time.Time{}, false
	}
	var claims struct {
		Exp int64 `json:"exp"`
	}
	if err := json.Unmarshal(payload, &claims); err != nil {
		return time.Time{}, false
	}
	if claims.Exp == 0 {
		return time.Time{}, false
	}
	return time.Unix(claims.Exp, 0), true
}

func padB64(s string) string {
	if pad := len(s) % 4; pad != 0 {
		return s + strings.Repeat("=", 4-pad)
	}
	return s
}

type refreshResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token,omitempty"`
	IDToken      string `json:"id_token,omitempty"`
	TokenType    string `json:"token_type,omitempty"`
	ExpiresIn    int    `json:"expires_in,omitempty"`
}

func (c *ChatGPTOAuth) refresh(ctx context.Context, refreshToken string) (*refreshResponse, error) {
	tokenURL := strings.TrimRight(c.issuer, "/") + "/oauth/token"

	form := url.Values{}
	form.Set("grant_type", "refresh_token")
	form.Set("client_id", c.clientID)
	form.Set("refresh_token", refreshToken)
	// openai-oauth also sends this scope; mirror it.
	form.Set("scope", "openid profile email offline_access")

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("oauth refresh status %d: %s", resp.StatusCode, truncate(string(body), 200))
	}
	var out refreshResponse
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, fmt.Errorf("oauth refresh json: %w", err)
	}
	if out.AccessToken == "" {
		return nil, errors.New("oauth refresh: no access_token in response")
	}
	return &out, nil
}

// persistLocked writes the current tokens back to disk atomically. Caller
// must hold c.mu in write mode.
func (c *ChatGPTOAuth) persistLocked() error {
	j := authJSON{
		Tokens: authTokens{
			AccessToken:  c.accessToken,
			RefreshToken: c.refreshTok,
			IDToken:      c.idToken,
			AccountID:    c.accountID,
		},
		LastRefresh: c.lastRefresh.UTC().Format(time.RFC3339Nano),
	}
	data, err := json.MarshalIndent(j, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Dir(c.filePath)
	tmp, err := os.CreateTemp(dir, ".auth-*.json")
	if err != nil {
		return fmt.Errorf("create temp: %w", err)
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName) // ignored if Rename succeeds

	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("write temp: %w", err)
	}
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return fmt.Errorf("chmod temp: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close temp: %w", err)
	}
	if err := os.Rename(tmpName, c.filePath); err != nil {
		return fmt.Errorf("rename temp: %w", err)
	}
	return nil
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}
