package auth

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// OAuth parameters for the ChatGPT/Codex backend. These mirror the constants
// baked into the official Codex CLI and codex-proxy; the client_id in
// particular is the public Codex desktop app id (not a secret).
const (
	chatGPTDefaultIssuer   = "https://auth.openai.com"
	chatGPTDefaultClientID = "app_EMoamEEZ73f0CkXaXp7hrann"

	// Refresh when the access token is within this margin of expiry, or when
	// this much wall-clock time has elapsed since the last refresh and the JWT
	// expiry can't be read. Mirrors codex-proxy's 5-min margin / passive cycle.
	chatGPTRefreshMargin  = 5 * time.Minute
	chatGPTPassiveMaxAge  = 55 * time.Minute
	chatGPTRefreshTimeout = 15 * time.Second
)

// ChatGPTOAuth authenticates upstream requests to the ChatGPT/Codex backend
// using an OAuth access token that it refreshes on demand against
// auth.openai.com. The refresh token lineage is owned solely by this process
// (single owner) — no external CLI or proxy shares it, so there is no
// rotate-each-other-out war.
//
// Apply sets Authorization: Bearer <access token> and, when derivable,
// ChatGPT-Account-Id (extracted from the access token's JWT claims). The Codex
// backend returns HTML 403 login pages instead of JSON without the latter.
//
// Every refresh — attempt, success, and failure — is logged, so a silent
// refresh failure that would otherwise only surface as downstream 401s is
// visible in the gate's own logs.
type ChatGPTOAuth struct {
	filePath string
	issuer   string
	clientID string
	client   *http.Client
	logger   *slog.Logger

	mu          sync.RWMutex
	accessToken string
	refreshTok  string
	idToken     string
	accountID   string
	lastRefresh time.Time
}

// NewChatGPTOAuth reads the initial token state from filePath. Returns an
// error if the file is missing or malformed — this fails startup fast on a
// misconfigured credentials path.
func NewChatGPTOAuth(filePath, issuer, clientID string, logger *slog.Logger) (*ChatGPTOAuth, error) {
	if issuer == "" {
		issuer = chatGPTDefaultIssuer
	}
	if clientID == "" {
		clientID = chatGPTDefaultClientID
	}
	if logger == nil {
		logger = slog.Default()
	}
	c := &ChatGPTOAuth{
		filePath: filePath,
		issuer:   issuer,
		clientID: clientID,
		client:   &http.Client{Timeout: chatGPTRefreshTimeout},
		logger:   logger,
	}
	if err := c.load(); err != nil {
		return nil, err
	}
	if c.refreshTok == "" {
		return nil, fmt.Errorf("oauth_chatgpt: %s has no refresh_token", filePath)
	}
	return c, nil
}

// Apply implements Authenticator. Refreshes synchronously when the access
// token is stale, then sets the bearer + account-id headers.
func (c *ChatGPTOAuth) Apply(ctx context.Context, req *http.Request) error {
	if err := c.ensureFresh(ctx); err != nil {
		return fmt.Errorf("oauth_chatgpt refresh: %w", err)
	}
	c.mu.RLock()
	tok := c.accessToken
	acct := c.accountID
	c.mu.RUnlock()
	if tok == "" {
		return errors.New("oauth_chatgpt: no access token available")
	}
	req.Header.Set("Authorization", "Bearer "+tok)
	if acct == "" {
		acct = accountIDFromJWT(tok)
	}
	if acct != "" {
		req.Header.Set("ChatGPT-Account-Id", acct)
	}
	return nil
}

// credentialFile is the on-disk credential layout. It accepts both a flat
// shape ({access_token, refresh_token, ...}) and the nested shape written by
// the Codex CLI / openai-oauth ({tokens:{access_token, ...}, last_refresh}),
// so tokens can be seeded from either source without reshaping.
type credentialFile struct {
	AccessToken  string          `json:"access_token"`
	RefreshToken string          `json:"refresh_token"`
	IDToken      string          `json:"id_token"`
	AccountID    string          `json:"account_id"`
	LastRefresh  string          `json:"last_refresh"`
	Tokens       *credentialFile `json:"tokens"`
}

func (c *ChatGPTOAuth) load() error {
	data, err := os.ReadFile(c.filePath)
	if err != nil {
		return fmt.Errorf("oauth_chatgpt: read %s: %w", c.filePath, err)
	}
	var f credentialFile
	if err := json.Unmarshal(data, &f); err != nil {
		return fmt.Errorf("oauth_chatgpt: parse %s: %w", c.filePath, err)
	}
	// Prefer the nested "tokens" block when present (Codex CLI layout), but
	// fall back to top-level fields for the flat shape.
	src := f
	if f.Tokens != nil && f.Tokens.AccessToken != "" {
		src = *f.Tokens
		if f.LastRefresh != "" {
			src.LastRefresh = f.LastRefresh
		}
	}
	last, _ := time.Parse(time.RFC3339Nano, src.LastRefresh)

	c.mu.Lock()
	c.accessToken = src.AccessToken
	c.refreshTok = src.RefreshToken
	c.idToken = src.IDToken
	c.accountID = src.AccountID
	c.lastRefresh = last
	c.mu.Unlock()
	return nil
}

// ensureFresh refreshes the access token if it is close to expiry. Safe for
// concurrent callers: a single-flight lock serializes refreshes and re-checks
// under the lock to avoid a thundering herd.
func (c *ChatGPTOAuth) ensureFresh(ctx context.Context) error {
	c.mu.RLock()
	access := c.accessToken
	last := c.lastRefresh
	c.mu.RUnlock()

	if !chatGPTNeedsRefresh(access, last, time.Now()) {
		return nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if !chatGPTNeedsRefresh(c.accessToken, c.lastRefresh, time.Now()) {
		return nil
	}
	if c.refreshTok == "" {
		return errors.New("oauth_chatgpt: refresh token missing")
	}

	c.logger.Info("oauth_chatgpt: refreshing access token", "file", c.filePath)
	newTokens, err := c.refresh(ctx, c.refreshTok)
	if err != nil {
		// A refresh failure is the exact silent-death mode that used to only
		// surface as downstream 401s — log it loudly at Error.
		c.logger.Error("oauth_chatgpt: token refresh FAILED", "file", c.filePath, "err", err)
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
	if acct := accountIDFromJWT(c.accessToken); acct != "" {
		c.accountID = acct
	}
	if err := c.persistLocked(); err != nil {
		// Non-fatal: the in-memory token is valid; persistence only matters
		// across restarts. Log so a broken creds path is visible.
		c.logger.Warn("oauth_chatgpt: persist refreshed token failed", "file", c.filePath, "err", err)
	}
	c.logger.Info("oauth_chatgpt: token refreshed", "file", c.filePath)
	return nil
}

func chatGPTNeedsRefresh(accessToken string, lastRefresh, now time.Time) bool {
	if accessToken == "" {
		return true
	}
	if exp, ok := jwtExpiry(accessToken); ok {
		return exp.Add(-chatGPTRefreshMargin).Before(now)
	}
	// JWT unreadable — fall back to a time-based passive refresh.
	if !lastRefresh.IsZero() && now.Sub(lastRefresh) > chatGPTPassiveMaxAge {
		return true
	}
	return false
}

type chatGPTRefreshResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	IDToken      string `json:"id_token"`
}

func (c *ChatGPTOAuth) refresh(ctx context.Context, refreshToken string) (*chatGPTRefreshResponse, error) {
	tokenURL := strings.TrimRight(c.issuer, "/") + "/oauth/token"
	form := url.Values{}
	form.Set("grant_type", "refresh_token")
	form.Set("client_id", c.clientID)
	form.Set("refresh_token", refreshToken)
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
		return nil, fmt.Errorf("status %d: %s", resp.StatusCode, chatGPTTruncate(string(body), 200))
	}
	var out chatGPTRefreshResponse
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	if out.AccessToken == "" {
		return nil, errors.New("no access_token in refresh response")
	}
	return &out, nil
}

// persistLocked atomically writes the current tokens back to the credentials
// file in the flat shape. Caller must hold c.mu for writing.
func (c *ChatGPTOAuth) persistLocked() error {
	out := credentialFile{
		AccessToken:  c.accessToken,
		RefreshToken: c.refreshTok,
		IDToken:      c.idToken,
		AccountID:    c.accountID,
		LastRefresh:  c.lastRefresh.UTC().Format(time.RFC3339Nano),
	}
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}
	dir := filepath.Dir(c.filePath)
	tmp, err := os.CreateTemp(dir, ".codex-oauth-*.json")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, c.filePath)
}

// accountIDFromJWT extracts the chatgpt_account_id claim from an unverified
// access-token JWT. The claim lives under the namespaced
// "https://api.openai.com/auth" object. Signature is never validated — the
// upstream rejects a bad token anyway.
func accountIDFromJWT(token string) string {
	claims := jwtClaims(token)
	if claims == nil {
		return ""
	}
	authClaim, ok := claims["https://api.openai.com/auth"].(map[string]any)
	if !ok {
		return ""
	}
	if id, ok := authClaim["chatgpt_account_id"].(string); ok {
		return id
	}
	return ""
}

func jwtExpiry(token string) (time.Time, bool) {
	claims := jwtClaims(token)
	if claims == nil {
		return time.Time{}, false
	}
	exp, ok := claims["exp"].(float64)
	if !ok || exp == 0 {
		return time.Time{}, false
	}
	return time.Unix(int64(exp), 0), true
}

func jwtClaims(token string) map[string]any {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return nil
	}
	// JWT segments are unpadded base64url per RFC 7519. Fall back to the
	// padded/standard alphabet for lenient producers.
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		payload, err = base64.URLEncoding.DecodeString(parts[1])
		if err != nil {
			return nil
		}
	}
	var claims map[string]any
	if err := json.Unmarshal(payload, &claims); err != nil {
		return nil
	}
	return claims
}

func chatGPTTruncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}
