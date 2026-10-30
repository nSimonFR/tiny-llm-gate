package server

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/auth"
)

// The /v0/management routes speak the subset of CLIProxyAPI's management API
// that usage hubs such as T3 Code read: list the subscription accounts, and
// relay a request upstream with one account's credentials. The relay is how
// quota is read without the OAuth tokens ever leaving the gate.

// managementHosts are the only upstreams api-call may reach: the Anthropic
// and ChatGPT usage/credit endpoints. Anything else would turn the relay into
// an open proxy carrying the accounts' tokens.
var managementHosts = map[string]bool{
	"api.anthropic.com": true,
	"chatgpt.com":       true,
}

const (
	managementTimeout = 15 * time.Second
	managementMaxBody = 1 << 20
	// tokenPlaceholder marks header values the hub expects the gate to fill
	// with the account's token. The account's authenticator sets them instead.
	tokenPlaceholder = "$TOKEN$"
)

// managedAccount is one subscription account exposed on the management API.
type managedAccount struct {
	index    string // stable auth_index, "<provider>:<name>"
	provider string // "claude" or "codex"
	email    string
	auth     auth.Authenticator
	// claude is set for Anthropic pool accounts, so reset-quota can lift
	// the gate's own cooldown.
	claude *anthropicAccount
	// chatGPTAccountID and plan come from the Codex token claims.
	chatGPTAccountID string
	plan             string
}

// managedAccounts lists the Anthropic pool and every Codex provider whose
// OAuth authenticator built, sorted by auth_index for a stable order.
func (s *Server) managedAccounts() []managedAccount {
	var out []managedAccount
	for _, a := range s.anthropicAccounts {
		if a.auth == nil {
			continue
		}
		out = append(out, managedAccount{
			index: "claude:" + a.name, provider: "claude", email: a.email, auth: a.auth, claude: a,
		})
	}
	for name, a := range s.auths {
		oauth, ok := a.(*auth.ChatGPTOAuth)
		if !ok {
			continue
		}
		acct, email, plan := oauth.Identity()
		out = append(out, managedAccount{
			index: "codex:" + name, provider: "codex", email: email, auth: oauth,
			chatGPTAccountID: acct, plan: plan,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].index < out[j].index })
	return out
}

func (s *Server) managedAccount(index string) (managedAccount, bool) {
	for _, a := range s.managedAccounts() {
		if a.index == index {
			return a, true
		}
	}
	return managedAccount{}, false
}

// management requires the management key as a bearer token.
func (s *Server) management(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		got := sha256.Sum256([]byte(presentedKey(r)))
		if subtle.ConstantTimeCompare(got[:], s.managementKey[:]) != 1 {
			writeJSONError(w, http.StatusUnauthorized, "missing or invalid management key")
			return
		}
		next(w, r)
	}
}

type authFileIDToken struct {
	ChatGPTAccountID string `json:"chatgpt_account_id,omitempty"`
	ChatGPTPlanType  string `json:"chatgpt_plan_type,omitempty"`
}

type authFile struct {
	ID        string           `json:"id"`
	AuthIndex string           `json:"auth_index"`
	Provider  string           `json:"provider"`
	Email     string           `json:"email,omitempty"`
	Disabled  bool             `json:"disabled"`
	IDToken   *authFileIDToken `json:"id_token,omitempty"`
}

func (s *Server) handleAuthFiles(w http.ResponseWriter, _ *http.Request) {
	files := []authFile{}
	for _, a := range s.managedAccounts() {
		f := authFile{ID: a.index, AuthIndex: a.index, Provider: a.provider, Email: a.email}
		if a.chatGPTAccountID != "" || a.plan != "" {
			f.IDToken = &authFileIDToken{ChatGPTAccountID: a.chatGPTAccountID, ChatGPTPlanType: a.plan}
		}
		files = append(files, f)
	}
	writeJSON(w, http.StatusOK, map[string]any{"files": files})
}

type apiCallRequest struct {
	AuthIndex string            `json:"auth_index"`
	Method    string            `json:"method"`
	URL       string            `json:"url"`
	Header    map[string]string `json:"header"`
	Data      string            `json:"data"`
}

// handleAPICall relays one request upstream with an account's credentials
// and returns the upstream status and body. An upstream error status is a
// successful relay: the hub reads status_code.
func (s *Server) handleAPICall(w http.ResponseWriter, r *http.Request) {
	var in apiCallRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, managementMaxBody)).Decode(&in); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	acct, ok := s.managedAccount(in.AuthIndex)
	if !ok {
		writeJSONError(w, http.StatusNotFound, "unknown auth_index")
		return
	}
	target, err := url.Parse(in.URL)
	if err != nil || target.Scheme != "https" || !managementHosts[target.Hostname()] || target.Port() != "" {
		writeJSONError(w, http.StatusForbidden, "url is not an allowed usage endpoint")
		return
	}
	method := strings.ToUpper(in.Method)
	if method != http.MethodGet && method != http.MethodPost {
		writeJSONError(w, http.StatusBadRequest, "method must be GET or POST")
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), managementTimeout)
	defer cancel()
	var body io.Reader
	if in.Data != "" {
		body = strings.NewReader(in.Data)
	}
	req, err := http.NewRequestWithContext(ctx, method, target.String(), body)
	if err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	for k, v := range in.Header {
		if strings.Contains(v, tokenPlaceholder) {
			continue
		}
		req.Header.Set(k, v)
	}
	if err := acct.auth.Apply(ctx, req); err != nil {
		s.logger.Warn("management: account auth failed", "account", acct.index, "err", err)
		writeJSONError(w, http.StatusBadGateway, "account credentials unavailable")
		return
	}
	resp, err := s.client.Do(req)
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, "upstream request failed")
		return
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, managementMaxBody))
	if err != nil {
		writeJSONError(w, http.StatusBadGateway, "upstream response failed")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status_code": resp.StatusCode, "body": string(data)})
}

// handleResetQuota lifts the gate's cooldown on an account, called after a
// hub redeems a reset credit so routing uses the account again right away.
// Codex providers keep no cooldown, so they only acknowledge.
func (s *Server) handleResetQuota(w http.ResponseWriter, r *http.Request) {
	var in struct {
		AuthIndex string `json:"auth_index"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, managementMaxBody)).Decode(&in); err != nil {
		writeJSONError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	acct, ok := s.managedAccount(in.AuthIndex)
	if !ok {
		writeJSONError(w, http.StatusNotFound, "unknown auth_index")
		return
	}
	if acct.claude != nil {
		acct.claude.coolingDownUntil.Store(0)
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}
