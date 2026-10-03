package server

import (
	"context"
	"crypto/sha256"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
	"github.com/nSimonFR/tiny-llm-gate/internal/resolve"
)

const ctxKeyClient ctxKey = ctxKeyRequestID + 1

type clientKey struct {
	name     string
	allowAll bool
	exact    map[string]bool
	prefixes []string
	rpm      int

	mu     sync.Mutex
	window int64
	count  int
}

// buildClientKeys indexes keys by SHA-256 so lookup never compares secrets
// byte-by-byte. Allowlist names are resolved to canonical models up front.
func buildClientKeys(cfg *config.Config, res *resolve.Resolver) (map[[32]byte]*clientKey, error) {
	out := make(map[[32]byte]*clientKey, len(cfg.ClientKeys))
	for _, k := range cfg.ClientKeys {
		secret, err := k.LoadKey()
		if err != nil {
			return nil, err
		}
		h := sha256.Sum256([]byte(secret))
		if _, dup := out[h]; dup {
			return nil, fmt.Errorf("client key %q: same secret as another key", k.Name)
		}
		ck := &clientKey{name: k.Name, rpm: k.RPM, exact: map[string]bool{}, allowAll: len(k.Models) == 0}
		for _, m := range k.Models {
			switch {
			case m == "*":
				ck.allowAll = true
			case strings.HasSuffix(m, "*"):
				ck.prefixes = append(ck.prefixes, strings.TrimSuffix(m, "*"))
			default:
				ck.exact[m] = true
				if r, err := res.Resolve(m); err == nil {
					ck.exact[r.ModelName] = true
				}
			}
		}
		out[h] = ck
	}
	return out, nil
}

// allows reports whether the key may use a model, matched by any of its
// names (client-supplied, canonical, vendor slug; empty names are skipped).
func (ck *clientKey) allows(names ...string) bool {
	if ck == nil || ck.allowAll {
		return true
	}
	for _, n := range names {
		if n == "" {
			continue
		}
		if ck.exact[n] {
			return true
		}
		for _, p := range ck.prefixes {
			if strings.HasPrefix(n, p) {
				return true
			}
		}
	}
	return false
}

// take counts one request against the per-minute window; returns the
// seconds until the window resets when over the limit.
func (ck *clientKey) take(now time.Time) (retryAfter int, ok bool) {
	if ck.rpm == 0 {
		return 0, true
	}
	ck.mu.Lock()
	defer ck.mu.Unlock()
	w := now.Unix() / 60
	if w != ck.window {
		ck.window, ck.count = w, 0
	}
	if ck.count >= ck.rpm {
		return int(60 - now.Unix()%60), false
	}
	ck.count++
	return 0, true
}

// presentedKey reads the key from any header an OpenAI, OpenRouter,
// Anthropic or Gemini SDK uses.
func presentedKey(r *http.Request) string {
	if v := r.Header.Get("Authorization"); v != "" {
		if len(v) > 7 && strings.EqualFold(v[:7], "bearer ") {
			return strings.TrimSpace(v[7:])
		}
	}
	for _, h := range []string{"x-api-key", "x-goog-api-key"} {
		if v := r.Header.Get(h); v != "" {
			return strings.TrimSpace(v)
		}
	}
	return r.URL.Query().Get("key")
}

// guard enforces client keys (when configured) and the key's rate limit.
func (s *Server) guard(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if len(s.clientKeys) == 0 {
			next(w, r)
			return
		}
		key := presentedKey(r)
		ck := s.clientKeys[sha256.Sum256([]byte(key))]
		if key == "" || ck == nil {
			writeJSONError(w, http.StatusUnauthorized, "missing or invalid API key")
			return
		}
		if after, ok := ck.take(time.Now()); !ok {
			w.Header().Set("Retry-After", strconv.Itoa(after))
			writeJSONError(w, http.StatusTooManyRequests,
				fmt.Sprintf("rate limit exceeded for key %q (%d requests/min)", ck.name, ck.rpm))
			return
		}
		next(w, r.WithContext(context.WithValue(r.Context(), ctxKeyClient, ck)))
	}
}

func clientOf(ctx context.Context) *clientKey {
	ck, _ := ctx.Value(ctxKeyClient).(*clientKey)
	return ck
}

// clientName is the key name for logs ("" on an open gate).
func clientName(r *http.Request) string {
	if ck := clientOf(r.Context()); ck != nil {
		return ck.name
	}
	return ""
}

// appName is the OpenRouter app attribution (X-Title, else HTTP-Referer).
func appName(r *http.Request) string {
	if v := r.Header.Get("X-Title"); v != "" {
		return v
	}
	return r.Header.Get("HTTP-Referer")
}

// modelAllowed checks the caller's allowlist for a resolved (res may be nil
// for Anthropic passthrough ids) model, writing a 403 when denied.
func (s *Server) modelAllowed(w http.ResponseWriter, r *http.Request, clientModel string, res *resolve.Resolved) bool {
	ck := clientOf(r.Context())
	if ck == nil {
		return true
	}
	names := []string{clientModel}
	if res != nil {
		names = append(names, res.ModelName, res.Slug)
	}
	if ck.allows(names...) {
		return true
	}
	writeJSONError(w, http.StatusForbidden, fmt.Sprintf("key %q may not use model %q", ck.name, clientModel))
	return false
}
