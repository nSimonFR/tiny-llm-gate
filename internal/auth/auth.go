// Package auth implements authentication strategies for upstream providers.
package auth

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
)

// Authenticator is applied to every outbound request to add authentication
// headers. Implementations must be safe for concurrent use.
type Authenticator interface {
	Apply(ctx context.Context, req *http.Request) error
}

// AuthConfig mirrors the auth fields needed by Build. Kept minimal to avoid
// a circular import with config — callers map their config type to this.
type AuthConfig struct {
	Type      string
	Token     string
	TokenFile string
}

// Build constructs an Authenticator from an AuthConfig. Returns (nil, nil) when
// ac is nil (unauthenticated).
//
// For bearer auth with TokenFile, the file path is kept and re-read on every
// request. This lets external processes (e.g. claude-remote-control's OAuth
// refresh) rotate the token without restarting tiny-llm-gate. The initial
// read happens at Build() time only to surface configuration errors early.
func Build(ac *AuthConfig) (Authenticator, error) {
	if ac == nil {
		return nil, nil
	}
	switch ac.Type {
	case "bearer":
		if ac.TokenFile != "" {
			// Validate the path is readable now — fail fast on misconfiguration.
			if _, err := os.ReadFile(ac.TokenFile); err != nil {
				return nil, fmt.Errorf("read token_file: %w", err)
			}
			return FileBearer{Path: ac.TokenFile}, nil
		}
		return Bearer{Token: ac.Token}, nil
	default:
		return nil, fmt.Errorf("unsupported auth type %q", ac.Type)
	}
}

// Bearer sets `Authorization: Bearer <token>` with a fixed token.
type Bearer struct{ Token string }

func (b Bearer) Apply(_ context.Context, req *http.Request) error {
	if b.Token != "" {
		req.Header.Set("Authorization", "Bearer "+b.Token)
	}
	return nil
}

// FileBearer re-reads the token from a file on every Apply() call. Used when
// an external process rotates the token (e.g. a sidecar that extracts a
// refreshed OAuth access token from a credentials store).
type FileBearer struct{ Path string }

func (b FileBearer) Apply(_ context.Context, req *http.Request) error {
	data, err := os.ReadFile(b.Path)
	if err != nil {
		return fmt.Errorf("read token_file %s: %w", b.Path, err)
	}
	token := strings.TrimSpace(string(data))
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	return nil
}
