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
// ac is nil (unauthenticated). For bearer auth with TokenFile, the file is read
// once at call time.
func Build(ac *AuthConfig) (Authenticator, error) {
	if ac == nil {
		return nil, nil
	}
	switch ac.Type {
	case "bearer":
		token := ac.Token
		if ac.TokenFile != "" {
			data, err := os.ReadFile(ac.TokenFile)
			if err != nil {
				return nil, fmt.Errorf("read token_file: %w", err)
			}
			token = strings.TrimSpace(string(data))
		}
		return Bearer{Token: token}, nil
	default:
		return nil, fmt.Errorf("unsupported auth type %q", ac.Type)
	}
}

// Bearer sets `Authorization: Bearer <token>`.
type Bearer struct{ Token string }

func (b Bearer) Apply(_ context.Context, req *http.Request) error {
	if b.Token != "" {
		req.Header.Set("Authorization", "Bearer "+b.Token)
	}
	return nil
}
