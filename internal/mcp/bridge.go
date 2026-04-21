package mcp

import (
	"context"
	"log/slog"
	"net/http"
	"sync"

	"github.com/nSimonFR/tiny-llm-gate/internal/auth"
	"github.com/nSimonFR/tiny-llm-gate/internal/config"
)

// Bridge manages one MCP transport bridge (e.g. SSE frontend →
// StreamableHTTP backend). Multiple bridges can coexist on a single server,
// each with its own path prefix.
type Bridge struct {
	name     string
	cfg      config.MCPBridge
	auth     auth.Authenticator
	client   *http.Client
	sessions sync.Map // map[string]*session
	logger   *slog.Logger

	ctx    context.Context
	cancel context.CancelFunc
}

// NewBridge constructs a bridge from configuration. The provided client is
// used for all upstream HTTP requests.
func NewBridge(name string, cfg config.MCPBridge, client *http.Client, logger *slog.Logger) (*Bridge, error) {
	var ac *auth.AuthConfig
	if cfg.Auth != nil {
		ac = &auth.AuthConfig{
			Type:      cfg.Auth.Type,
			Token:     cfg.Auth.Token,
			TokenFile: cfg.Auth.TokenFile,
			File:      cfg.Auth.File,
			Issuer:    cfg.Auth.Issuer,
			ClientID:  cfg.Auth.ClientID,
		}
	}
	authn, err := auth.Build(ac)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithCancel(context.Background())
	return &Bridge{
		name:   name,
		cfg:    cfg,
		auth:   authn,
		client: client,
		logger: logger.With("mcp_bridge", name),
		ctx:    ctx,
		cancel: cancel,
	}, nil
}

// RegisterRoutes adds the bridge's HTTP handlers to the given mux.
// Routes are prefixed with the bridge's PathPrefix from config.
func (b *Bridge) RegisterRoutes(mux *http.ServeMux) {
	prefix := b.cfg.PathPrefix
	mux.HandleFunc("GET "+prefix+"/sse", b.HandleSSE)
	mux.HandleFunc("POST "+prefix+"/message", b.HandleMessage)
	b.logger.Info("MCP bridge routes registered",
		"sse", prefix+"/sse",
		"message", prefix+"/message",
		"upstream", b.cfg.UpstreamURL,
	)
}

// Shutdown cancels the bridge context and waits for active sessions to drain.
func (b *Bridge) Shutdown(_ context.Context) error {
	b.cancel()
	b.sessions.Range(func(key, value any) bool {
		value.(*session).Close()
		b.sessions.Delete(key)
		return true
	})
	return nil
}
