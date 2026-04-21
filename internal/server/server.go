// Package server wires HTTP handlers to the resolver and manages the upstream
// HTTP client. All request/response plumbing lives here.
package server

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/auth"
	"github.com/nSimonFR/tiny-llm-gate/internal/config"
	"github.com/nSimonFR/tiny-llm-gate/internal/resolve"
)

// Server is the public type exposed to main.
type Server struct {
	cfg      *config.Config
	resolver *resolve.Resolver
	client   *http.Client
	logger   *slog.Logger
	// auths is a per-provider Authenticator, built once at startup.
	auths map[string]auth.Authenticator
}

// New builds a Server. The *http.Client has generous timeouts for streaming
// LLM responses but tight idle-connection limits to keep memory bounded.
func New(cfg *config.Config, logger *slog.Logger) (*Server, error) {
	transport := &http.Transport{
		// Bounded pool keeps FD + memory tight.
		MaxIdleConns:          16,
		MaxIdleConnsPerHost:   4,
		IdleConnTimeout:       60 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		ResponseHeaderTimeout: 30 * time.Second,
		// Force HTTP/1.1 for predictable streaming and lower memory
		// overhead than HTTP/2's frame buffering.
		ForceAttemptHTTP2: false,
		DialContext: (&net.Dialer{
			Timeout:   10 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
	}
	auths, err := buildAuthenticators(cfg)
	if err != nil {
		return nil, err
	}
	return &Server{
		cfg:      cfg,
		resolver: resolve.New(cfg),
		// No overall Timeout — streaming responses can legitimately run
		// for minutes. Per-phase timeouts live on the Transport.
		client: &http.Client{Transport: transport},
		logger: logger,
		auths:  auths,
	}, nil
}

// buildAuthenticators constructs one auth.Authenticator per provider based on
// config. Providers without authentication get no entry — sendUpstream treats
// missing as "send without auth header".
func buildAuthenticators(cfg *config.Config) (map[string]auth.Authenticator, error) {
	out := make(map[string]auth.Authenticator, len(cfg.Providers))
	for name, p := range cfg.Providers {
		a := p.EffectiveAuth()
		if a == nil {
			continue
		}
		switch a.Type {
		case "bearer":
			out[name] = auth.Bearer{Token: a.Token}
		case "oauth_chatgpt":
			oa, err := auth.NewChatGPTOAuth(a.File, a.Issuer, a.ClientID)
			if err != nil {
				return nil, fmt.Errorf("provider %q oauth: %w", name, err)
			}
			out[name] = oa
		default:
			return nil, fmt.Errorf("provider %q: unsupported auth type %q", name, a.Type)
		}
	}
	return out, nil
}

// Handler returns the HTTP handler for this server.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// OpenAI frontend
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("POST /v1/embeddings", s.handleEmbeddings)
	mux.HandleFunc("GET /v1/models", s.handleModels)

	// Health and readiness
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /ready", s.handleReady)

	return withRequestID(mux)
}

// Shutdown gracefully closes idle upstream connections.
func (s *Server) Shutdown(ctx context.Context) error {
	s.client.CloseIdleConnections()
	return nil
}

func (s *Server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ok"}` + "\n"))
}

func (s *Server) handleReady(w http.ResponseWriter, _ *http.Request) {
	// Phase 1: readiness == config loaded. Upstream probes come later.
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ready"}` + "\n"))
}
