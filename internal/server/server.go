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
	"github.com/nSimonFR/tiny-llm-gate/internal/mcp"
	"github.com/nSimonFR/tiny-llm-gate/internal/resolve"
)

// Server is the public type exposed to main.
type Server struct {
	cfg      *config.Config
	resolver *resolve.Resolver
	client   *http.Client
	logger   *slog.Logger
	// auths is a per-provider Authenticator, built once at startup.
	auths   map[string]auth.Authenticator
	bridges []*mcp.Bridge
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

	// Build MCP bridges. They share a separate HTTP client with longer
	// timeouts suited to MCP streaming responses.
	var bridges []*mcp.Bridge
	if len(cfg.MCPBridges) > 0 {
		mcpTransport := &http.Transport{
			MaxIdleConns:          8,
			MaxIdleConnsPerHost:   2,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ResponseHeaderTimeout: 120 * time.Second,
			ForceAttemptHTTP2:     false,
			DialContext: (&net.Dialer{
				Timeout:   10 * time.Second,
				KeepAlive: 30 * time.Second,
			}).DialContext,
		}
		mcpClient := &http.Client{Transport: mcpTransport}
		for name, bcfg := range cfg.MCPBridges {
			br, err := mcp.NewBridge(name, bcfg, mcpClient, logger)
			if err != nil {
				return nil, fmt.Errorf("mcp_bridge %q: %w", name, err)
			}
			bridges = append(bridges, br)
		}
	}

	return &Server{
		cfg:      cfg,
		resolver: resolve.New(cfg),
		// No overall Timeout — streaming responses can legitimately run
		// for minutes. Per-phase timeouts live on the Transport.
		client:  &http.Client{Transport: transport},
		logger:  logger,
		auths:   auths,
		bridges: bridges,
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
		authn, err := auth.Build(authConfigFromConfig(a))
		if err != nil {
			return nil, fmt.Errorf("provider %q: %w", name, err)
		}
		if authn != nil {
			out[name] = authn
		}
	}
	return out, nil
}

// authConfigFromConfig converts a config.Auth to an auth.AuthConfig.
func authConfigFromConfig(a *config.Auth) *auth.AuthConfig {
	if a == nil {
		return nil
	}
	return &auth.AuthConfig{
		Type:      a.Type,
		Token:     a.Token,
		TokenFile: a.TokenFile,
		File:      a.File,
		Issuer:    a.Issuer,
		ClientID:  a.ClientID,
	}
}

// Handler returns the HTTP handler for this server.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// OpenAI frontend
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("POST /v1/embeddings", s.handleEmbeddings)
	mux.HandleFunc("GET /v1/models", s.handleModels)

	// Gemini frontend. The Gemini URL form is /v1beta/models/{model}:action
	// where `:action` is a suffix on the final path segment, not a separator
	// Go's ServeMux handles natively. We route by prefix and dispatch on the
	// action in a single handler.
	mux.HandleFunc("GET /v1beta/models", s.handleGeminiModels)
	mux.HandleFunc("POST /v1beta/models/", s.routeGemini)

	// Health and readiness
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /ready", s.handleReady)

	// MCP bridges
	for _, br := range s.bridges {
		br.RegisterRoutes(mux)
	}

	return withRequestID(mux)
}

// routeGemini dispatches /v1beta/models/{model}:action to the right handler.
// Splitting on the last colon gives us the action suffix.
func (s *Server) routeGemini(w http.ResponseWriter, r *http.Request) {
	// Find last ':' in the path — the action separator Gemini uses.
	path := r.URL.Path
	colon := -1
	for i := len(path) - 1; i >= 0; i-- {
		if path[i] == ':' {
			colon = i
			break
		}
		if path[i] == '/' {
			break
		}
	}
	if colon < 0 {
		writeJSONError(w, http.StatusNotFound, "expected /v1beta/models/{model}:action")
		return
	}
	action := path[colon+1:]
	switch action {
	case "generateContent":
		s.handleGenerateContent(w, r)
	case "streamGenerateContent":
		s.handleStreamGenerateContent(w, r)
	case "embedContent":
		s.handleEmbedContent(w, r)
	case "batchEmbedContents":
		s.handleBatchEmbedContents(w, r)
	default:
		writeJSONError(w, http.StatusNotFound, "unknown Gemini action: "+action)
	}
}

// Shutdown gracefully closes idle upstream connections and MCP bridges.
func (s *Server) Shutdown(ctx context.Context) error {
	for _, br := range s.bridges {
		_ = br.Shutdown(ctx)
	}
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
