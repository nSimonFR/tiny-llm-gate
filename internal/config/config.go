// Package config loads and validates tiny-llm-gate YAML configuration.
package config

import (
	"bytes"
	"errors"
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Config is the root configuration.
type Config struct {
	Listen     string                `yaml:"listen"`
	Providers  map[string]Provider   `yaml:"providers"`
	Models     map[string]Model      `yaml:"models"`
	Aliases    map[string]string     `yaml:"aliases"`
	DropParams bool                  `yaml:"drop_params"`
	MCPBridges map[string]MCPBridge  `yaml:"mcp_bridges,omitempty"`
}

// MCPBridge configures an MCP protocol transport bridge. It accepts
// connections using one MCP transport (frontend) and forwards JSON-RPC
// messages to an upstream MCP server using another transport (backend).
type MCPBridge struct {
	// Frontend is the transport exposed to clients. Supported: "sse".
	Frontend string `yaml:"frontend"`
	// Backend is the transport used to reach the upstream. Supported: "streamable_http".
	Backend string `yaml:"backend"`
	// UpstreamURL is the upstream MCP endpoint.
	UpstreamURL string `yaml:"upstream_url"`
	// PathPrefix is the URL prefix for this bridge's routes (e.g. "/mcp/affine").
	PathPrefix string `yaml:"path_prefix"`
	// Auth configures authentication for the upstream connection. Optional.
	Auth *Auth `yaml:"auth,omitempty"`
}

// Provider describes one upstream LLM endpoint.
type Provider struct {
	// Type selects the backend implementation. Currently supported: "openai".
	Type string `yaml:"type"`
	// BaseURL is the upstream root (e.g. "http://host:port/v1").
	BaseURL string `yaml:"base_url"`
	// APIKey is a shorthand for `auth: { type: bearer, token: <value> }`.
	// Kept for backwards compatibility with simple configs. Do not set when
	// `auth` is present — validation rejects the combination.
	APIKey string `yaml:"api_key"`
	// Auth describes how to authenticate upstream requests. Optional: when
	// absent and APIKey is empty, requests are sent unauthenticated.
	Auth *Auth `yaml:"auth,omitempty"`
}

// Auth is the upstream authentication strategy.
type Auth struct {
	// Type is one of: "bearer", "oauth_chatgpt".
	Type string `yaml:"type"`

	// -- bearer --
	Token string `yaml:"token,omitempty"`
	// TokenFile is a path to a file containing the bearer token. The file
	// is read once at startup. Alternative to Token for secret management
	// systems (e.g. agenix) that write tokens to files.
	TokenFile string `yaml:"token_file,omitempty"`

	// -- oauth_chatgpt: ChatGPT/Codex OAuth token file with auto-refresh --
	// File is the path to the JSON file containing {tokens: {access_token,
	// refresh_token, id_token}, last_refresh}. Matches the layout used by
	// `openai-oauth` (github.com/EvanZhouDev/openai-oauth).
	File string `yaml:"file,omitempty"`
	// Issuer is the OAuth issuer root. Defaults to https://auth.openai.com.
	Issuer string `yaml:"issuer,omitempty"`
	// ClientID is passed to the refresh endpoint. Defaults to the published
	// ChatGPT Codex client id.
	ClientID string `yaml:"client_id,omitempty"`
}

// Model binds a canonical model name to a provider and an upstream model id.
type Model struct {
	Provider      string   `yaml:"provider"`
	UpstreamModel string   `yaml:"upstream_model"`
	Fallback      []string `yaml:"fallback,omitempty"`
	// DefaultEmbedDimensions, when set, is injected into embedding requests
	// that arrive without an explicit dimension parameter (Gemini
	// outputDimensionality / OpenAI dimensions). Prevents Matryoshka-capable
	// models from returning their full native dimension when the client
	// SDK omits the field.
	DefaultEmbedDimensions *int `yaml:"default_embed_dimensions,omitempty"`
}

// Load reads and validates a YAML config file.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	return Parse(data)
}

// Parse validates a YAML config from bytes.
func Parse(data []byte) (*Config, error) {
	var c Config
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	if err := dec.Decode(&c); err != nil {
		return nil, fmt.Errorf("parse yaml: %w", err)
	}
	if err := c.validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}
	return &c, nil
}

func (c *Config) validate() error {
	if c.Listen == "" {
		c.Listen = "127.0.0.1:4001"
	}
	if len(c.Providers) == 0 {
		return errors.New("no providers defined")
	}
	for name, p := range c.Providers {
		if p.Type == "" {
			return fmt.Errorf("provider %q: type is required", name)
		}
		if p.Type != "openai" {
			return fmt.Errorf("provider %q: unknown type %q (supported: openai)", name, p.Type)
		}
		if p.BaseURL == "" {
			return fmt.Errorf("provider %q: base_url is required", name)
		}
		if p.APIKey != "" && p.Auth != nil {
			return fmt.Errorf("provider %q: use either api_key or auth, not both", name)
		}
		if p.Auth != nil {
			if err := validateAuth(name, p.Auth); err != nil {
				return err
			}
		}
	}
	if len(c.Models) == 0 {
		return errors.New("no models defined")
	}
	for name, m := range c.Models {
		if m.Provider == "" {
			return fmt.Errorf("model %q: provider is required", name)
		}
		if _, ok := c.Providers[m.Provider]; !ok {
			return fmt.Errorf("model %q: unknown provider %q", name, m.Provider)
		}
		if m.UpstreamModel == "" {
			return fmt.Errorf("model %q: upstream_model is required", name)
		}
		for _, f := range m.Fallback {
			if _, ok := c.Models[f]; !ok {
				return fmt.Errorf("model %q: unknown fallback %q", name, f)
			}
			if f == name {
				return fmt.Errorf("model %q: cannot fallback to itself", name)
			}
		}
	}
	for alias, target := range c.Aliases {
		if _, ok := c.Aliases[target]; ok {
			// chained alias — resolver handles cycle detection at runtime.
			continue
		}
		if _, ok := c.Models[target]; !ok {
			return fmt.Errorf("alias %q: target %q is not a model or alias", alias, target)
		}
	}
	if err := c.validateMCPBridges(); err != nil {
		return err
	}
	return nil
}

func (c *Config) validateMCPBridges() error {
	if len(c.MCPBridges) == 0 {
		return nil
	}
	seen := make(map[string]string) // path_prefix → bridge name
	for name, b := range c.MCPBridges {
		if b.Frontend != "sse" {
			return fmt.Errorf("mcp_bridge %q: unsupported frontend %q (supported: sse)", name, b.Frontend)
		}
		if b.Backend != "streamable_http" {
			return fmt.Errorf("mcp_bridge %q: unsupported backend %q (supported: streamable_http)", name, b.Backend)
		}
		if b.UpstreamURL == "" {
			return fmt.Errorf("mcp_bridge %q: upstream_url is required", name)
		}
		if b.PathPrefix == "" || b.PathPrefix[0] != '/' {
			return fmt.Errorf("mcp_bridge %q: path_prefix must start with /", name)
		}
		if other, dup := seen[b.PathPrefix]; dup {
			return fmt.Errorf("mcp_bridge %q: path_prefix %q conflicts with bridge %q", name, b.PathPrefix, other)
		}
		seen[b.PathPrefix] = name
		if b.Auth != nil {
			if err := validateAuth("mcp_bridge "+name, b.Auth); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateAuth(providerName string, a *Auth) error {
	switch a.Type {
	case "bearer":
		if a.Token == "" && a.TokenFile == "" {
			return fmt.Errorf("provider %q: auth.type=bearer requires auth.token or auth.token_file", providerName)
		}
		if a.Token != "" && a.TokenFile != "" {
			return fmt.Errorf("provider %q: use either auth.token or auth.token_file, not both", providerName)
		}
	case "oauth_chatgpt":
		if a.File == "" {
			return fmt.Errorf("provider %q: auth.type=oauth_chatgpt requires auth.file", providerName)
		}
	default:
		return fmt.Errorf("provider %q: unknown auth.type %q (supported: bearer, oauth_chatgpt)", providerName, a.Type)
	}
	return nil
}

// EffectiveAuth returns the authentication strategy for a provider, folding
// the legacy APIKey field into a bearer Auth. Returns nil when no auth is
// configured (unauthenticated upstream).
func (p Provider) EffectiveAuth() *Auth {
	if p.Auth != nil {
		return p.Auth
	}
	if p.APIKey != "" {
		return &Auth{Type: "bearer", Token: p.APIKey}
	}
	return nil
}
