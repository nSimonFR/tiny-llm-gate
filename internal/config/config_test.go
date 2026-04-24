package config

import "testing"

const validConfig = `
listen: 127.0.0.1:4001
providers:
  ollama:
    type: openai
    base_url: http://localhost:11434/v1
    api_key: ollama
models:
  "gemma4:e4b":
    provider: ollama
    upstream_model: gemma4:e4b
  "qwen3-embedding:8b":
    provider: ollama
    upstream_model: qwen3-embedding:8b
  "gpt-5.4":
    provider: ollama
    upstream_model: gpt-5.4
    fallback: ["gemma4:e4b"]
aliases:
  "gpt-4o": "gemma4:e4b"
  "text-embedding-3-small": "qwen3-embedding:8b"
  "openai/gpt-5.4": "gpt-5.4"
drop_params: true
`

func TestParseValid(t *testing.T) {
	c, err := Parse([]byte(validConfig))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c.Listen != "127.0.0.1:4001" {
		t.Errorf("Listen = %q", c.Listen)
	}
	if len(c.Providers) != 1 {
		t.Errorf("expected 1 provider, got %d", len(c.Providers))
	}
	if len(c.Models) != 3 {
		t.Errorf("expected 3 models, got %d", len(c.Models))
	}
	if len(c.Aliases) != 3 {
		t.Errorf("expected 3 aliases, got %d", len(c.Aliases))
	}
	if !c.DropParams {
		t.Errorf("expected drop_params true")
	}
}

func TestParseDefaultListen(t *testing.T) {
	c, err := Parse([]byte(`
providers:
  p: { type: openai, base_url: http://x }
models:
  m: { provider: p, upstream_model: m }
`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c.Listen != "127.0.0.1:4001" {
		t.Errorf("expected default listen, got %q", c.Listen)
	}
}

func TestParseAuthBearer(t *testing.T) {
	c, err := Parse([]byte(`
providers:
  p:
    type: openai
    base_url: http://x/v1
    auth:
      type: bearer
      token: abc
models:
  m: { provider: p, upstream_model: m }
`))
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	a := c.Providers["p"].EffectiveAuth()
	if a == nil || a.Type != "bearer" || a.Token != "abc" {
		t.Errorf("effective auth = %+v", a)
	}
}

func TestParseAuthOAuthChatGPT(t *testing.T) {
	c, err := Parse([]byte(`
providers:
  p:
    type: openai
    base_url: http://x
    auth:
      type: oauth_chatgpt
      file: /tmp/auth.json
      issuer: https://auth.example
      client_id: app_x
models:
  m: { provider: p, upstream_model: m }
`))
	if err != nil {
		t.Fatalf("unexpected: %v", err)
	}
	a := c.Providers["p"].EffectiveAuth()
	if a == nil || a.Type != "oauth_chatgpt" {
		t.Fatalf("effective auth = %+v", a)
	}
	if a.File != "/tmp/auth.json" || a.Issuer != "https://auth.example" || a.ClientID != "app_x" {
		t.Errorf("oauth fields not parsed: %+v", a)
	}
}

func TestApiKeyBecomesBearer(t *testing.T) {
	c, err := Parse([]byte(`
providers:
  p: { type: openai, base_url: http://x, api_key: hunter2 }
models:
  m: { provider: p, upstream_model: m }
`))
	if err != nil {
		t.Fatal(err)
	}
	a := c.Providers["p"].EffectiveAuth()
	if a == nil || a.Type != "bearer" || a.Token != "hunter2" {
		t.Errorf("api_key did not fold into bearer: %+v", a)
	}
}

func TestParseRejectsInvalid(t *testing.T) {
	cases := map[string]string{
		"no providers": `
models:
  m: { provider: p, upstream_model: m }
`,
		"no models": `
providers:
  p: { type: openai, base_url: http://x }
`,
		"model references unknown provider": `
providers:
  p: { type: openai, base_url: http://x }
models:
  m: { provider: other, upstream_model: m }
`,
		"provider missing base_url": `
providers:
  p: { type: openai }
models:
  m: { provider: p, upstream_model: m }
`,
		"provider unknown type": `
providers:
  p: { type: bogus, base_url: http://x }
models:
  m: { provider: p, upstream_model: m }
`,
		"model missing upstream_model": `
providers:
  p: { type: openai, base_url: http://x }
models:
  m: { provider: p }
`,
		"unknown fallback": `
providers:
  p: { type: openai, base_url: http://x }
models:
  m: { provider: p, upstream_model: m, fallback: ["missing"] }
`,
		"self-fallback": `
providers:
  p: { type: openai, base_url: http://x }
models:
  m: { provider: p, upstream_model: m, fallback: ["m"] }
`,
		"alias to nothing": `
providers:
  p: { type: openai, base_url: http://x }
models:
  m: { provider: p, upstream_model: m }
aliases:
  a: "nowhere"
`,
		"unknown field": `
providers:
  p: { type: openai, base_url: http://x }
models:
  m: { provider: p, upstream_model: m }
bogus_top_level: 1
`,
		"both api_key and auth": `
providers:
  p:
    type: openai
    base_url: http://x
    api_key: k
    auth: { type: bearer, token: k }
models:
  m: { provider: p, upstream_model: m }
`,
		"unknown auth type": `
providers:
  p:
    type: openai
    base_url: http://x
    auth: { type: magic }
models:
  m: { provider: p, upstream_model: m }
`,
		"bearer auth missing token": `
providers:
  p:
    type: openai
    base_url: http://x
    auth: { type: bearer }
models:
  m: { provider: p, upstream_model: m }
`,
		"oauth missing file": `
providers:
  p:
    type: openai
    base_url: http://x
    auth: { type: oauth_chatgpt }
models:
  m: { provider: p, upstream_model: m }
`,
		"mcp_bridge bad frontend": `
providers:
  p: { type: openai, base_url: http://x }
models:
  m: { provider: p, upstream_model: m }
mcp_bridges:
  b: { frontend: websocket, backend: streamable_http, upstream_url: http://x, path_prefix: /mcp }
`,
		"mcp_bridge bad backend": `
providers:
  p: { type: openai, base_url: http://x }
models:
  m: { provider: p, upstream_model: m }
mcp_bridges:
  b: { frontend: sse, backend: stdio, upstream_url: http://x, path_prefix: /mcp }
`,
		"mcp_bridge missing upstream_url": `
providers:
  p: { type: openai, base_url: http://x }
models:
  m: { provider: p, upstream_model: m }
mcp_bridges:
  b: { frontend: sse, backend: streamable_http, path_prefix: /mcp }
`,
		"mcp_bridge missing path_prefix slash": `
providers:
  p: { type: openai, base_url: http://x }
models:
  m: { provider: p, upstream_model: m }
mcp_bridges:
  b: { frontend: sse, backend: streamable_http, upstream_url: http://x, path_prefix: mcp }
`,
		"mcp_bridge bearer both token and token_file": `
providers:
  p: { type: openai, base_url: http://x }
models:
  m: { provider: p, upstream_model: m }
mcp_bridges:
  b:
    frontend: sse
    backend: streamable_http
    upstream_url: http://x
    path_prefix: /mcp
    auth: { type: bearer, token: abc, token_file: /tmp/t }
`,
	}
	for name, cfg := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := Parse([]byte(cfg)); err == nil {
				t.Errorf("expected error for %q, got nil", name)
			}
		})
	}
}

func TestParseMCPBridgeValid(t *testing.T) {
	c, err := Parse([]byte(`
providers:
  p: { type: openai, base_url: http://x }
models:
  m: { provider: p, upstream_model: m }
mcp_bridges:
  affine:
    frontend: sse
    backend: streamable_http
    upstream_url: http://127.0.0.1:13010/mcp
    path_prefix: /mcp/affine
    auth:
      type: bearer
      token: secret123
`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(c.MCPBridges) != 1 {
		t.Fatalf("expected 1 MCP bridge, got %d", len(c.MCPBridges))
	}
	b := c.MCPBridges["affine"]
	if b.Frontend != "sse" || b.Backend != "streamable_http" {
		t.Errorf("bridge types: frontend=%q backend=%q", b.Frontend, b.Backend)
	}
	if b.UpstreamURL != "http://127.0.0.1:13010/mcp" {
		t.Errorf("upstream_url = %q", b.UpstreamURL)
	}
	if b.PathPrefix != "/mcp/affine" {
		t.Errorf("path_prefix = %q", b.PathPrefix)
	}
	if b.Auth == nil || b.Auth.Token != "secret123" {
		t.Errorf("auth = %+v", b.Auth)
	}
}

func TestParseMCPBridgeTokenFile(t *testing.T) {
	_, err := Parse([]byte(`
providers:
  p: { type: openai, base_url: http://x }
models:
  m: { provider: p, upstream_model: m }
mcp_bridges:
  b:
    frontend: sse
    backend: streamable_http
    upstream_url: http://x
    path_prefix: /mcp/b
    auth:
      type: bearer
      token_file: /run/agenix/token
`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestParseMCPBridgeNoBridges(t *testing.T) {
	// Config without mcp_bridges should still be valid.
	_, err := Parse([]byte(validConfig))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestParseAnthropic(t *testing.T) {
	c, err := Parse([]byte(`
providers:
  p: { type: openai, base_url: http://x }
models:
  m: { provider: p, upstream_model: m }
anthropic:
  upstream: https://api.anthropic.com
  shadow_url: http://ai.gate-mintaka.ts.net/v1/chat/completions
`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c.Anthropic == nil {
		t.Fatal("expected anthropic section")
	}
	if c.Anthropic.Upstream != "https://api.anthropic.com" {
		t.Errorf("upstream = %q", c.Anthropic.Upstream)
	}
	if c.Anthropic.ShadowURL != "http://ai.gate-mintaka.ts.net/v1/chat/completions" {
		t.Errorf("shadow_url = %q", c.Anthropic.ShadowURL)
	}
}

func TestParseAnthropicNoShadow(t *testing.T) {
	c, err := Parse([]byte(`
providers:
  p: { type: openai, base_url: http://x }
models:
  m: { provider: p, upstream_model: m }
anthropic:
  upstream: https://api.anthropic.com
`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c.Anthropic == nil || c.Anthropic.ShadowURL != "" {
		t.Errorf("expected anthropic with empty shadow_url, got %+v", c.Anthropic)
	}
}

func TestParseAnthropicMissingUpstream(t *testing.T) {
	_, err := Parse([]byte(`
providers:
  p: { type: openai, base_url: http://x }
models:
  m: { provider: p, upstream_model: m }
anthropic:
  shadow_url: http://x/v1/chat/completions
`))
	if err == nil {
		t.Error("expected error for anthropic without upstream")
	}
}

func TestParseAnthropicAbsent(t *testing.T) {
	c, err := Parse([]byte(validConfig))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if c.Anthropic != nil {
		t.Errorf("expected nil anthropic, got %+v", c.Anthropic)
	}
}
