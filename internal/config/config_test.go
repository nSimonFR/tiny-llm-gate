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
	}
	for name, cfg := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := Parse([]byte(cfg)); err == nil {
				t.Errorf("expected error for %q, got nil", name)
			}
		})
	}
}
