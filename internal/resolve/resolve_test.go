package resolve

import (
	"testing"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
)

func testConfig() *config.Config {
	return &config.Config{
		Listen: "127.0.0.1:4001",
		Providers: map[string]config.Provider{
			"ollama": {Type: "openai", BaseURL: "http://ollama/v1", APIKey: "ollama"},
			"codex":  {Type: "openai", BaseURL: "http://codex/v1", APIKey: "unused"},
		},
		Models: map[string]config.Model{
			"gemma4:e4b": {
				Provider:      "ollama",
				UpstreamModel: "gemma4:e4b",
			},
			"qwen3-embedding:8b": {
				Provider:      "ollama",
				UpstreamModel: "qwen3-embedding:8b",
			},
			"gpt-5.4": {
				Provider:      "codex",
				UpstreamModel: "gpt-5.4",
				Fallback:      []string{"gemma4:e4b"},
			},
		},
		Aliases: map[string]string{
			"gpt-4o":                 "gemma4:e4b",
			"text-embedding-3-small": "qwen3-embedding:8b",
			"openai/gpt-5.4":         "gpt-5.4",
			// chained alias
			"gpt-4o-mini": "gpt-4o",
		},
	}
}

func TestResolveDirect(t *testing.T) {
	r := New(testConfig())
	got, err := r.Resolve("gemma4:e4b")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got.ModelName != "gemma4:e4b" || got.UpstreamModel != "gemma4:e4b" || got.ProviderName != "ollama" {
		t.Errorf("unexpected: %+v", got)
	}
}

func TestResolveAlias(t *testing.T) {
	r := New(testConfig())
	got, err := r.Resolve("gpt-4o")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got.ModelName != "gemma4:e4b" {
		t.Errorf("expected resolved to gemma4:e4b, got %q", got.ModelName)
	}
}

func TestResolveChainedAlias(t *testing.T) {
	r := New(testConfig())
	got, err := r.Resolve("gpt-4o-mini")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got.ModelName != "gemma4:e4b" {
		t.Errorf("expected chain to resolve to gemma4:e4b, got %q", got.ModelName)
	}
}

func TestResolveAliasPrefixed(t *testing.T) {
	r := New(testConfig())
	got, err := r.Resolve("openai/gpt-5.4")
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if got.ModelName != "gpt-5.4" {
		t.Errorf("expected gpt-5.4, got %q", got.ModelName)
	}
	if len(got.Fallback) != 1 || got.Fallback[0] != "gemma4:e4b" {
		t.Errorf("expected fallback to gemma4:e4b, got %v", got.Fallback)
	}
}

func TestResolveUnknown(t *testing.T) {
	r := New(testConfig())
	if _, err := r.Resolve("does-not-exist"); err == nil {
		t.Error("expected error, got nil")
	}
}

func TestResolveCycle(t *testing.T) {
	cfg := testConfig()
	cfg.Aliases["a"] = "b"
	cfg.Aliases["b"] = "a"
	r := New(cfg)
	if _, err := r.Resolve("a"); err == nil {
		t.Error("expected cycle error, got nil")
	}
}

func TestListModels(t *testing.T) {
	r := New(testConfig())
	models := r.ListModels()
	if len(models) != 7 {
		t.Errorf("expected 7 models/aliases, got %d: %v", len(models), models)
	}
}
