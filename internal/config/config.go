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
	Listen     string              `yaml:"listen"`
	Providers  map[string]Provider `yaml:"providers"`
	Models     map[string]Model    `yaml:"models"`
	Aliases    map[string]string   `yaml:"aliases"`
	DropParams bool                `yaml:"drop_params"`
}

// Provider describes one upstream LLM endpoint.
type Provider struct {
	// Type selects the backend implementation. Currently supported: "openai".
	Type string `yaml:"type"`
	// BaseURL is the upstream root (e.g. "http://host:port/v1").
	BaseURL string `yaml:"base_url"`
	// APIKey is used as a bearer token when non-empty.
	APIKey string `yaml:"api_key"`
}

// Model binds a canonical model name to a provider and an upstream model id.
type Model struct {
	Provider      string   `yaml:"provider"`
	UpstreamModel string   `yaml:"upstream_model"`
	Fallback      []string `yaml:"fallback,omitempty"`
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
	return nil
}

