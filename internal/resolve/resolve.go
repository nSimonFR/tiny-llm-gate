// Package resolve maps a client-supplied model name through alias chains to a
// concrete provider and upstream model id. It also exposes the fallback chain
// for a model so callers can iterate the chain without re-looking-things-up.
package resolve

import (
	"fmt"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
)

// Resolver is safe for concurrent reads. Rebuilt on config reload.
type Resolver struct {
	cfg *config.Config
}

// New creates a Resolver over an already-validated config.
func New(cfg *config.Config) *Resolver { return &Resolver{cfg: cfg} }

// Resolved describes a concrete routing decision.
type Resolved struct {
	// ModelName is the canonical model name after alias resolution.
	ModelName string
	// UpstreamModel is the model id to send to the provider.
	UpstreamModel string
	// ProviderName is the provider key in the config.
	ProviderName string
	// Provider is the resolved provider config.
	Provider config.Provider
	// Fallback is the chain of model names to try if this one fails.
	// The chain is a list of canonical model names; each must still be
	// re-resolved to pick up its own provider/upstream_model.
	Fallback []string
	// DefaultEmbedDimensions, when set, is the default dimension to inject
	// into embedding requests that don't specify one.
	DefaultEmbedDimensions *int
}

// Resolve looks up a model by the name a client provided. Alias chains are
// followed (with cycle detection) until we land on a concrete model entry.
func (r *Resolver) Resolve(name string) (*Resolved, error) {
	seen := map[string]struct{}{}
	current := name
	for {
		if _, ok := seen[current]; ok {
			return nil, fmt.Errorf("alias cycle involving %q", current)
		}
		seen[current] = struct{}{}
		if target, ok := r.cfg.Aliases[current]; ok {
			current = target
			continue
		}
		break
	}
	m, ok := r.cfg.Models[current]
	if !ok {
		return nil, fmt.Errorf("unknown model %q", name)
	}
	p, ok := r.cfg.Providers[m.Provider]
	if !ok {
		// Validation should have caught this, but be defensive.
		return nil, fmt.Errorf("model %q references unknown provider %q", current, m.Provider)
	}
	return &Resolved{
		ModelName:              current,
		UpstreamModel:          m.UpstreamModel,
		ProviderName:           m.Provider,
		Provider:               p,
		Fallback:               m.Fallback,
		DefaultEmbedDimensions: m.DefaultEmbedDimensions,
	}, nil
}

// ListModels returns every model name addressable by a client (aliases and
// canonical names). Used for /v1/models responses.
func (r *Resolver) ListModels() []string {
	out := make([]string, 0, len(r.cfg.Models)+len(r.cfg.Aliases))
	for name := range r.cfg.Models {
		out = append(out, name)
	}
	for name := range r.cfg.Aliases {
		out = append(out, name)
	}
	return out
}
