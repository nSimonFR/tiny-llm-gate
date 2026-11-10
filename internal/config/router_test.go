package config

import (
	"strings"
	"testing"
	"time"
)

const routerBase = `
providers:
  p: { type: openai, base_url: http://x }
models:
  cheap: { provider: p, upstream_model: c }
  strong: { provider: p, upstream_model: s }
aliases:
  openai/smart: smart
`

func TestParseRouter(t *testing.T) {
	c, err := Parse([]byte(routerBase + `
routers:
  smart:
    classifier: { model: jev-1.13.0, auth: { type: bearer, token: k } }
    tiers: [cheap, strong]
    questions:
      hard: { instructions: "Is it hard?", escalate_to: strong, threshold: 0.4 }
`))
	if err != nil {
		t.Fatal(err)
	}
	r := c.Routers["smart"]
	if r.ErrorTier() != "strong" || r.TTL() != DefaultRouterCacheTTL || r.Classifier.Timeout() != DefaultRouterTimeout {
		t.Errorf("defaults: on_error=%s ttl=%s timeout=%s", r.ErrorTier(), r.TTL(), r.Classifier.Timeout())
	}
	r.CacheTTL, r.Classifier.TimeoutMS = "1h", 900
	if r.TTL() != time.Hour || r.Classifier.Timeout() != 900*time.Millisecond {
		t.Error("explicit ttl/timeout ignored")
	}
}

func TestParseRejectsInvalidRouter(t *testing.T) {
	router := func(body string) string { return routerBase + "routers:\n  smart:\n" + body }
	ok := `    classifier: { model: jev }
    tiers: [cheap, strong]
    questions:
      hard: { instructions: x, escalate_to: strong, threshold: 0.5 }
`
	cases := map[string]string{
		"name clashes with model":  strings.Replace(router(ok), "routers:\n  smart:", "routers:\n  cheap:", 1),
		"missing classifier model": router(strings.Replace(ok, "model: jev", "endpoint: http://j", 1)),
		"no tiers":                 router(strings.Replace(ok, "[cheap, strong]", "[]", 1)),
		"unknown tier":             router(strings.Replace(ok, "[cheap, strong]", "[cheap, nope]", 1)),
		"duplicate tier":           router(strings.Replace(ok, "[cheap, strong]", "[cheap, cheap]", 1)),
		"escalate_to not a tier":   router(strings.Replace(ok, "escalate_to: strong", "escalate_to: other", 1)),
		"threshold out of range":   router(strings.Replace(ok, "threshold: 0.5", "threshold: 1.5", 1)),
		"empty instructions":       router(strings.Replace(ok, "instructions: x", `instructions: " "`, 1)),
		"on_error not a tier":      router(ok + "    on_error: other\n"),
		"bad cache_ttl":            router(ok + "    cache_ttl: soon\n"),
		"no questions":             router("    classifier: { model: jev }\n    tiers: [cheap]\n"),
		"router tier is a router":  router(strings.Replace(ok, "[cheap, strong]", "[cheap, openai/smart]", 1)),
	}
	for name, yml := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := Parse([]byte(yml)); err == nil {
				t.Errorf("accepted:\n%s", yml)
			}
		})
	}
}
