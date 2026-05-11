# tiny-llm-gate

A memory-conscious, OpenAI-compatible LLM gateway.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Why

If you self-host LLMs (Ollama, llama.cpp) on a small box and want a single endpoint that:

- speaks the OpenAI API,
- rewrites client-facing model names to real upstream models,
- falls back transparently from one upstream to another on failure,
- streams responses token-by-token,

…your main option today is [LiteLLM](https://github.com/BerriAI/litellm). It's great, but it's a full Python stack (~65 MB RSS + sizeable dependency tree). On a Raspberry Pi 5 with 4 GB of RAM that's a meaningful chunk of the memory budget.

**tiny-llm-gate** is a single Go binary doing the same job in **under 10 MB of RSS**.

## Status

**v0.3.0**: OpenAI & Gemini frontends, OpenAI backend, streaming, aliases, fallbacks.

See [ROADMAP.md](ROADMAP.md) for remaining phases (production cutover, SIGHUP hot-reload).

## Quick start

```bash
cat > config.yaml <<'YAML'
listen: 127.0.0.1:4001

providers:
  ollama:
    type: openai
    base_url: http://192.168.1.10:11434/v1
    api_key: ollama

models:
  "gemma3:4b":
    provider: ollama
    upstream_model: gemma3:4b

aliases:
  "gpt-4o-mini": "gemma3:4b"
YAML

go run ./cmd/tiny-llm-gate --config config.yaml
```

Now point any OpenAI SDK at `http://127.0.0.1:4001/v1` and request `model: "gpt-4o-mini"` — it'll hit your Ollama instance as `gemma3:4b`.

## Config reference

```yaml
listen: 127.0.0.1:4001      # host:port (default 127.0.0.1:4001)

providers:                  # upstream LLM endpoints
  <name>:
    type: openai            # only "openai" today
    base_url: <url>         # e.g. http://host:11434/v1
    api_key: <string>       # Bearer token; omit for unauthenticated upstreams

models:                     # canonical model names
  <name>:
    provider: <provider>
    upstream_model: <id>    # model id actually sent to provider
    fallback:               # optional: try these models on 5xx upstream errors
      - <name>
      - <name>

aliases:                    # client-facing model name → canonical model
  <alias>: <model>          # chains are supported (cycle-detected)

```

See [`testdata/example-config.yaml`](testdata/example-config.yaml) for a fuller example.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST   | `/v1/chat/completions` | OpenAI chat, streaming + non-streaming |
| POST   | `/v1/embeddings`       | OpenAI embeddings |
| GET    | `/v1/models`           | list all model names + aliases |
| POST   | `/v1beta/models/{m}:generateContent`       | Gemini chat, non-streaming |
| POST   | `/v1beta/models/{m}:streamGenerateContent` | Gemini chat, streaming (newline-delimited JSON) |
| POST   | `/v1beta/models/{m}:embedContent`          | Gemini single-item embedding |
| POST   | `/v1beta/models/{m}:batchEmbedContents`    | Gemini batch embedding |
| GET    | `/health`              | liveness |
| GET    | `/ready`               | readiness (config loaded) |

Gemini requests are transparently translated to OpenAI format before being forwarded to the upstream — you can point an AFFiNE Gemini provider at this gateway and route to an Ollama backend.

## Memory discipline

| | |
|-|-|
| Binary size | **6.5 MiB** stripped |
| RSS idle    | **6.9 MiB** measured |
| Runtime deps | `gopkg.in/yaml.v3` only |
| HTTP client | stdlib `net/http`, bounded idle pool, HTTP/2 disabled for predictable streaming |
| Request body cap | 8 MiB |
| GOMEMLIMIT (recommended) | `20MiB` |
| MemoryMax (systemd, recommended) | `30M` |

The CI (TODO) fails PRs whose RSS regresses past these targets.

## Running on NixOS

Use the flake input directly:

```nix
# flake.nix
{
  inputs.tiny-llm-gate.url = "github:nSimonFR/tiny-llm-gate";

  outputs = { nixpkgs, tiny-llm-gate, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      modules = [
        tiny-llm-gate.nixosModules.default
        {
          services.tiny-llm-gate = {
            enable = true;
            package = tiny-llm-gate.packages.aarch64-linux.default;
            settings = {
              listen = "127.0.0.1:4001";
              providers.ollama = {
                type = "openai";
                base_url = "http://192.168.1.10:11434/v1";
                api_key = "ollama";
              };
              models."gemma3:4b" = {
                provider = "ollama";
                upstream_model = "gemma3:4b";
              };
              aliases."gpt-4o-mini" = "gemma3:4b";
            };
          };
        }
      ];
    };
  };
}
```

The systemd unit applies sandboxing (`DynamicUser`, `ProtectSystem=strict`, …) and sets `GOMEMLIMIT=20MiB` and `MemoryMax=30M` by default. Both are tunable via the module options.

## Comparison

| | tiny-llm-gate | LiteLLM | [Bifrost](https://github.com/maximhq/bifrost) | [one-api](https://github.com/songquanpeng/one-api) |
|---|---|---|---|---|
| Runtime | Go | Python | Go | Go + React |
| RSS idle | ~7 MiB | ~65 MiB | ~25 MiB | ~60 MiB |
| YAML config | ✅ | ✅ | partial | DB-backed |
| Server-side fallbacks | ✅ | ✅ | per-request | ✅ |
| Model aliases | ✅ (unified) | ✅ (two kinds) | per-key | via UI |
| Streaming (SSE) | ✅ | ✅ | ✅ | ✅ |
| Gemini-format frontend | roadmap | ✅ | partial | ❌ |
| OAuth backend | ❌ | ✅ | ❌ | ❌ |

## Non-goals

- Cost tracking in USD
- Prometheus `/metrics`
- Multi-tenant billing / quotas
- Request rate limiting
- Tokenizer-based features (counting, truncation)

Bring your own observability. Structured JSON logs are on stderr.

## Extensibility

Frontends and backends are the two extension points. Adding, say, Anthropic's `/v1/messages` frontend or an Anthropic-native backend is a single package implementing a small interface — no changes to the router.

This is enforced in the tree layout:

```
internal/
├── config/       # YAML + validation
├── resolve/      # model name → provider decision
└── server/       # HTTP wiring + frontend + backend (monolithic today)
```

Once Phase 3/4 land, the server package will split into `frontends/` and `backends/`.

## Setup for AI coding agents

For an autonomous agent (e.g. Claude Code) picking up the repo cold. Verified
against the current tree — no fabrication.

### Toolchain, dev shell, build, test, vet

Go `1.25` (`go.mod`); only runtime dep is `gopkg.in/yaml.v3`; built with
`CGO_ENABLED=0` and `-ldflags "-s -w"`. The flake's dev shell ships `go`,
`gopls`, `gotools`:

```bash
nix develop                                  # dev shell
nix build && ./result/bin/tiny-llm-gate -h   # reproducible build
go build -o tiny-llm-gate ./cmd/tiny-llm-gate   # faster Go iteration
go test ./...                                # tests under every internal/* pkg + server
go vet ./...                                 # always run before committing
```

CLI flags (`cmd/tiny-llm-gate/main.go`): `--config <path>` (default
`config.yaml` in CWD) and `--version`. No implicit `/etc/...` lookup — the
path is always explicit; the NixOS module renders `settings` to a Nix-store
YAML and passes it.

### Adding a new provider backend

Provider type today is `"openai"` only — gated by the switch in
`internal/config/config.go:validate()`. Outbound auth is pluggable via
`internal/auth/auth.go`'s `Authenticator` interface
(`Apply(ctx, *http.Request) error`); add a strategy by extending the
`switch ac.Type` in `auth.Build` and implementing a new type.

A non-OpenAI backend (anthropic-native, vertex, …) needs:

1. A new branch in the `Provider.Type` switch in `config.go`.
2. A backend-specific request builder where `proxyOpenAI` in
   `internal/server/openai.go` currently hard-codes `chatPath` /
   `embedPath` against `hop.Provider.BaseURL`.
3. Body translation when the wire format differs from OpenAI.
   `internal/gemini/translate.go` is the reference: it rewrites Gemini
   requests into OpenAI shape before reusing the OpenAI proxy path.

### Adding a new route

Routes live in `internal/server/server.go` inside `(*Server).Handler()` —
existing entries: `POST /v1/chat/completions`, `POST /v1/embeddings`,
`GET /v1/models`, `GET /v1beta/models`, `POST /v1beta/models/`
(dispatched in `routeGemini`), and conditionally `POST /v1/messages` when
`cfg.Anthropic != nil`. MCP bridges register their own routes via
`br.RegisterRoutes(mux)`.

Add a handler method on `*Server` in the matching file (`openai.go`,
`gemini.go`, `anthropic.go`, or a new sibling) and wire it in `Handler()`.
Gemini's `:action` suffix isn't natively expressible in Go's `http.ServeMux`;
the existing pattern is to register a prefix and dispatch on the suffix.

### GitHub operations & commit style

Hard rule for this org: every `gh` and `git push` runs as `nSimonFR-ai`,
never `nSimonFR`. Run `gh auth switch -u nSimonFR-ai` (verify with
`gh auth status`) before opening a PR.

Conventional commits with optional scope (`feat`, `fix`, `chore`; scopes
like `auth`, `gemini`, `oauth`, `anthropic`). From recent history:

```
feat(auth): FileBearer re-reads token_file on every request
fix(anthropic): also strip incoming x-api-key
chore: remove dead code (oauth_chatgpt, DropParams, ...)
```

Subjects under ~72 chars; the body explains the *why* when the diff isn't
self-evident.

### NixOS module — runtime files

The flake exports `nixosModules.default` and `nixosModules.tiny-llm-gate`
(both `nixos-module.nix`); see "Running on NixOS" above for the usage
shape. Notable extra options: `memoryMax` (default `30M`), `goMemLimit`
(`20MiB`), and `secretPaths` (paths added to `ReadOnlyPaths`). The unit
runs under `DynamicUser=true`, `ProtectSystem=strict`, `PrivateTmp=true`,
empty `CapabilityBoundingSet` — anything the binary reads at runtime must
appear in `secretPaths` or the sandbox denies it.

### claude-oauth tokens (Anthropic passthrough)

When the gate fronts Anthropic's `/v1/messages` for an OAuth-authenticated
client, the bearer token is supplied via `anthropic.auth.token_file`.
`auth.FileBearer.Apply` re-reads the file on every request, so an external
rotator updating the file (no service restart) is transparent. The nic-os
deployment stores it at `/run/agenix/claude-oauth`:

```nix
services.tiny-llm-gate = {
  enable = true;
  package = tiny-llm-gate.packages.aarch64-linux.default;
  secretPaths = [ "/run/agenix/claude-oauth" ];
  settings.anthropic = {
    upstream = "https://api.anthropic.com";
    auth = { type = "bearer"; token_file = "/run/agenix/claude-oauth"; };
  };
};
```

## License

MIT (see [LICENSE](LICENSE) once added).

## Contributing

Early days — issues and discussions welcome at https://github.com/nSimonFR/tiny-llm-gate/issues.
