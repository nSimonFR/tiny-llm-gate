# tiny-llm-gate

A memory-conscious, OpenAI-compatible LLM gateway.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## AI Installation Prompt

> Install `tiny-llm-gate` on a generic Linux or macOS host. **Done when** `curl -sH "Authorization: Bearer <gateway-token>" http://127.0.0.1:4001/v1/models` returns a JSON object with a non-empty `data` array.
>
> 1. Clone: `git clone https://github.com/nSimonFR/tiny-llm-gate && cd tiny-llm-gate`
> 2. Read first: `README.md`, `go.mod`, `cmd/tiny-llm-gate/main.go`, any `*.example.yaml` config. Toolchain is Go ≥ 1.25.
> 3. Build (try in this order):
>    - `nix build && cp result/bin/tiny-llm-gate ~/.local/bin/`
>    - `go build -o ~/.local/bin/tiny-llm-gate ./cmd/tiny-llm-gate`
>    Verify: `tiny-llm-gate --help` lists `-config` (or the equivalent flag).
> 4. Write a `config.yaml`. At minimum it must declare:
>    - One `auth:` entry (a bearer token clients will use against the gateway).
>    - One `providers:` entry (Anthropic / OpenAI / Gemini). For Anthropic OAuth, set `auth_file:` to a path containing the raw bearer token; the gateway re-reads on each request.
>    - One `routes:` entry mapping a model name to a provider.
>    Examples live in the README or `*.example.yaml`.
> 5. Run: `tiny-llm-gate -config config.yaml`. Default listen is `:4001`.
> 6. Point any OpenAI-compatible client at `http://<host>:4001/v1`, using whichever token you put under `auth:`.
> 7. NixOS hosts have a `services.tiny-llm-gate` module — see `module.nix`.
>
> **The gateway refuses unauthenticated requests by design.** If `auth:` is missing the daemon exits at startup, not silently — do not patch around it.
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

## License

MIT (see [LICENSE](LICENSE) once added).

## Contributing

Early days — issues and discussions welcome at https://github.com/nSimonFR/tiny-llm-gate/issues.
