# tiny-llm-gate

A tight, memory-conscious OpenAI-compatible LLM gateway.

## Goals

- **Tiny footprint**: target ≤ 12 MB RSS idle, ≤ 25 MB under load.
- **Drop-in OpenAI API**: `/v1/chat/completions`, `/v1/embeddings`, `/v1/models`.
- **Pluggable frontends & backends**: add Gemini/Anthropic protocols without touching the core.
- **Fallback chains**: transparent, server-side — clients don't know about them.
- **No framework, no runtime deps** beyond `net/http` stdlib + `gopkg.in/yaml.v3`.

## Non-goals

- Cost tracking in USD (bring your own observability).
- Prometheus metrics (use structured logs + Telegram alerting).
- Rate limiting / multi-tenant billing (this is a single-user gateway).
- Tokenizer-heavy features (would bloat the binary).

## Status

Phase 1: OpenAI in / OpenAI out. Non-streaming + streaming. Aliases, fallbacks.

See `ROADMAP.md` for planned phases.

## Config

See `testdata/example-config.yaml`.

## License

MIT (once pushed publicly).
