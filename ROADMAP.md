# Roadmap

| Phase | Scope | Status |
|-------|-------|--------|
| 0     | Repo skeleton, Go module, flake, NixOS module | in progress |
| 1     | OpenAI frontend + OpenAI backend (API key) + aliases + fallbacks + streaming | in progress |
| 2     | Validated streaming end-to-end against OpenWebUI + PicoClaw | |
| 3     | `openai-codex-oauth` backend absorbs `openai-codex-proxy` (OAuth token + refresh) | |
| 4     | Gemini frontend absorbs `affine-embed-proxy` (`generateContent`, `streamGenerateContent`, `embedContent`, `batchEmbedContents`) | |
| 5     | Cutover: delete LiteLLM + codex-proxy + embed-proxy from nic-os flake | |
| 6     | Tier-1 generic features: SIGHUP hot-reload, structured logs, retries w/ backoff, circuit breakers, graceful shutdown | |
| 7     | Tier-2 polish: embedding cache, client-side API keys (optional) | |

## Memory budget (hard constraint, all phases)

- ≤ 12 MB RSS idle
- ≤ 25 MB RSS under sustained load
- Binary ≤ 8 MB stripped (`-ldflags="-s -w"`)
- `GOMEMLIMIT=20MiB` set via systemd env
- `MemoryMax=30M` in systemd unit
- CI memory regression test fails PRs that regress
