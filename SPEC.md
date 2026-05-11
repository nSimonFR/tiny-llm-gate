# tiny-llm-gate — Specification

This document specifies `tiny-llm-gate` **as currently implemented** at commit
`6317e13` (v0.3.3-dev). Verified against the code in `cmd/`, `internal/`,
`flake.nix`, `nixos-module.nix`, and `testdata/example-config.yaml`.

It is intended as the authoritative reference for agents and humans making
changes. Where the README or memory hints diverge, this document wins because
it was derived directly from the source.

---

## 1. Purpose

`tiny-llm-gate` is a single-binary, memory-conscious LLM gateway written in
Go. It was built as a drop-in replacement for [LiteLLM] on memory-constrained
hosts (Raspberry Pi 5 with 4 GB RAM is the reference target).

It serves three concurrent HTTP frontends in one process:

1. **OpenAI-compatible `/v1/...`** — primary endpoint used by clients that
   speak OpenAI, including AFFiNE, Open WebUI, PicoClaw, and any
   `openai`-flavoured SDK.
2. **Anthropic-compatible `/v1/messages`** — a thin pass-through proxy used by
   Claude Code → Aperture (the user's observability layer). The gateway does
   not interpret the body; it strips client auth and applies its own.
3. **Gemini-compatible `/v1beta/...`** — translation frontend used by
   AFFiNE's Gemini provider (text embeddings primarily, also `generateContent`
   / `streamGenerateContent`). Requests are translated to OpenAI shape on the
   wire to a single upstream (typically Ollama).

A fourth surface exists alongside these:

4. **MCP transport bridges** — generic SSE-frontend, StreamableHTTP-backend
   protocol bridges. Used to expose AFFiNE's StreamableHTTP MCP server as SSE
   on the tailnet for older MCP clients (this is the `/mcp/affine` route).

It replaced LiteLLM and a previous `affine-embed-proxy` Node service in the
user's `nic-os` flake. Resident memory under sustained load is ~7 MiB
(stripped binary is ~6.5 MiB).

[LiteLLM]: https://github.com/BerriAI/litellm

---

## 2. Process model & entry point

- Binary: `cmd/tiny-llm-gate/main.go`.
- Flags:
  - `-config <path>` — YAML config path. Default `config.yaml`.
  - `-version` — print version and exit.
- Version is injected at build time via `-ldflags "-X main.Version=…"`.
- Logging: `slog` JSON handler on **stderr**, level Info.
- Listen address comes from `cfg.Listen`. Default `127.0.0.1:4001`.
- HTTP server config: `ReadHeaderTimeout: 10s`, **no `WriteTimeout`** (streaming
  responses can be arbitrarily long).
- Graceful shutdown on `SIGINT` / `SIGTERM`: 10-second drain, MCP bridge
  shutdown, idle-connection close.

There is no SIGHUP hot-reload (listed under Phase 6 in `ROADMAP.md`, not
implemented). Config changes require a full restart.

---

## 3. Configuration

### 3.1 File format

YAML, decoded via `gopkg.in/yaml.v3` with `KnownFields(true)`. Unknown keys
fail loading. Validation runs at startup and exits non-zero on failure.

The Go type is `internal/config.Config`:

```go
type Config struct {
    Listen     string                  // host:port; default 127.0.0.1:4001
    Providers  map[string]Provider     // required, ≥1 entry
    Models     map[string]Model        // required, ≥1 entry
    Aliases    map[string]string       // optional, may chain
    MCPBridges map[string]MCPBridge    // optional
    Anthropic  *Anthropic              // optional
}
```

### 3.2 `providers`

Map of provider name → upstream LLM endpoint.

```yaml
providers:
  <name>:
    type: openai          # required; currently the only supported value
    base_url: <url>       # required; e.g. http://host:port/v1
    api_key: <string>     # legacy shorthand; mutually exclusive with `auth`
    auth:                 # optional; preferred
      type: bearer        # only supported value
      token: <string>     # mutually exclusive with token_file
      token_file: <path>  # mutually exclusive with token
```

Validation:

- `type` must be `"openai"`. Any other value rejected.
- `base_url` required.
- `api_key` and `auth` cannot both be set.
- For `auth.type=bearer`: exactly one of `token` or `token_file` must be set.
- For `token_file`: the file must be readable at startup (read once to fail
  fast on misconfig; see §6 for re-read semantics).

When neither `api_key` nor `auth` is set, upstream requests are sent without an
`Authorization` header.

### 3.3 `models`

Map of canonical model name → routing decision.

```yaml
models:
  <name>:
    provider: <provider-name>          # required; must exist in providers
    upstream_model: <id>               # required; what we send upstream
    fallback:                          # optional; tried on retryable failure
      - <model-name>
    default_embed_dimensions: <int>    # optional; injected into embedding
                                       #   requests that don't specify dimensions
```

Validation:

- `provider` must reference a known provider.
- `upstream_model` required.
- Fallback names must each be a known model name.
- A model cannot list itself in `fallback`.

`default_embed_dimensions` is critical for Matryoshka-capable embedding models
(e.g. `qwen3-embedding:8b`) — without it the model returns its full native
dimension, breaking callers that expect a smaller fixed width (e.g. AFFiNE's
pgvector column).

### 3.4 `aliases`

Map of client-facing model name → another model name.

```yaml
aliases:
  "gpt-4o-mini": "gemma3:4b"
  "openai/gemma4:e4b": "gemma4:e4b"
```

- Targets may be other aliases (chains supported; cycle-detected at resolve
  time, not at validation time — see §5.2).
- At validation time, every alias target must be either a known alias or a
  known model.

### 3.5 `mcp_bridges`

Map of bridge name → MCP transport-bridge spec.

```yaml
mcp_bridges:
  <name>:
    frontend: sse                          # required; only "sse" supported
    backend: streamable_http               # required; only "streamable_http" supported
    upstream_url: <url>                    # required; full URL to upstream MCP endpoint
    path_prefix: /mcp/<name>               # required; must start with /
    auth:                                  # optional
      type: bearer
      token: <string> | token_file: <path>
```

Validation:

- `frontend` must equal `"sse"`. Any other value rejected.
- `backend` must equal `"streamable_http"`. Any other value rejected.
- `upstream_url` required.
- `path_prefix` required and must start with `/`.
- Path prefixes must be unique across bridges.

The path prefix is arbitrary; the actual handler paths are
`<path_prefix>/sse` (GET) and `<path_prefix>/message` (POST). See §4.4.

### 3.6 `anthropic`

Optional. When present, enables the `POST /v1/messages` route.

```yaml
anthropic:
  upstream: https://api.anthropic.com    # required; root of the upstream
  auth:                                  # optional
    type: bearer
    token: <string> | token_file: <path>
```

Validation:

- `upstream` required.
- If `auth` present, validated the same way as provider auth.

---

## 4. HTTP surface

The handler is built in `internal/server/server.go` via `http.ServeMux`,
wrapped by `withRequestID` middleware (see §7.1).

### 4.1 OpenAI frontend (`/v1/...`)

| Method | Path | Source | Purpose |
|--------|------|--------|---------|
| POST   | `/v1/chat/completions` | `handleChatCompletions` | OpenAI chat completion, streaming + non-streaming |
| POST   | `/v1/embeddings`       | `handleEmbeddings`      | OpenAI embeddings |
| GET    | `/v1/models`           | `handleModels`          | Lists all canonical model names + aliases |

#### Request handling (`/v1/chat/completions`, `/v1/embeddings`)

1. **Read body** — bounded by `maxRequestBytes = 8 MiB` (`io.MaxBytesReader`).
2. **Peek** the `model` and `stream` fields. Missing/invalid → `400`.
3. **Resolve** the model through `aliases → models → providers`. Unknown →
   `404`.
4. **Build chain** = `[resolved_model, fallback[0], fallback[1], ...]`.
5. For each hop in the chain:
   - Rewrite the body's top-level `model` field to the hop's
     `upstream_model`. Field order in the body is preserved (`rewriteModelField`
     uses a streaming decoder for this; the request body's `model` value is
     the only field mutated).
   - For embedding requests, inject `dimensions` from the model's
     `default_embed_dimensions` if (a) the model has a default and (b) the
     client didn't specify `dimensions`.
   - POST to `<base_url>/chat/completions` or `<base_url>/embeddings`.
   - Apply the provider's authenticator (see §6) — adds `Authorization: Bearer ...`.
   - On upstream 5xx, if a fallback remains, drain & retry the next hop.
     **Exception**: 5xx bodies that look like wrapped 4xx client errors are
     treated as non-retryable and surfaced as 400 (detected via JSON envelope
     `error.type == "invalid_request_error"` or `error.message` prefix
     `"Invalid '"` — the `openai-oauth` proxy used to wrap 4xx as 500).
   - On success (or non-retryable failure), copy hop-by-hop-filtered headers,
     write status, stream/copy body.
6. **Streaming**: when `stream: true` and upstream returned `200`, body is
   piped via `streamCopy` (4 KiB reads, `Flush` after every write).
7. **All hops failed**: respond `502` with `{"error":{"message":"all upstreams failed: ...","type":"tiny_llm_gate_error"}}`.

Headers stripped on the upstream-response path: `Connection`, `Keep-Alive`,
`Proxy-Authenticate`, `Proxy-Authorization`, `Te`, `Trailer`,
`Transfer-Encoding`, `Upgrade`. All other headers are forwarded to the
client.

#### `GET /v1/models`

Returns:

```json
{
  "object": "list",
  "data": [
    {"id": "<name>", "object": "model", "created": 0, "owned_by": "tiny-llm-gate"},
    ...
  ]
}
```

Lists every `models.<name>` key **and** every `aliases.<name>` key. Order is
map-iteration order (not stable).

### 4.2 Anthropic frontend (`/v1/messages`)

Registered **only when** `anthropic` is configured. Otherwise clients get
`404`.

| Method | Path | Source |
|--------|------|--------|
| POST   | `/v1/messages` | `handleAnthropicMessages` |

This is a **pure pass-through proxy** — no body translation. The body is
forwarded byte-identical to the configured upstream. The model field is
**not** rewritten or routed through the resolver (this is intentional:
Claude Code uses real Anthropic model names).

Request flow:

1. Read body bounded by 8 MiB.
2. Peek `model` and `stream` for logging only.
3. Build upstream URL: `cfg.Anthropic.Upstream + r.URL.Path` (+ raw query).
4. Copy all client headers to the upstream request. Then:
   - Delete `Authorization` (the gateway replaces it).
   - **Delete `x-api-key`** — Anthropic prefers `x-api-key` over
     `Authorization` when both are present, so leaving a client-supplied one
     would defeat the auth swap (`81d8cd5 fix(anthropic): also strip
     incoming x-api-key`). Clients like `pi-coding-agent` send `x-api-key` by
     default; Claude Code uses `Authorization: Bearer ...`, so this is a
     no-op for it but load-bearing for the former.
   - Set `Accept-Encoding: identity` to avoid the upstream sending us a
     gzipped body that we'd then relay compressed.
   - Apply `s.anthropicAuth` (which is the `FileBearer` when configured with
     `token_file`).
5. POST upstream with the original body.
6. Copy upstream response headers (with hop-by-hop filter) and status to the
   client.
7. Stream the body via `streamCopy` if `stream=true` or response
   `Content-Type` starts with `text/event-stream`. Otherwise plain
   `io.Copy`.

No fallback chain is run for Anthropic — single upstream, single attempt.

### 4.3 Gemini frontend (`/v1beta/...`)

Gemini uses a `:action` suffix on the final URL segment. The mux registers
the prefix `POST /v1beta/models/` and dispatches via `routeGemini` by
splitting on the last `:`.

| Method | Path pattern | Source | Purpose |
|--------|--------------|--------|---------|
| GET    | `/v1beta/models` | `handleGeminiModels` | Lists models in Gemini's `models/<name>` form |
| POST   | `/v1beta/models/{m}:generateContent` | `handleGenerateContent` | Non-streaming chat |
| POST   | `/v1beta/models/{m}:streamGenerateContent` | `handleStreamGenerateContent` | Streaming chat |
| POST   | `/v1beta/models/{m}:embedContent` | `handleEmbedContent` | Single text embedding |
| POST   | `/v1beta/models/{m}:batchEmbedContents` | `handleBatchEmbedContents` | Batch text embeddings |

The model is extracted from the URL via `extractGeminiModel` — splits on the
**last** `:` so model names containing colons (`gemma4:e4b`) round-trip
correctly.

#### `:generateContent` / `:streamGenerateContent`

Flow:

1. Parse model from URL. Read body bounded by 8 MiB.
2. Decode body as `gemini.ChatRequest` (see §4.3.1 for shape coverage).
3. Resolve the client model through aliases. Build the same fallback chain
   as the OpenAI frontend.
4. For each hop:
   - Translate the Gemini request to an OpenAI chat request via
     `gemini.ChatRequestToOpenAI`.
   - POST `<base_url>/chat/completions` with auth.
   - On 5xx with remaining fallbacks, drain and continue.
   - On any other non-200, surface upstream status + first 400 bytes of body
     as `502 Bad Gateway`.
5. Non-streaming: decode upstream JSON, translate via
   `gemini.ChatResponseFromOpenAI`, return as `application/json`.
6. Streaming: pipe upstream SSE (`data: {...}` lines) into either
   - **newline-delimited JSON** (default, used by `@google/genai`), or
   - **SSE** (`Content-Type: text/event-stream`, `data: {...}\r\n\r\n`) when
     the request has query `?alt=sse` (used by the Vercel `@ai-sdk/google`
     SDK).

Streaming tool-call handling: OpenAI streams tool calls incrementally (name
in the first chunk, argument fragments thereafter) but Gemini expects
complete `FunctionCall` parts in one chunk. The handler uses a
`ToolCallAccumulator` to buffer deltas and flushes them as a single Gemini
chunk on `finish_reason`. Text deltas are emitted immediately.

The finish-reason chunk for text-only streams is also emitted (as a chunk
with empty candidate parts but a `finishReason`) — dropping it makes clients
hang waiting for completion.

#### `:embedContent` / `:batchEmbedContents`

1. Decode Gemini embed request.
2. Resolve the client model.
3. Translate to OpenAI `/embeddings`:
   - `embedContent`: single input, `outputDimensionality` → `dimensions`.
   - `batchEmbedContents`: array of inputs concatenated into one OpenAI call;
     `dimensions` taken from the first sub-request that sets
     `outputDimensionality`.
4. If the request didn't specify dimensions and the model has
   `default_embed_dimensions`, inject it.
5. POST `<base_url>/embeddings` with auth.
6. Translate response:
   - `embedContent`: `{"embedding":{"values":[...]}}`.
   - `batchEmbedContents`: `{"embeddings":[{"values":[...]}, ...]}` —
     ordered by OpenAI `index` field (with bounds-check guard).

Embedding requests do **not** run the fallback chain — a single attempt only.

#### 4.3.1 Gemini wire-format coverage

Implemented in `internal/gemini/translate.go`. Today's translator handles:

- **Text content** — `Content.Parts[].Text` joined with `\n` when multiple
  parts are present.
- **System prompt** — `systemInstruction` → OpenAI `system` message.
- **Roles** — Gemini `"user"`/`"model"` → OpenAI `"user"`/`"assistant"`;
  empty role defaults to `"user"`; `"system"` passes through.
- **Generation config** — `temperature`, `topP`, `maxOutputTokens`,
  `stopSequences` translate to OpenAI fields. `topK` is **dropped** (OpenAI
  has no equivalent).
- **Function calling** —
  - `tools[].functionDeclarations[]` → OpenAI `tools[].function`.
  - Gemini `functionCall` parts → OpenAI `assistant` message with
    `tool_calls[]`. Synthetic IDs are minted as `gemini_call_<n>` so
    subsequent `functionResponse` parts can be matched back.
  - Gemini `functionResponse` parts → OpenAI `tool` messages with
    `tool_call_id` matched against the pending-ID queue (FIFO by function
    name). Empty `response` becomes `"{}"`.
- **Tool-call streaming** — see ToolCallAccumulator behaviour above.
- **Finish reasons** —
  | OpenAI | Gemini |
  |--------|--------|
  | `stop` | `STOP` |
  | `tool_calls` | `STOP` |
  | `length` | `MAX_TOKENS` |
  | `content_filter` | `SAFETY` |
  | other | uppercased verbatim |
- **Usage metadata** — `prompt_tokens`/`completion_tokens`/`total_tokens` ↔
  `promptTokenCount`/`candidatesTokenCount`/`totalTokenCount`.

Out of scope (deliberately, per package comment):

- Multimodal parts (images, audio, video).
- Safety settings.
- `topK`.
- Anything beyond text and function calling.

### 4.4 MCP bridges

When `mcp_bridges.<name>` is set, the bridge registers two routes per entry:

| Method | Path | Source |
|--------|------|--------|
| GET    | `<path_prefix>/sse` | `Bridge.HandleSSE` |
| POST   | `<path_prefix>/message` | `Bridge.HandleMessage` |

Transport: `sse` frontend ↔ `streamable_http` backend. Messages are
**opaque JSON-RPC bytes** — the bridge does not parse or interpret them.

The reference deployment has one bridge named `affine` with prefix
`/mcp/affine`, used to expose AFFiNE's MCP StreamableHTTP server as SSE on
the tailnet.

#### `GET <prefix>/sse`

1. Generate a 16-hex-char session ID via `crypto/rand`.
2. Open a new `backendConn` for this session.
3. Set headers `Content-Type: text/event-stream`,
   `Cache-Control: no-cache`, `Connection: keep-alive`.
4. Emit an initial `event: endpoint\ndata: <path_prefix>/message?sessionId=<id>\n\n`.
5. Stream loop: forward every message arriving on the session's `outCh` as
   `data: <bytes>\n\n` until the client disconnects, the bridge context is
   cancelled, or the upstream closes.

#### `POST <prefix>/message?sessionId=<id>`

1. `sessionId` query param required.
2. Read body bounded by 1 MiB.
3. Look up the session; 404 on unknown.
4. Hand the body off to a goroutine that POSTs it to the bridge's
   `upstream_url` with `Content-Type: application/json` and
   `Accept: application/json, text/event-stream` (and any bridge auth).
5. Respond `202 Accepted` immediately; the upstream response is asynchronously
   forwarded onto the session's `outCh` and out via the SSE stream.

#### Backend (`backendConn`)

- Tracks `Mcp-Session-Id` learned from the first upstream response and
  echoes it on subsequent requests.
- Detects upstream content type:
  - `text/event-stream` → parses SSE (`data:` lines, blank-line dispatch,
    multi-line concatenation, 1 MiB scanner limit) and forwards each event's
    data as one outCh message.
  - Anything else → reads up to 1 MiB and forwards as one message.
- Optional bearer auth (`Authorization: Bearer ...`) per bridge.
- MCP bridges use a separate `http.Client` from the LLM frontends with
  longer timeouts:
  - `MaxIdleConns 8`, `MaxIdleConnsPerHost 2`, `IdleConnTimeout 90s`,
    `ResponseHeaderTimeout 120s`, `ForceAttemptHTTP2 false`.
- Session channel buffer: 64 messages (`sessionChanCap`).

### 4.5 Health & readiness

| Method | Path | Body |
|--------|------|------|
| GET    | `/health` | `{"status":"ok"}` |
| GET    | `/ready`  | `{"status":"ready"}` |

`/ready` currently == `/health` (config is loaded by the time we're serving).
Future upstream probes are not implemented.

There is **no `/metrics` endpoint** — Prometheus instrumentation is an
explicit non-goal (see README "Non-goals").

---

## 5. Routing & resolver

### 5.1 Resolver structure

`internal/resolve.Resolver` holds a pointer to the validated `*Config`. Its
single public method is `Resolve(name string) (*Resolved, error)`.

`Resolved` carries:

- `ModelName` — canonical model after alias resolution.
- `UpstreamModel` — what to send upstream.
- `ProviderName` / `Provider` — the resolved provider.
- `Fallback []string` — list of canonical model names to try in order.
- `DefaultEmbedDimensions *int` — optional dimension hint.

### 5.2 Alias chain resolution

1. Walk `aliases` recursively while tracking visited names.
2. Cycle detection: revisiting a name returns `alias cycle involving "<name>"`.
3. Once the walk lands on a name that is **not** an alias, look it up in
   `models`. Unknown → `unknown model "<name>"`.
4. Validate the model's provider still exists (defensive — startup validation
   already catches this, but the resolver is paranoid because configs may
   change in the future).

### 5.3 Fallback semantics

- Each model's `fallback` is a list of canonical model names (not aliases —
  validation rejects fallback to an alias).
- Each hop in the fallback chain is **re-resolved** so it can use its own
  provider and `upstream_model`.
- A fallback fires **only** when the upstream returns a 5xx and the
  `isWrappedClientError` heuristic rules out a wrapped 4xx (see §4.1).
- Transport-level errors (`s.client.Do` returns a non-context-canceled error)
  surface as `502` immediately for the Gemini frontend; for the OpenAI
  frontend they also trigger a fallback hop, since `sendUpstream` returns
  `done=false, err=...` and the outer loop tries the next hop.
- Once **any byte** is written to the client (status code or body), no
  further fallback is attempted. `sendUpstream` returns `done=true` after
  the first `WriteHeader`.

### 5.4 Model list

`Resolver.ListModels()` returns every key from both `models` and `aliases`,
unsorted. Used by `/v1/models` and `/v1beta/models`.

---

## 6. Authentication

### 6.1 Inbound

**The gateway does not authenticate inbound requests.** All exposed routes
are open. The README's claim that "the gateway refuses unauthenticated
requests by design" describes a *non-existent* inbound auth layer — it is
aspirational. Today, security relies on:

- Binding to `127.0.0.1` (default Listen).
- Tailscale Serve providing tailnet-only HTTPS termination upstream.
- The systemd sandbox in the NixOS module.

If you need client auth, add it externally (e.g. via Tailscale Aperture, an
Nginx auth_request, or a separate reverse-proxy).

### 6.2 Outbound (`internal/auth`)

Two `Authenticator` implementations, both setting
`Authorization: Bearer <token>`:

#### `Bearer{Token}` — fixed token

Set at config load. Never re-read.

#### `FileBearer{Path}` — re-read on every request

```go
func (b FileBearer) Apply(_ context.Context, req *http.Request) error {
    data, err := os.ReadFile(b.Path)
    if err != nil {
        return fmt.Errorf("read token_file %s: %w", b.Path, err)
    }
    token := strings.TrimSpace(string(data))
    if token != "" {
        req.Header.Set("Authorization", "Bearer "+token)
    }
    return nil
}
```

**Critical invariant**: `FileBearer.Apply` re-reads the file from disk on
**every** outbound request. This is established and tested in
`internal/auth/auth_test.go::TestFileBearerRereadsPerRequest` and was
introduced by commit `941c69a feat(auth): FileBearer re-reads token_file on
every request`. The intent is to let an external sidecar (e.g. a
`claude-remote-control` OAuth refresher) rotate the token without restarting
`tiny-llm-gate`. Do **not** add caching here without a corresponding rotation
notification path.

`Build()` reads the file once at startup to fail fast on a missing path; the
in-memory `FileBearer` carries only the path, not the cached token.

If the file disappears between the startup check and an `Apply()` call,
that `Apply()` returns an error and the request fails — it does **not**
silently send an empty bearer.

### 6.3 Per-provider authenticator map

`server.New` builds `auths map[string]auth.Authenticator` keyed by provider
name. Providers without auth get no entry — `sendUpstream` checks
`auths[hop.ProviderName]` with `ok` and only applies it when present.

The Anthropic frontend has its own dedicated `anthropicAuth` field on the
server, built from `cfg.Anthropic.Auth` (independent of the provider map).

MCP bridges build their own `Authenticator` from their per-bridge `auth`.

### 6.4 Header stripping

The Anthropic handler strips both `Authorization` **and** `x-api-key` from
the inbound request before applying the gateway's auth. Stripping `x-api-key`
is non-obvious but required — Anthropic prefers it over `Authorization`
when both are present, so a leftover client `x-api-key` would defeat the
auth swap entirely. See commit `81d8cd5 fix(anthropic): also strip incoming
x-api-key`.

It also forces `Accept-Encoding: identity` so the upstream doesn't send
gzip that the relay would otherwise need to decompress (or re-encode).

The OpenAI and Gemini handlers build outbound requests from scratch (no
header copy from the client request), so they don't need explicit
stripping — only `Content-Type` and the gateway's `Authorization` end up on
the wire.

---

## 7. Observability

### 7.1 Request IDs

`withRequestID` middleware wraps the mux. For every request:

- Reads `X-Request-ID` from the client; if absent, generates a fresh
  6-byte hex ID (12 chars).
- Echoes it in the response header.
- Stashes it in the request context under `ctxKeyRequestID`.

Handlers retrieve it via `requestID(ctx)` and include it in every structured
log line.

### 7.2 Structured logging

`slog.NewJSONHandler` on stderr, level Info. Standard fields per served
request:

- OpenAI frontend (`s.logger.Info("served", ...)`):
  - `request_id`, `client_model`, `resolved_model`, `provider`, `stream`,
    `fallback_index`, `latency_ms`.
- Anthropic frontend:
  - `request_id`, `frontend: "anthropic"`, `model`, `stream`, `status`,
    `latency_ms`.
- Gemini frontend:
  - `request_id`, `frontend: "gemini"`, `client_model`, `resolved_model`,
    `provider`, `stream`, `fallback_index`, `latency_ms`.

Errors (`logger.Warn` / `logger.Error`) include `request_id` and a `err`
field.

MCP bridge logs are prefixed by `mcp_bridge=<name>` and, within sessions,
`session=<id>`.

### 7.3 Metrics

None. Prometheus `/metrics` is an explicit non-goal.

---

## 8. Memory & networking discipline

- Binary built with `-ldflags="-s -w"` and `CGO_ENABLED=0`. Stripped binary
  ~6.5 MiB.
- Recommended `GOMEMLIMIT=20MiB`, `GOGC=50`, `MemoryMax=30M` (set by the
  NixOS module).
- HTTP client (LLM upstreams): `MaxIdleConns 16`, `MaxIdleConnsPerHost 4`,
  `IdleConnTimeout 60s`, `TLSHandshakeTimeout 10s`,
  `ResponseHeaderTimeout 30s`, `ExpectContinueTimeout 1s`,
  `ForceAttemptHTTP2 false`. No client-level `Timeout` so streaming requests
  can run for minutes.
- HTTP client (MCP bridges): looser timeouts (90s idle, 120s response
  header), smaller idle pool (`8 / 2`).
- Request body cap: **8 MiB** (`maxRequestBytes`). MCP message bodies are
  capped at **1 MiB**.
- Streaming reads use a 4 KiB buffer with explicit `Flush` per write.
- Gemini non-streaming upstream response read is limited to 4 MiB; embedding
  responses to 8 MiB.

---

## 9. NixOS module (`nixos-module.nix`)

`services.tiny-llm-gate` options:

| Option | Type | Default | Purpose |
|--------|------|---------|---------|
| `enable` | bool | false | toggle |
| `package` | package | — | the `tiny-llm-gate` derivation |
| `settings` | attrset (yaml format) | `{}` | inline config; serialized to YAML |
| `configFile` | nullable path | null | path to a YAML config (overrides `settings`) |
| `memoryMax` | str | `"30M"` | systemd `MemoryMax=` |
| `goMemLimit` | str | `"20MiB"` | `GOMEMLIMIT` env |
| `secretPaths` | list of str | `[]` | systemd `ReadOnlyPaths=` so the unit can read agenix secrets |

The unit ships with: `DynamicUser`, `NoNewPrivileges`,
`ProtectSystem=strict`, `ProtectHome`, `PrivateTmp`, `PrivateDevices`,
`ProtectKernelTunables`, `ProtectKernelModules`, `ProtectControlGroups`,
`RestrictSUIDSGID`, `LockPersonality`, `RestrictRealtime`,
`SystemCallArchitectures=native`, `CapabilityBoundingSet=""`,
`Restart=on-failure`, `RestartSec=5s`.

---

## 10. Failure modes

| Failure | Behaviour |
|---------|-----------|
| Config file missing/unparseable | exit 1 at startup with `slog` error |
| `auth.token_file` unreadable at startup | exit 1 |
| `auth.token_file` deleted post-startup | `Apply()` returns error → 502 on next request |
| Unknown `model` in inbound request | 404 `{"error":{"message":"unknown model ..."}}` |
| Alias cycle | 404 (resolver error surfaces via the same path) |
| Inbound body > 8 MiB | 400 from `http.MaxBytesReader` |
| Inbound body invalid JSON | 400 `"invalid JSON body"` |
| Inbound body missing `model` | 400 `"missing 'model' field"` (OpenAI/Gemini) |
| Upstream 5xx, fallback available | retry next hop |
| Upstream 5xx wrapped client error (`type=invalid_request_error` or message starting `Invalid '`) | treat as non-retryable, surface to client as 400 |
| Upstream transport error (OpenAI frontend) | `502 "all upstreams failed: ..."` after exhausting chain |
| Upstream transport error (Anthropic frontend) | `502 "upstream transport: ..."` |
| Upstream non-200, non-5xx (Gemini frontend) | `502` with upstream status + first 400 bytes of body |
| Client disconnects mid-stream | handler returns silently; `context.Canceled` is swallowed |
| MCP `sessionId` missing or unknown | 400 / 404 from the SSE handler |

All gateway-originated error bodies use the shape:

```json
{"error":{"message":"<text>","type":"tiny_llm_gate_error"}}
```

Upstream error bodies (after `WriteHeader`) are forwarded byte-for-byte and
keep whatever shape the upstream chose.

---

## 11. Invariants

These are properties a refactor must preserve. Each is load-bearing.

1. **`FileBearer.Apply` re-reads the file on every request.** Caching would
   break OAuth-token rotation by an external sidecar
   (`claude-remote-control`). Established by commit `941c69a` and tested in
   `auth_test.go::TestFileBearerRereadsPerRequest`. The test must stay
   green.
2. **Anthropic handler strips both `Authorization` *and* `x-api-key`.**
   Without the `x-api-key` strip, clients like `pi-coding-agent` defeat the
   gateway's auth swap. Established by commit `81d8cd5`.
3. **Anthropic handler does not mutate the request body.** The point of the
   pass-through is verbatim forwarding for the observability layer
   (Aperture) sitting upstream. Don't add body parsing or model rewriting
   on this path.
4. **Fallback only fires before bytes are written to the client.**
   `sendUpstream` returns `done=true` once `WriteHeader` runs; the caller
   must not try another upstream after that.
5. **Body rewriter preserves field order.** `rewriteModelField` uses a
   token-level streaming decoder rather than `map[string]any` so request
   bodies stay byte-stable for observability sniffers (Aperture).
6. **Config validation rejects unknown YAML fields** (`KnownFields(true)`).
   Add new fields to the struct *before* shipping a config that uses them.
7. **Inbound body cap is 8 MiB** and MCP body cap is 1 MiB. Both are
   conservative; raising them needs a corresponding memory-budget review.
8. **`GET /v1/models` and `GET /v1beta/models` enumerate aliases too.**
   AFFiNE's `CopilotProviderFactory` selects providers by checking whether
   the requested model id appears in `onlineModelList`; dropping aliases
   here breaks routing.
9. **Streaming uses HTTP/1.1 and explicit `Flush` per write.** HTTP/2's
   frame buffering and the lack of an explicit flush were both observed to
   produce stalls; the current setup is deliberate.

---

## 12. Non-goals

Listed in README "Non-goals", reaffirmed here:

- USD cost tracking.
- Prometheus `/metrics`.
- Multi-tenant billing or quotas.
- Request rate limiting.
- Tokenizer-based features (counting, truncation).
- Multimodal Gemini parts (images, audio, video).
- Gemini safety settings.
- Anthropic body translation or model rewriting.
- Inbound client authentication. (The README's "refuses unauthenticated
  requests" claim is incorrect today — see §6.1.)
- SIGHUP hot-reload (roadmap, not implemented).

---

## 13. File map

```
cmd/tiny-llm-gate/main.go             # entry point, flag parsing, signal handling
internal/config/config.go             # YAML schema, validation, Provider.EffectiveAuth
internal/auth/auth.go                 # Authenticator interface, Bearer, FileBearer
internal/resolve/resolve.go           # alias chain + fallback resolution
internal/server/server.go             # routing, http.Client setup, MCP wiring
internal/server/middleware.go         # withRequestID
internal/server/openai.go             # /v1/chat/completions, /v1/embeddings, /v1/models
internal/server/anthropic.go          # /v1/messages pass-through
internal/server/gemini.go             # /v1beta/models/... dispatcher + handlers
internal/gemini/types.go              # Gemini wire format
internal/gemini/translate.go          # Gemini ↔ OpenAI translation
internal/mcp/bridge.go                # MCP bridge lifecycle
internal/mcp/sse.go                   # SSE frontend handlers
internal/mcp/streamhttp.go            # StreamableHTTP backend
internal/mcp/session.go               # per-connection session state
flake.nix                             # Nix flake (build + dev shell)
nixos-module.nix                      # services.tiny-llm-gate
testdata/example-config.yaml          # full annotated config example
```

---

## 14. Versioning

`Version` is injected at build time via Go ldflags. The flake derives the
version from the git short rev (`0.3.3-<shortRev>`, or `-dirty` /
`-dev` suffixes). The current development version is **0.3.3-dev**;
phase status per `ROADMAP.md` is **v0.3.0 features shipped, phase 5
(LiteLLM cutover) underway**.
