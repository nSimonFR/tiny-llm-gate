# tiny-llm-gate — Specification

This document specifies the `tiny-llm-gate` contract: what it does **on the
wire** and what behaviors any implementation must preserve. It is intentionally
language- and architecture-agnostic. A rewrite in any language, or split into
any number of processes, must satisfy everything in §1–§12 to be conformant.
Pointers into the current source tree live in the appendix (§13).

Where the README diverges from this document, this document wins.

---

## 1. Purpose

`tiny-llm-gate` is an LLM gateway that accepts requests in **OpenAI,
Anthropic, and Gemini wire formats** and proxies — translating where necessary
— to upstream LLM providers. Alongside the LLM surfaces, it exposes **MCP
transport bridges** that translate one MCP transport into another.

It is designed for memory-constrained hosts (Raspberry Pi 5 with 4 GB RAM is
the reference target) as a drop-in replacement for [LiteLLM] on small
deployments.

The four wire surfaces are:

1. **OpenAI-compatible `/v1/...`** — used by clients that speak OpenAI
   (AFFiNE, Open WebUI, PicoClaw, any `openai`-flavoured SDK).
2. **Anthropic-compatible `/v1/messages`** — pass-through proxy used by Claude
   Code → an observability layer (Aperture in the reference deployment). The
   gateway does not interpret the body; it strips client auth and applies its
   own.
3. **Gemini-compatible `/v1beta/...`** — translation frontend used by
   AFFiNE's Gemini provider (text embeddings primarily, also `generateContent`
   / `streamGenerateContent`). Requests are translated to OpenAI shape against
   a single upstream (typically Ollama).
4. **MCP transport bridges** — generic SSE-frontend, StreamableHTTP-backend
   protocol bridges. Used to expose AFFiNE's StreamableHTTP MCP server as SSE
   on the tailnet for older MCP clients (this is the `/mcp/affine` route in
   the reference deployment).

[LiteLLM]: https://github.com/BerriAI/litellm

---

## 2. Process model

- Listens on a single HTTP listener for all four surfaces. Default
  `127.0.0.1:4001`; configurable via the `listen` config field.
- HTTP server tuning: read-header timeout 10s; **no write timeout** —
  streaming responses can be arbitrarily long, and a write timeout would
  truncate them.
- Structured JSON logs to standard error at Info level.
- Graceful shutdown on `SIGINT` / `SIGTERM`: ~10-second drain, MCP-bridge
  shutdown, idle-connection close.
- **No hot-reload.** Config changes require a full restart. There is no
  SIGHUP handler.

---

## 3. Configuration

### 3.1 File format

YAML. Decoded in **strict** mode — unknown keys cause startup to fail.
Validation runs at startup; the process exits non-zero on any validation
error.

Top-level fields:

| Field | Type | Required | Purpose |
|---|---|---|---|
| `listen` | string `host:port` | no (default `127.0.0.1:4001`) | inbound bind |
| `providers` | map of `<name>` → provider | yes, ≥1 | upstream LLM endpoints |
| `models` | map of `<name>` → model | yes, ≥1 | canonical model → routing decision |
| `aliases` | map of `<name>` → string | no | client-facing name → another model or alias |
| `mcp_bridges` | map of `<name>` → bridge | no | MCP transport bridges |
| `anthropic` | object | no | enables `POST /v1/messages` when present |

### 3.2 `providers`

Map of provider name → upstream LLM endpoint.

```yaml
providers:
  <name>:
    type: openai          # required; "openai" or "codex"
    base_url: <url>       # required; e.g. http://host:port/v1
    api_key: <string>     # legacy shorthand; mutually exclusive with `auth`
    auth:                 # optional; preferred
      type: bearer        # "bearer" or "oauth_chatgpt"
      token: <string>     # bearer: mutually exclusive with token_file
      token_file: <path>  # bearer: mutually exclusive with token
      file: <path>        # oauth_chatgpt: credentials file (refresh_token)
      issuer: <url>       # oauth_chatgpt: optional OAuth issuer override
      client_id: <string> # oauth_chatgpt: optional client id override
```

Validation:

- `type` must be `"openai"` or `"codex"`. Any other value rejected.
- `base_url` required.
- `api_key` and `auth` cannot both be set.
- For `auth.type=bearer`: exactly one of `token` or `token_file` must be set.
- For `token_file`: the file must be readable at startup (read once to fail
  fast on misconfig; the runtime re-read invariant is in §6.2.1).
- For `auth.type=oauth_chatgpt`: `file` is required (see §6.2.2).

When neither `api_key` nor `auth` is set, upstream requests are sent without an
`Authorization` header.

**`type: codex`** targets the ChatGPT/Codex Responses API
(`POST <base_url>/responses`) instead of a plain OpenAI-compatible server. The
gateway translates each OpenAI `/v1/chat/completions` request into a Codex
Responses request and translates the Codex SSE response back to OpenAI chat
shape (see §4.1.1). A codex provider participates in the same model/alias/
fallback machinery as any other provider; it only differs in the wire
translation and in requiring `auth.type: oauth_chatgpt`. Embeddings are not
supported by codex providers.

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
    reasoning_effort: <string>         # optional; injected into chat/completions
                                       #   requests that don't specify one
```

Validation:

- `provider` must reference a known provider.
- `upstream_model` required.
- Each fallback entry must be a known **canonical model name** (not an alias).
- A model cannot list itself in `fallback`.

`default_embed_dimensions` is critical for Matryoshka-capable embedding models
(e.g. `qwen3-embedding:8b`) — without it the model returns its full native
dimension, breaking callers that expect a smaller fixed width (e.g. AFFiNE's
pgvector column).

`reasoning_effort`, when set, is injected into `POST /chat/completions` request
bodies that lack the field (client-supplied values always win). It exists to
pin a fixed effort level on hybrid-reasoning upstreams. In particular Ollama
honors `reasoning_effort: "none"` on its OpenAI-compatible `/v1` endpoint to
disable a model's thinking, whereas the native `think` field is ignored there —
so this is the only in-band way to force no-think for a Qwen model via the
OpenAI surface. Injection applies to the chat path only, never embeddings.

### 3.4 `aliases`

Map of client-facing model name → another model name.

```yaml
aliases:
  "gpt-4o-mini": "gemma3:4b"
  "openai/gemma4:e4b": "gemma4:e4b"
```

- Targets may themselves be aliases (chains are supported; **cycles MUST be
  detected and rejected** — see §5.2).
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

All routes are mounted on a single HTTP server. Every request is tagged with
a request ID (see §7.1).

### 4.1 OpenAI frontend (`/v1/...`)

| Method | Path | Purpose |
|--------|------|---------|
| POST   | `/v1/chat/completions` | OpenAI chat completion, streaming + non-streaming |
| POST   | `/v1/embeddings`       | OpenAI embeddings |
| GET    | `/v1/models`           | Lists all canonical model names + aliases |

#### Request handling (`/v1/chat/completions`, `/v1/embeddings`)

1. **Read body** — bounded by **8 MiB**.
2. **Inspect** the `model` and `stream` fields. Missing/invalid → `400`.
3. **Resolve** the model through `aliases → models → providers`. Unknown →
   `404`.
4. **Build chain** = `[resolved_model, fallback[0], fallback[1], ...]`.
5. For each hop in the chain:
   - Rewrite the body's top-level `model` field to the hop's
     `upstream_model`. **Field order in the body MUST be preserved** — only
     the `model` value changes. (See invariant §11.5 — observability sniffers
     downstream rely on byte-stable bodies.)
   - For embedding requests, inject `dimensions` from the model's
     `default_embed_dimensions` if (a) the model has a default and (b) the
     client did not specify `dimensions`.
   - POST to `<base_url>/chat/completions` or `<base_url>/embeddings`.
   - Apply the provider's outbound auth (see §6) — typically adds
     `Authorization: Bearer ...`.
   - On upstream 5xx, if a fallback remains, drain & retry the next hop.
     **Exception**: 5xx bodies that look like wrapped 4xx client errors are
     treated as non-retryable and surfaced as 400. Detection: JSON envelope
     `error.type == "invalid_request_error"`, or `error.message` prefix
     `"Invalid '"`. (Some upstream proxies, e.g. `openai-oauth`, wrap 4xx as
     5xx; retrying makes the same call fail the same way.)
   - On success (or non-retryable failure), copy hop-by-hop-filtered headers
     to the client, write status, stream/copy body.
6. **Streaming**: when `stream: true` and upstream returned `200`, body is
   piped to the client with explicit flushes (see §4.6).
7. **All hops failed**: respond `502` with
   `{"error":{"message":"all upstreams failed: ...","type":"tiny_llm_gate_error"}}`.

Headers stripped on the upstream-response path: `Connection`, `Keep-Alive`,
`Proxy-Authenticate`, `Proxy-Authorization`, `Te`, `Trailer`,
`Transfer-Encoding`, `Upgrade`. All other headers are forwarded to the
client.

#### 4.1.1 Codex provider translation

When a chain hop resolves to a `type: codex` provider, the hop does NOT
byte-forward. Instead:

1. The OpenAI chat body is translated to a Codex Responses request:
   - `system`/`developer` messages are concatenated into `instructions`
     (default `"You are a helpful assistant."` when none).
   - Other messages become `input[]` items. Assistant `tool_calls` become
     `{type:"function_call", call_id, name, arguments}`; `tool`/`function`
     results become `{type:"function_call_output", call_id, output}`; user
     content preserves images as `input_image` parts.
   - `tools` → Codex function tools (object schemas get an injected empty
     `properties` when missing); `tool_choice` string passes through, and
     `{type:function, function:{name}}` flattens to `{type:function, name}`.
   - `reasoning_effort` → `reasoning:{effort, summary:"auto"}`;
     `response_format` → `text.format`.
   - `stream:true` and `store:false` are always sent upstream.
2. The request is POSTed to `<base_url>/responses` with the Codex desktop
   fingerprint headers (`originator`, Codex Desktop `User-Agent`,
   `x-codex-installation-id`, etc.) and the `oauth_chatgpt` auth
   (`Authorization: Bearer` + `ChatGPT-Account-Id`).
3. The Codex SSE response is translated back to OpenAI shape:
   - `response.output_text.delta` → content deltas.
   - `response.output_item.added`(function_call) + `.function_call_arguments.
     delta`/`.done` → `tool_calls` (arguments streamed incrementally). An
     item-id-keyed map resolves argument events that reference the output-item
     id rather than the call_id.
   - `response.completed` `usage` → OpenAI `usage`
     (`input_tokens`→`prompt_tokens`, `output_tokens`→`completion_tokens`).
   - `error`/`response.failed` → a gateway error (before any bytes are
     written, so fallback still applies).
   - Streaming clients receive OpenAI `chat.completion.chunk` SSE terminated
     by `data: [DONE]`; non-streaming clients receive a single aggregated
     `chat.completion` JSON object.

Fallback semantics are unchanged: a codex hop that returns 5xx (or a transport
error) before writing bytes falls through to the next chain entry.

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
not guaranteed stable.

### 4.2 Anthropic frontend (`/v1/messages`)

Registered **only when** `anthropic` is configured. Otherwise clients get
`404`.

| Method | Path |
|--------|------|
| POST   | `/v1/messages` |

This is a **pure pass-through proxy** — no body translation. The body is
forwarded byte-identical to the configured upstream. The model field is
**not** rewritten or routed through the resolver (this is intentional:
Claude Code uses real Anthropic model names; see invariant §11.3).

Request flow:

1. Read body bounded by 8 MiB.
2. Inspect `model` and `stream` for logging only.
3. Build upstream URL: `<anthropic.upstream> + <request path>` (+ raw query).
4. Copy all client headers to the upstream request. Then:
   - **Delete `Authorization`** — the gateway replaces it.
   - **Delete `x-api-key`** — Anthropic prefers `x-api-key` over
     `Authorization` when both are present, so a leftover client-supplied
     `x-api-key` would defeat the auth swap. Clients like `pi-coding-agent`
     send `x-api-key` by default; Claude Code uses `Authorization: Bearer ...`,
     so this is a no-op for Claude Code but load-bearing for the former.
     (See invariant §11.2.)
   - **Set `Accept-Encoding: identity`** — avoids the upstream sending a
     gzipped body that the gateway would have to relay compressed.
   - Apply the configured Anthropic outbound auth (see §6).
5. POST upstream with the original body.
6. Copy upstream response headers (with hop-by-hop filter) and status to the
   client.
7. Stream the body with per-write flushing if `stream=true` or response
   `Content-Type` starts with `text/event-stream`. Otherwise plain byte copy.

**No fallback chain is run for Anthropic** — single upstream, single attempt.

### 4.3 Gemini frontend (`/v1beta/...`)

Gemini uses a `:action` suffix on the final URL segment. Routing MUST split on
the **last** `:` so model names containing colons (`gemma4:e4b`) round-trip
correctly.

| Method | Path pattern | Purpose |
|--------|--------------|---------|
| GET    | `/v1beta/models` | Lists models in Gemini's `models/<name>` form |
| POST   | `/v1beta/models/{m}:generateContent` | Non-streaming chat |
| POST   | `/v1beta/models/{m}:streamGenerateContent` | Streaming chat |
| POST   | `/v1beta/models/{m}:embedContent` | Single text embedding |
| POST   | `/v1beta/models/{m}:batchEmbedContents` | Batch text embeddings |

#### `:generateContent` / `:streamGenerateContent`

Flow:

1. Parse model from URL. Read body bounded by 8 MiB.
2. Decode body as a Gemini chat request (see §4.3.1 for shape coverage).
3. Resolve the client model through aliases. Build the same fallback chain
   as the OpenAI frontend.
4. For each hop:
   - Translate the Gemini request to an OpenAI chat-completions request
     (see §4.3.1).
   - POST `<base_url>/chat/completions` with auth.
   - On 5xx with remaining fallbacks, drain and continue.
   - On any other non-200, surface upstream status + first 400 bytes of body
     as `502 Bad Gateway`.
5. Non-streaming: decode upstream JSON, translate to Gemini response shape
   (see §4.3.1), return as `application/json`.
6. Streaming: pipe upstream SSE (`data: {...}` lines) into either
   - **newline-delimited JSON** (default; used by `@google/genai`), or
   - **SSE** (`Content-Type: text/event-stream`, `data: {...}\r\n\r\n`) when
     the request has query `?alt=sse` (used by the Vercel `@ai-sdk/google`
     SDK).

**Streaming tool-call handling.** OpenAI emits tool calls incrementally
(name in the first chunk, argument fragments thereafter) but Gemini expects
complete `FunctionCall` parts in one chunk. The handler MUST buffer tool-call
deltas across upstream chunks and flush each tool call as a single Gemini
chunk on `finish_reason`. Text deltas are emitted immediately.

**Finish-reason chunk for text-only streams** MUST be emitted (as a chunk
with empty candidate parts but a `finishReason`). Dropping it causes clients
to hang waiting for completion.

#### `:embedContent` / `:batchEmbedContents`

1. Decode Gemini embed request.
2. Resolve the client model.
3. Translate to OpenAI `/embeddings`:
   - `embedContent`: single input, `outputDimensionality` → `dimensions`.
   - `batchEmbedContents`: array of inputs concatenated into one OpenAI call;
     `dimensions` taken from the first sub-request that sets
     `outputDimensionality`.
4. If the request did not specify dimensions and the model has
   `default_embed_dimensions`, inject it.
5. POST `<base_url>/embeddings` with auth.
6. Translate response:
   - `embedContent`: `{"embedding":{"values":[...]}}`.
   - `batchEmbedContents`: `{"embeddings":[{"values":[...]}, ...]}` —
     ordered by OpenAI `index` field (with bounds-check guard).

Embedding requests do **not** run the fallback chain — single attempt only.

#### 4.3.1 Gemini ↔ OpenAI wire-format translation

The Gemini ↔ OpenAI translator covers:

- **Text content** — `Content.Parts[].Text` joined with `\n` when multiple
  parts are present.
- **System prompt** — `systemInstruction` → OpenAI `system` message.
- **Role mapping**

  | Gemini | OpenAI |
  |--------|--------|
  | `"user"` | `"user"` |
  | `"model"` | `"assistant"` |
  | empty | defaults to `"user"` |
  | `"system"` | passes through |

- **Generation config** — `temperature`, `topP`, `maxOutputTokens`,
  `stopSequences` translate to the corresponding OpenAI fields. `topK` is
  **dropped** (OpenAI has no equivalent).
- **Function calling** —
  - `tools[].functionDeclarations[]` → OpenAI `tools[].function`.
  - Gemini `functionCall` parts → OpenAI `assistant` message with
    `tool_calls[]`. Synthetic IDs MUST be minted as `gemini_call_<n>` so
    subsequent `functionResponse` parts can be matched back.
  - Gemini `functionResponse` parts → OpenAI `tool` messages with
    `tool_call_id` matched against the pending-ID queue (FIFO by function
    name). Empty `response` becomes `"{}"`.
- **Tool-call streaming accumulation** — described above.
- **Finish-reason mapping**

  | OpenAI | Gemini |
  |--------|--------|
  | `stop` | `STOP` |
  | `tool_calls` | `STOP` |
  | `length` | `MAX_TOKENS` |
  | `content_filter` | `SAFETY` |
  | other | uppercased verbatim |

- **Usage metadata** — `prompt_tokens`/`completion_tokens`/`total_tokens` ↔
  `promptTokenCount`/`candidatesTokenCount`/`totalTokenCount`.

Out of scope (deliberately):

- Multimodal parts (images, audio, video).
- Safety settings.
- `topK`.
- Anything beyond text and function calling.

### 4.4 MCP bridges

When `mcp_bridges.<name>` is set, the bridge registers two routes per entry:

| Method | Path |
|--------|------|
| GET    | `<path_prefix>/sse` |
| POST   | `<path_prefix>/message` |

Transport: `sse` frontend ↔ `streamable_http` backend. **Only this pairing is
supported.** Messages are **opaque JSON-RPC bytes** — the bridge does not
parse or interpret them.

The reference deployment has one bridge named `affine` with prefix
`/mcp/affine`, used to expose AFFiNE's MCP StreamableHTTP server as SSE on
the tailnet.

#### `GET <prefix>/sse`

1. Generate a 16-hex-char session ID using a cryptographically-secure RNG.
2. Open a new backend connection for this session.
3. Set headers `Content-Type: text/event-stream`,
   `Cache-Control: no-cache`, `Connection: keep-alive`.
4. Emit an initial `event: endpoint\ndata: <path_prefix>/message?sessionId=<id>\n\n`.
5. Stream loop: forward every message arriving on the session's outbound
   channel as `data: <bytes>\n\n` until the client disconnects, the bridge
   context is cancelled, or the upstream closes.

#### `POST <prefix>/message?sessionId=<id>`

1. `sessionId` query param required. Missing → 400. Unknown → 404.
2. Read body bounded by 1 MiB.
3. Hand the body off asynchronously: POST to the bridge's `upstream_url`
   with `Content-Type: application/json` and
   `Accept: application/json, text/event-stream` (and any bridge auth).
4. Respond `202 Accepted` immediately; the upstream response is forwarded
   onto the session's outbound channel and out via the SSE stream.

#### Backend semantics

- Track the `Mcp-Session-Id` learned from the first upstream response and
  echo it on subsequent requests.
- Detect upstream content type:
  - `text/event-stream` → parse SSE (`data:` lines, blank-line dispatch,
    multi-line concatenation, 1 MiB scanner limit) and forward each event's
    data as one outbound message.
  - Anything else → read up to 1 MiB and forward as one message.
- Optional bearer auth per bridge.
- MCP bridges SHOULD use HTTP connections with longer timeouts than the LLM
  frontends — recommended bounds: idle pool ~8 connections (2 per host),
  idle timeout 90s, response-header timeout 120s. HTTP/2 is not required.
- Session outbound-channel buffer: 64 messages.

### 4.5 Health & readiness

| Method | Path | Body |
|--------|------|------|
| GET    | `/health` | `{"status":"ok"}` |
| GET    | `/ready`  | `{"status":"ready"}` |

`/ready` currently == `/health` (config is loaded by the time we're serving).
Upstream probes are not implemented.

There is **no `/metrics` endpoint** — Prometheus instrumentation is an
explicit non-goal (§12).

### 4.6 Streaming wire format

All streaming responses MUST:

- Use **HTTP/1.1**. HTTP/2 frame buffering was observed to produce stalls; the
  HTTP/1.1 constraint is deliberate.
- Use **chunked transfer encoding** with **per-chunk flushing** to the
  client. Without explicit flushing, intermediate buffers hold output and
  clients see latency spikes or full hangs.
- Use a small read buffer (4 KiB is sufficient) when piping upstream bytes.
- Preserve upstream chunk boundaries — do not coalesce.
- Use the wire framing required by the surface: SSE (`data: <bytes>\n\n`)
  for OpenAI, Anthropic pass-through, Gemini `?alt=sse`, and MCP; NDJSON
  (one JSON object per line) for Gemini default streaming.

---

## 5. Routing & resolver

### 5.1 Resolution result

Given a client-supplied model name, resolution produces:

- `ModelName` — canonical model after alias resolution.
- `UpstreamModel` — what to send upstream.
- `Provider` — the resolved provider (name + endpoint + auth).
- `Fallback` — ordered list of canonical model names to try on retryable
  failure.
- `DefaultEmbedDimensions` — optional dimension hint.

### 5.2 Alias chain resolution

- Aliases MAY chain N levels (alias → alias → ... → model) but MUST NOT
  cycle. Cycles MUST be detected and surfaced as a resolution error
  (`alias cycle involving "<name>"`).
- Once the chain lands on a name that is **not** an alias, look it up in
  `models`. Unknown → `unknown model "<name>"`.
- The resolved model's provider MUST exist.

### 5.3 Fallback semantics

- Each model's `fallback` is a list of canonical model names (not aliases —
  validation rejects fallback to an alias).
- Each hop in the fallback chain is **re-resolved** so it can use its own
  provider and `upstream_model`.
- A fallback fires **only** when the upstream returns a 5xx **and** the
  wrapped-client-error heuristic (§4.1) rules out a wrapped 4xx.
- Transport-level errors (DNS, connect, TLS, response-header timeout) also
  trigger a fallback hop on the OpenAI frontend. On the Gemini frontend
  they surface as `502` immediately.
- **Once any byte has been written to the client (status code or body), no
  further fallback is attempted** (see invariant §11.4).

### 5.4 Model list

Model listing MUST return every key from both `models` and `aliases`
(unsorted is acceptable). Used by `/v1/models` and `/v1beta/models`. See
invariant §11.8 — AFFiNE's `CopilotProviderFactory` keys provider routing on
this list and breaks if aliases are dropped.

---

## 6. Authentication

### 6.1 Inbound

**The gateway does NOT authenticate inbound requests.** All exposed routes
are open. Security relies entirely on:

- Binding to `127.0.0.1` (default `listen`).
- Tailscale Serve providing tailnet-only HTTPS termination upstream.
- The systemd sandbox in the NixOS module.

The README's earlier claim that "the gateway refuses unauthenticated
requests by design" describes a non-existent inbound auth layer. If you need
client auth, add it externally (Tailscale Aperture, an Nginx `auth_request`,
or a separate reverse-proxy).

### 6.2 Outbound

Two outbound bearer variants, both setting `Authorization: Bearer <token>`:

**Fixed-token bearer.** Token is read from config at startup. Never re-read.

**File-backed bearer.** Token value lives in a file. The path is validated
at startup (read once to fail fast on missing/unreadable files).

#### 6.2.1 Critical invariant: per-request token re-read (file-backed)

> **A file-backed bearer MUST re-read the token file from disk on every
> outbound request.**

This invariant exists so external sidecars (e.g. `claude-remote-control`'s
OAuth refresher) can rotate the token by writing the file, **without
restarting the gateway**. Do not introduce caching, in-memory promotion,
fs-watching, or any other deferred read — those all silently break
rotation. The invariant has been broken once already and is non-obvious to
re-derive from a clean rewrite. It MUST have direct test coverage.

Behaviour on read failure: the request MUST fail (5xx). It MUST NOT
silently send an empty bearer.

#### 6.2.2 ChatGPT OAuth (`oauth_chatgpt`)

Used by `type: codex` providers. The strategy reads a credentials file — a
JSON document holding at least a `refresh_token`, either flat
(`{access_token, refresh_token, ...}`) or nested under `tokens` (the Codex
CLI / openai-oauth layout) — and manages the access token itself:

- On each request, if the cached access token is within ~5 minutes of expiry
  (read from the JWT `exp` claim; a time-based passive refresh at ~55 minutes
  is the fallback when the JWT is unreadable), it refreshes synchronously
  against `<issuer>/oauth/token` with `grant_type=refresh_token`. Refreshes
  are single-flighted.
- The gateway is the **sole owner** of the refresh-token lineage — no external
  CLI or proxy shares it — which is what prevents the mutual token-rotation
  invalidation that a shared refresh token causes. The refreshed tokens
  (including a rotated refresh token) are persisted back to `file` atomically.
- `Apply` sets `Authorization: Bearer <access token>` and, when derivable from
  the token's `https://api.openai.com/auth.chatgpt_account_id` claim,
  `ChatGPT-Account-Id` (the Codex backend returns HTML 403 pages without it).
- Every refresh — attempt, success, and failure — MUST be logged (a silent
  refresh failure otherwise only surfaces as downstream 401s).
- A missing `refresh_token` at startup is a fatal configuration error.

### 6.3 Per-surface auth selection

Each provider has its own optional outbound bearer. The Anthropic frontend
has its own dedicated bearer (independent of the provider map). MCP bridges
have their own per-bridge bearers.

Providers without configured auth send no `Authorization` header upstream.

### 6.4 Inbound header stripping

The Anthropic handler strips both `Authorization` **and** `x-api-key` from
the inbound request before applying the gateway's auth (§4.2, invariant
§11.2). It also forces `Accept-Encoding: identity` so the upstream does not
send gzip the relay would otherwise have to decompress (or re-encode).

The OpenAI and Gemini handlers construct outbound requests from scratch
(no header copy from the client request), so explicit stripping is not
needed there — only `Content-Type` and the gateway's `Authorization` end up
on the wire.

---

## 7. Observability

### 7.1 Request IDs

Every request is tagged:

- If `X-Request-ID` is present on the inbound request, use it.
- Otherwise, generate a fresh ID using a cryptographically-secure RNG. The
  reference implementation uses 6 random bytes, hex-encoded (12 chars).
- Echo the ID in the response header.
- Make the ID available to every log line emitted while serving the request.

### 7.2 Structured logging

JSON to stderr, level Info. Each served request emits a `served` line with:

- **OpenAI frontend**: `request_id`, `client_model`, `resolved_model`,
  `provider`, `stream`, `fallback_index`, `latency_ms`.
- **Anthropic frontend**: `request_id`, `frontend: "anthropic"`, `model`,
  `stream`, `status`, `latency_ms`.
- **Gemini frontend**: `request_id`, `frontend: "gemini"`, `client_model`,
  `resolved_model`, `provider`, `stream`, `fallback_index`, `latency_ms`.

Errors include `request_id` and an `err` field.

MCP bridge logs include `mcp_bridge=<name>` and, within sessions,
`session=<id>`.

### 7.3 Metrics

None. Prometheus `/metrics` is an explicit non-goal (§12).

---

## 8. Memory & networking discipline

The reference target is a 4 GB-RAM Raspberry Pi 5 sharing the host with
other workloads. The gateway is configured to live in ~30 MB RSS.

- LLM-upstream HTTP client: small idle pool (recommended 16 total / 4 per
  host), 60s idle timeout, 10s TLS handshake, 30s response-header timeout,
  1s expect-continue. **No client-level overall timeout** — streaming
  requests can run for minutes. HTTP/2 disabled (HTTP/1.1 is required for
  reliable streaming; see §4.6).
- MCP-bridge HTTP client: longer timeouts (idle 90s, response-header 120s),
  smaller idle pool (8 / 2).
- Request body cap: **8 MiB** for LLM frontends. MCP message bodies are
  capped at **1 MiB**.
- Streaming reads use a small buffer (4 KiB in the reference
  implementation) with explicit flush per chunk.
- Gemini non-streaming upstream response read is limited to 4 MiB; embedding
  responses to 8 MiB.

---

## 9. NixOS module

The reference NixOS module exposes `services.tiny-llm-gate` with:

| Option | Type | Default | Purpose |
|--------|------|---------|---------|
| `enable` | bool | false | toggle |
| `package` | package | — | gateway derivation |
| `settings` | attrset (yaml format) | `{}` | inline config; serialized to YAML |
| `configFile` | nullable path | null | path to a YAML config (overrides `settings`) |
| `memoryMax` | str | `"30M"` | systemd `MemoryMax=` |
| `goMemLimit` | str | `"20MiB"` | runtime memory-limit hint env var (implementation-specific) |
| `secretPaths` | list of str | `[]` | systemd `ReadOnlyPaths=` so the unit can read agenix secrets |

The unit ships with: `DynamicUser`, `NoNewPrivileges`,
`ProtectSystem=strict`, `ProtectHome`, `PrivateTmp`, `PrivateDevices`,
`ProtectKernelTunables`, `ProtectKernelModules`, `ProtectControlGroups`,
`RestrictSUIDSGID`, `LockPersonality`, `RestrictRealtime`,
`SystemCallArchitectures=native`, `CapabilityBoundingSet=""`,
`Restart=on-failure`, `RestartSec=5s`.

A rewrite in another language would replace `goMemLimit` with whatever
memory-budget knob that runtime exposes.

---

## 10. Failure modes

| Failure | Behaviour |
|---------|-----------|
| Config file missing/unparseable | exit non-zero at startup with structured error |
| `auth.token_file` unreadable at startup | exit non-zero |
| `auth.token_file` deleted post-startup | next request returns 5xx with read error |
| Unknown `model` in inbound request | 404 `{"error":{"message":"unknown model ..."}}` |
| Alias cycle | 404 (resolver error surfaces via the same path) |
| Inbound body > 8 MiB | 400 from request-size enforcement |
| Inbound body invalid JSON | 400 `"invalid JSON body"` |
| Inbound body missing `model` | 400 `"missing 'model' field"` (OpenAI/Gemini) |
| Upstream 5xx, fallback available | retry next hop |
| Upstream 5xx wrapped client error (`type=invalid_request_error` or message starting `Invalid '`) | treat as non-retryable, surface to client as 400 |
| Upstream transport error (OpenAI frontend) | `502 "all upstreams failed: ..."` after exhausting chain |
| Upstream transport error (Anthropic frontend) | `502 "upstream transport: ..."` |
| Upstream non-200, non-5xx (Gemini frontend) | `502` with upstream status + first 400 bytes of body |
| Client disconnects mid-stream | handler returns silently; context-cancellation is swallowed |
| MCP `sessionId` missing | 400 from the message handler |
| MCP `sessionId` unknown | 404 from the message handler |

All gateway-originated error bodies use the shape:

```json
{"error":{"message":"<text>","type":"tiny_llm_gate_error"}}
```

Upstream error bodies (after status has been written to the client) are
forwarded byte-for-byte and keep whatever shape the upstream chose.

---

## 11. Invariants

These are properties any implementation MUST preserve. Each is load-bearing
and has either been broken in the past or has non-obvious consequences if
broken.

1. **A file-backed bearer re-reads the token file on every outbound
   request.** Caching would break OAuth-token rotation by an external
   sidecar (`claude-remote-control`). This invariant MUST have direct test
   coverage. See §6.2.1.

2. **The Anthropic handler strips both `Authorization` *and* `x-api-key`
   from the inbound request before applying gateway auth.** Without the
   `x-api-key` strip, clients that send `x-api-key` (e.g. `pi-coding-agent`)
   defeat the gateway's auth swap, because Anthropic prefers `x-api-key`
   when both headers are present. See §4.2, §6.4.

3. **The Anthropic handler does not mutate the request body.** The point of
   the pass-through is verbatim forwarding for the observability layer
   (Aperture) sitting upstream. Do not add body parsing or model rewriting
   on this path. See §4.2.

4. **Fallback only fires before bytes are written to the client.** Once the
   status code or any body byte has been emitted to the client, the
   gateway MUST NOT try another upstream — partial responses cannot be
   recombined. See §5.3.

5. **The body rewriter preserves field order.** When the inbound `model`
   value is replaced before forwarding, the surrounding JSON document MUST
   remain byte-stable except for the replaced value. A parse-into-map /
   re-serialize approach reorders fields by map iteration and is therefore
   forbidden. Observability sniffers downstream (Aperture) rely on
   byte-stable bodies for diffing and replay. See §4.1.

6. **Config validation rejects unknown YAML fields.** Strict mode is
   mandatory — silently ignoring an unknown key has historically masked
   typos that caused production routing failures. New fields must be added
   to the schema *before* shipping a config that uses them.

7. **Inbound body cap is 8 MiB; MCP body cap is 1 MiB.** Both are
   conservative; raising them needs a corresponding memory-budget review
   on the reference RPi5 target.

8. **`GET /v1/models` and `GET /v1beta/models` enumerate aliases too.**
   AFFiNE's `CopilotProviderFactory` selects providers by checking whether
   the requested model id appears in the listed set; dropping aliases here
   breaks routing. See §5.4.

9. **Streaming uses HTTP/1.1 with explicit per-chunk flushing.** HTTP/2
   frame buffering and the absence of explicit flushes were both observed
   to produce stalls. The HTTP/1.1 + flush combination is deliberate. See
   §4.6.

---

## 12. Non-goals

- USD cost tracking.
- Prometheus `/metrics`.
- Multi-tenant billing or quotas.
- Request rate limiting.
- Tokenizer-based features (counting, truncation).
- Multimodal Gemini parts (images, audio, video).
- Gemini safety settings.
- Anthropic body translation or model rewriting.
- Inbound client authentication. (See §6.1.)
- SIGHUP / hot-reload.

---

## 13. Implementation pointers (current)

> Everything above is contract. This section is a pointer into the current
> implementation and is the only part of this document that is expected to
> change under a rewrite.

- **Language**: Go (CGO disabled, stripped binary ~6.5 MiB).
- **Entrypoint**: `cmd/tiny-llm-gate/main.go`; CLI flags `-config <path>`
  (default `config.yaml`) and `-version`.
- **Config & validation**: `internal/config` — strict YAML via
  `gopkg.in/yaml.v3` with `KnownFields(true)`.
- **Auth**: `internal/auth` — fixed and file-backed bearers; the per-request
  re-read invariant (§6.2.1) lives here and is pinned by a dedicated test.
  The `oauth_chatgpt` strategy (§6.2.2) is `internal/auth/oauth_chatgpt.go`.
- **Codex translation**: `internal/codex` — OpenAI↔Codex Responses request
  and SSE-response translators (§4.1.1); the server hop lives in
  `internal/server/codex.go`.
- **Routing & resolver**: `internal/resolve` — alias-chain walk with cycle
  detection.
- **HTTP server wiring**: `internal/server` — mux, HTTP clients, MCP bridge
  construction, request-ID middleware.
- **OpenAI frontend**: `internal/server/openai.go` — includes the
  order-preserving token-stream body rewriter for §11.5.
- **Anthropic frontend**: `internal/server/anthropic.go` — pass-through
  handler with header strip + auth swap.
- **Gemini frontend**: `internal/server/gemini.go` (dispatcher);
  `internal/gemini` (wire types + Gemini↔OpenAI translator).
- **MCP bridge**: `internal/mcp` — SSE frontend, StreamableHTTP backend,
  per-connection session state.
- **NixOS module**: `nixos-module.nix`.
- **Reference config**: `testdata/example-config.yaml`.
- **Versioning**: injected at build time; the flake derives the version
  from the git short rev. Current development version is **0.3.3-dev**;
  phase status per `ROADMAP.md` is **v0.3.0 features shipped, phase 5
  (LiteLLM cutover) underway**.
