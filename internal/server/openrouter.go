package server

import (
	"encoding/json"
	"net/http"
	"sort"
	"strings"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
	"github.com/nSimonFR/tiny-llm-gate/internal/resolve"
)

// openRouterOnly are request fields OpenRouter defines but no upstream
// accepts; strict upstreams (api.openai.com) 400 on unknown fields.
var openRouterOnly = []string{"models", "provider", "transforms", "route", "usage", "reasoning", "plugins"}

// normalizeOpenRouter folds OpenRouter's request extensions into plain
// OpenAI form. The body is returned untouched unless one is present, so the
// byte-stable forwarding invariant holds for plain OpenAI clients.
func normalizeOpenRouter(body []byte) (out []byte, models []string, err error) {
	var m map[string]json.RawMessage
	if json.Unmarshal(body, &m) != nil {
		return body, nil, nil
	}
	present := hasServerTools(m["tools"]) || hasOnlineSuffix(m["model"])
	for _, k := range openRouterOnly {
		if _, ok := m[k]; ok {
			present = true
			break
		}
	}
	if !present {
		return body, nil, nil
	}
	foldWebSearch(m)

	if raw, ok := m["models"]; ok {
		_ = json.Unmarshal(raw, &models)
		for i, mod := range models {
			models[i] = strings.TrimSuffix(mod, onlineSuffix)
		}
		if _, hasModel := m["model"]; !hasModel && len(models) > 0 {
			m["model"], _ = json.Marshal(models[0])
		}
	}
	if raw, ok := m["reasoning"]; ok {
		if _, set := m["reasoning_effort"]; !set {
			if eff := reasoningEffort(raw); eff != "" {
				m["reasoning_effort"], _ = json.Marshal(eff)
			}
		}
	}
	if raw, ok := m["usage"]; ok {
		var u struct {
			Include bool `json:"include"`
		}
		_ = json.Unmarshal(raw, &u)
		_, hasOpts := m["stream_options"]
		if u.Include && !hasOpts && string(m["stream"]) == "true" {
			m["stream_options"] = json.RawMessage(`{"include_usage":true}`)
		}
	}
	for _, k := range openRouterOnly {
		delete(m, k)
	}
	out, err = json.Marshal(m)
	return out, models, err
}

// webSearchTool is the hosted tool the codex (ChatGPT Responses) backend runs.
var webSearchTool = json.RawMessage(`{"type":"web_search"}`)

const onlineSuffix = ":online"

func hasOnlineSuffix(raw json.RawMessage) bool {
	var model string
	return json.Unmarshal(raw, &model) == nil && strings.HasSuffix(model, onlineSuffix)
}

func hasServerTools(raw json.RawMessage) bool {
	var tools []struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(raw, &tools) != nil {
		return false
	}
	for _, t := range tools {
		if strings.HasPrefix(t.Type, "openrouter:") {
			return true
		}
	}
	return false
}

// foldWebSearch maps OpenRouter's three spellings of web search — the
// openrouter:web_search tool, plugins:[{id:"web"}] and a `:online` model
// suffix — to one hosted web_search tool. Other openrouter:* server tools
// are dropped. Backends without web search shed it per hop (stripHostedTools).
func foldWebSearch(m map[string]json.RawMessage) {
	want := false
	var model string
	if json.Unmarshal(m["model"], &model) == nil && strings.HasSuffix(model, onlineSuffix) {
		m["model"], _ = json.Marshal(strings.TrimSuffix(model, onlineSuffix))
		want = true
	}
	var plugins []struct {
		ID string `json:"id"`
	}
	if json.Unmarshal(m["plugins"], &plugins) == nil {
		for _, p := range plugins {
			want = want || p.ID == "web"
		}
	}
	var tools []json.RawMessage
	_ = json.Unmarshal(m["tools"], &tools)
	kept := make([]json.RawMessage, 0, len(tools)+1)
	for _, t := range tools {
		var peek struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal(t, &peek)
		switch {
		case peek.Type == "openrouter:web_search" || peek.Type == "web_search":
			want = true
		case strings.HasPrefix(peek.Type, "openrouter:"):
		default:
			kept = append(kept, t)
		}
	}
	if want {
		kept = append(kept, webSearchTool)
	}
	if len(kept) == 0 {
		delete(m, "tools")
		delete(m, "tool_choice")
		return
	}
	m["tools"], _ = json.Marshal(kept)
}

// stripHostedTools drops non-function tools (e.g. web_search) for backends
// that only run function tools. The body is untouched when there are none.
func stripHostedTools(body []byte) []byte {
	var m map[string]json.RawMessage
	if json.Unmarshal(body, &m) != nil {
		return body
	}
	var tools []json.RawMessage
	if json.Unmarshal(m["tools"], &tools) != nil {
		return body
	}
	kept := make([]json.RawMessage, 0, len(tools))
	for _, t := range tools {
		var peek struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(t, &peek) == nil && (peek.Type == "function" || peek.Type == "") {
			kept = append(kept, t)
		}
	}
	if len(kept) == len(tools) {
		return body
	}
	if len(kept) == 0 {
		delete(m, "tools")
		delete(m, "tool_choice")
	} else {
		m["tools"], _ = json.Marshal(kept)
	}
	out, err := json.Marshal(m)
	if err != nil {
		return body
	}
	return out
}

// reasoningEffort maps OpenRouter's unified `reasoning` object to an effort.
func reasoningEffort(raw json.RawMessage) string {
	var r struct {
		Effort    string `json:"effort"`
		MaxTokens int    `json:"max_tokens"`
		Enabled   *bool  `json:"enabled"`
	}
	if json.Unmarshal(raw, &r) != nil {
		return ""
	}
	switch {
	case r.Enabled != nil && !*r.Enabled:
		return "none"
	case r.Effort != "":
		return r.Effort
	case r.MaxTokens > 0 && r.MaxTokens <= 4096:
		return "low"
	case r.MaxTokens > 16384:
		return "high"
	case r.MaxTokens > 0, r.Enabled != nil:
		return "medium"
	}
	return ""
}

// buildChain is the hop order for a request: the requested model, then the
// OpenRouter `models` list, then the primary's configured fallbacks; deduped,
// and filtered by the caller's allowlist (the primary was already checked).
func (s *Server) buildChain(r *http.Request, res *resolve.Resolved, extra []string) []string {
	ck := clientOf(r.Context())
	seen := map[string]bool{}
	chain := []string{}
	add := func(name string) {
		if !seen[name] {
			seen[name] = true
			chain = append(chain, name)
		}
	}
	add(res.ModelName)
	for _, m := range extra {
		if hop, err := s.resolver.Resolve(m); err == nil && ck.allows(m, hop.ModelName, hop.Slug) {
			add(hop.ModelName)
		}
	}
	for _, f := range res.Fallback {
		add(f)
	}
	return chain
}

type orArchitecture struct {
	Modality         string   `json:"modality"`
	InputModalities  []string `json:"input_modalities"`
	OutputModalities []string `json:"output_modalities"`
	Tokenizer        string   `json:"tokenizer"`
	InstructType     *string  `json:"instruct_type"`
}

type orTopProvider struct {
	ContextLength       *int `json:"context_length"`
	MaxCompletionTokens *int `json:"max_completion_tokens"`
	IsModerated         bool `json:"is_moderated"`
}

// orModel is an OpenRouter catalog entry; object/owned_by keep it a valid
// OpenAI model object too.
type orModel struct {
	ID                  string          `json:"id"`
	Object              string          `json:"object"`
	Created             int64           `json:"created"`
	OwnedBy             string          `json:"owned_by"`
	CanonicalSlug       string          `json:"canonical_slug"`
	Name                string          `json:"name"`
	Description         string          `json:"description"`
	ContextLength       *int            `json:"context_length"`
	Architecture        orArchitecture  `json:"architecture"`
	Pricing             config.Pricing  `json:"pricing"`
	TopProvider         orTopProvider   `json:"top_provider"`
	PerRequestLimits    *map[string]any `json:"per_request_limits"`
	SupportedParameters []string        `json:"supported_parameters"`
	// Required (nullable) by the official @openrouter/sdk schema.
	DefaultParameters *map[string]any `json:"default_parameters"`
	SupportedVoices   *[]string       `json:"supported_voices"`
	Links             orLinks         `json:"links"`
}

type orLinks struct {
	Details string `json:"details"`
}

var defaultSupportedParameters = []string{
	"max_tokens", "temperature", "top_p", "stop", "tools", "tool_choice",
	"response_format", "structured_outputs", "reasoning", "include_reasoning",
}

func intPtr(v int) *int {
	if v == 0 {
		return nil
	}
	return &v
}

func catalogEntry(id string, res *resolve.Resolved) orModel {
	info := res.Info
	if info == nil {
		info = &config.ModelInfo{}
	}
	in := info.InputModalities
	if len(in) == 0 {
		in = []string{"text"}
	}
	out := info.OutputModalities
	if len(out) == 0 {
		out = []string{"text"}
	}
	params := info.SupportedParameters
	if len(params) == 0 {
		params = defaultSupportedParameters
	}
	pricing := config.Pricing{Prompt: "0", Completion: "0"}
	if info.Pricing != nil {
		pricing = *info.Pricing
		if pricing.Prompt == "" {
			pricing.Prompt = "0"
		}
		if pricing.Completion == "" {
			pricing.Completion = "0"
		}
	}
	name := info.Name
	if name == "" {
		name = res.Slug
	}
	tokenizer := info.Tokenizer
	if tokenizer == "" {
		tokenizer = "Other"
	}
	return orModel{
		ID: id, Object: "model", Created: info.Created, OwnedBy: res.ProviderName,
		CanonicalSlug: res.Slug, Name: name, Description: info.Description,
		ContextLength: intPtr(info.ContextLength),
		Architecture: orArchitecture{
			Modality:        strings.Join(in, "+") + "->" + strings.Join(out, "+"),
			InputModalities: in, OutputModalities: out, Tokenizer: tokenizer,
		},
		Pricing: pricing,
		TopProvider: orTopProvider{
			ContextLength:       intPtr(info.ContextLength),
			MaxCompletionTokens: intPtr(info.MaxOutputTokens),
		},
		SupportedParameters: params,
		Links:               orLinks{Details: "/api/v1/models/" + id},
	}
}

// catalog lists every addressable id the caller may use: canonical names,
// aliases and `vendor/name` slugs, sorted by id.
func (s *Server) catalog(r *http.Request) []orModel {
	ck := clientOf(r.Context())
	ids := append(s.resolver.ListModels(), s.resolver.ListSlugs()...)
	sort.Strings(ids)
	want := r.URL.Query().Get("supported_parameters")
	out := make([]orModel, 0, len(ids))
	for _, id := range ids {
		res, err := s.resolver.Resolve(id)
		if err != nil || !ck.allows(id, res.ModelName, res.Slug) {
			continue
		}
		e := catalogEntry(id, res)
		if want != "" && !hasAll(e.SupportedParameters, strings.Split(want, ",")) {
			continue
		}
		out = append(out, e)
	}
	return out
}

func hasAll(have, want []string) bool {
	for _, w := range want {
		found := false
		for _, h := range have {
			if h == strings.TrimSpace(w) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	data := s.catalog(r)
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list", "data": data,
		"total_count": len(data), "links": map[string]any{"next": nil},
	})
}

// handleModel serves GET /v1/models/{id...} (ids may contain '/').
func (s *Server) handleModel(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	res, err := s.resolver.Resolve(id)
	if err != nil || !clientOf(r.Context()).allows(id, res.ModelName, res.Slug) {
		writeJSONError(w, http.StatusNotFound, "model not found: "+id)
		return
	}
	writeJSON(w, http.StatusOK, catalogEntry(id, res))
}

// handleKey mirrors OpenRouter's GET /api/v1/key, which apps call to
// validate a key. The gate meters nothing, so usage is 0 and limits null.
func (s *Server) handleKey(w http.ResponseWriter, r *http.Request) {
	label, requests := "open", -1
	if ck := clientOf(r.Context()); ck != nil {
		label = ck.name
		if ck.rpm > 0 {
			requests = ck.rpm
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"data": map[string]any{
		"label": label, "limit": nil, "limit_remaining": nil, "limit_reset": nil,
		"usage": 0, "usage_daily": 0, "usage_weekly": 0, "usage_monthly": 0,
		"is_free_tier": false, "is_provisioning_key": false,
		"rate_limit": map[string]any{"requests": requests, "interval": "1m"},
	}})
}

func (s *Server) handleCredits(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{"data": map[string]any{"total_credits": 0, "total_usage": 0}})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
