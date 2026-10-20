package codex

import (
	"encoding/json"
	"fmt"
	"io"
)

// droppedResponsesFields are Responses parameters the ChatGPT backend rejects;
// the chat translation never forwarded them either.
var droppedResponsesFields = []string{
	"max_output_tokens", "temperature", "top_p", "user", "metadata", "previous_response_id",
}

// NormalizeResponses prepares a client's /v1/responses body for the ChatGPT
// backend without translating it: tools, additional_tools, custom tools and
// every input item type pass through untouched. Only what the backend
// requires changes: the upstream model, stream=true, store=false.
func NormalizeResponses(body []byte, upstreamModel string, effort *string) ([]byte, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		return nil, fmt.Errorf("codex: parse responses request: %w", err)
	}
	m["model"], _ = json.Marshal(upstreamModel)
	m["stream"] = json.RawMessage("true")
	m["store"] = json.RawMessage("false")
	for _, k := range droppedResponsesFields {
		delete(m, k)
	}
	var text string
	if json.Unmarshal(m["input"], &text) == nil {
		m["input"], _ = json.Marshal([]codexInput{{Role: "user", Content: text}})
	}
	if _, ok := m["instructions"]; !ok && !hasDeveloperInput(m["input"]) {
		m["instructions"], _ = json.Marshal(defaultInstructions)
	}
	if effort != nil {
		var r map[string]json.RawMessage
		_ = json.Unmarshal(m["reasoning"], &r)
		if _, set := r["effort"]; !set {
			if r == nil {
				r = map[string]json.RawMessage{}
			}
			r["effort"], _ = json.Marshal(*effort)
			m["reasoning"], _ = json.Marshal(r)
		}
	}
	return json.Marshal(m)
}

func hasDeveloperInput(raw json.RawMessage) bool {
	var items []struct {
		Role string `json:"role"`
	}
	_ = json.Unmarshal(raw, &items)
	for _, it := range items {
		if it.Role == "developer" || it.Role == "system" {
			return true
		}
	}
	return false
}

// CollectResponse drains a Codex Responses SSE stream into the final Response
// object, for clients that did not ask to stream. The backend's
// response.completed can carry an empty output, so the output_item.done items
// fill it in.
func CollectResponse(src io.Reader) ([]byte, error) {
	var items []json.RawMessage
	var final map[string]json.RawMessage
	sc := newSSEScanner(src)
	for sc.next() {
		evt, ok := sc.event()
		if !ok {
			continue
		}
		switch evt.Type {
		case "response.output_item.done":
			var e struct {
				Item json.RawMessage `json:"item"`
			}
			if json.Unmarshal([]byte(sc.raw), &e) == nil && len(e.Item) > 0 {
				items = append(items, e.Item)
			}
		case "response.completed", "response.incomplete":
			var e struct {
				Response map[string]json.RawMessage `json:"response"`
			}
			if json.Unmarshal([]byte(sc.raw), &e) == nil {
				final = e.Response
			}
		case "error", "response.failed":
			if evt.Error != nil {
				return nil, fmt.Errorf("codex upstream error (%s): %s", evt.Error.Code, evt.Error.Message)
			}
			return nil, fmt.Errorf("codex upstream error: %s", sc.raw)
		}
	}
	if err := sc.err(); err != nil {
		return nil, err
	}
	if final == nil {
		return nil, fmt.Errorf("codex stream ended without response.completed")
	}
	var out []json.RawMessage
	if json.Unmarshal(final["output"], &out) != nil || len(out) == 0 {
		if items == nil {
			items = []json.RawMessage{}
		}
		final["output"], _ = json.Marshal(items)
	}
	return json.Marshal(final)
}
