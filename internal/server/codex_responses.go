package server

import (
	"fmt"
	"net/http"

	"github.com/nSimonFR/tiny-llm-gate/internal/codex"
	"github.com/nSimonFR/tiny-llm-gate/internal/resolve"
)

// serveCodexResponses answers /v1/responses for a codex hop natively: the
// ChatGPT backend speaks Responses, so the body is forwarded instead of going
// through the chat core, which cannot carry additional_tools, namespaces or
// custom tools. done reports that a response (success or upstream error) was
// written; otherwise retryable/err follow the chatUpstream contract.
func (s *Server) serveCodexResponses(
	w http.ResponseWriter,
	r *http.Request,
	hop *resolve.Resolved,
	body []byte,
	clientStream bool,
	canRetry bool,
) (done, retryable bool, err error) {
	if reason, bad := s.disabled[hop.ProviderName]; bad {
		return false, true, fmt.Errorf("provider %q disabled at startup: %s", hop.ProviderName, reason)
	}
	upBody, err := codex.NormalizeResponses(body, hop.UpstreamModel, hop.ReasoningEffort)
	if err != nil {
		return false, false, fmt.Errorf("%w: %v", errBuild, err)
	}
	up, retryable, err := s.codexPost(r, hop, upBody, canRetry)
	if err != nil {
		return false, retryable, err
	}
	defer up.Body.Close()
	if up.StatusCode < 200 || up.StatusCode >= 300 {
		writeJSONError(w, up.StatusCode, upstreamError(up))
		return true, false, nil
	}
	if clientStream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		streamCopy(w, up.Body)
		return true, false, nil
	}
	out, cerr := codex.CollectResponse(up.Body)
	if cerr != nil {
		writeJSONError(w, http.StatusBadGateway, cerr.Error())
		return true, false, nil
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(out)
	return true, false, nil
}
