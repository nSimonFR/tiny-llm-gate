package mcp

import (
	"crypto/rand"
	"fmt"
	"io"
	"net/http"
)

// HandleSSE serves GET requests that establish a persistent SSE connection.
// The client receives an initial "endpoint" event with the POST URL for
// sending messages, then receives JSON-RPC responses as data events.
func (b *Bridge) HandleSSE(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	id := newSessionID()
	bc := newBackendConn(b.cfg.UpstreamURL, b.auth, b.client, b.ctx, b.logger)
	sess := newSession(id, bc, b.logger)
	b.sessions.Store(id, sess)
	defer func() {
		b.sessions.Delete(id)
		sess.Close()
	}()

	b.logger.Info("SSE session opened", "session", id)

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	// Send the endpoint event so the client knows where to POST messages.
	postURL := b.cfg.PathPrefix + "/message?sessionId=" + id
	writeSSEEvent(w, "endpoint", postURL)
	flusher.Flush()

	// Stream loop: relay backend responses to the SSE client.
	for {
		select {
		case msg, ok := <-sess.OutCh():
			if !ok {
				return
			}
			writeSSEData(w, msg)
			flusher.Flush()
		case <-r.Context().Done():
			b.logger.Info("SSE session closed", "session", id)
			return
		case <-b.ctx.Done():
			return
		}
	}
}

// HandleMessage serves POST requests that deliver JSON-RPC messages from the
// SSE client to the upstream backend. Returns 202 Accepted; the response is
// delivered asynchronously on the SSE stream.
func (b *Bridge) HandleMessage(w http.ResponseWriter, r *http.Request) {
	sessionID := r.URL.Query().Get("sessionId")
	if sessionID == "" {
		http.Error(w, "missing sessionId parameter", http.StatusBadRequest)
		return
	}

	val, ok := b.sessions.Load(sessionID)
	if !ok {
		http.Error(w, "unknown session", http.StatusNotFound)
		return
	}
	sess := val.(*session)

	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20)) // 1 MiB
	if err != nil {
		http.Error(w, "failed to read body", http.StatusBadRequest)
		return
	}
	if len(body) == 0 {
		http.Error(w, "empty body", http.StatusBadRequest)
		return
	}

	sess.SendToBackend(body)
	w.WriteHeader(http.StatusAccepted)
}

// writeSSEEvent writes a named SSE event.
func writeSSEEvent(w io.Writer, event, data string) {
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
}

// writeSSEData writes an SSE data-only event.
func writeSSEData(w io.Writer, data []byte) {
	fmt.Fprintf(w, "data: %s\n\n", data)
}

// newSessionID generates a 16-character hex session ID.
func newSessionID() string {
	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		panic("crypto/rand failed: " + err.Error())
	}
	return fmt.Sprintf("%x", b)
}
