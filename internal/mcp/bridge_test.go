package mcp

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/nSimonFR/tiny-llm-gate/internal/config"
)

// mockUpstreamJSON returns a mock StreamableHTTP server that responds with
// a single JSON body (echoing back the request body wrapped in a result).
func mockUpstreamJSON() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Mcp-Session-Id", "test-session-123")
		fmt.Fprintf(w, `{"jsonrpc":"2.0","id":1,"result":%s}`, body)
	}))
}

// mockUpstreamSSE returns a mock StreamableHTTP server that responds with
// an SSE stream containing two data events.
func mockUpstreamSSE() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher := w.(http.Flusher)
		fmt.Fprint(w, "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":\"first\"}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":\"second\"}\n\n")
		flusher.Flush()
	}))
}

// mockUpstreamAuthCheck returns a server that verifies the Bearer token.
func mockUpstreamAuthCheck(expectedToken string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if auth != "Bearer "+expectedToken {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"jsonrpc":"2.0","id":1,"result":"ok"}`))
	}))
}

func newTestBridge(t *testing.T, upstreamURL string, authCfg *config.Auth) (*Bridge, *http.ServeMux) {
	t.Helper()
	cfg := config.MCPBridge{
		Frontend:    "sse",
		Backend:     "streamable_http",
		UpstreamURL: upstreamURL,
		PathPrefix:  "/mcp/test",
		Auth:        authCfg,
	}
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	br, err := NewBridge("test", cfg, &http.Client{Timeout: 10 * time.Second}, logger)
	if err != nil {
		t.Fatalf("NewBridge: %v", err)
	}
	mux := http.NewServeMux()
	br.RegisterRoutes(mux)
	return br, mux
}

func TestSSEEndpointEvent(t *testing.T) {
	upstream := mockUpstreamJSON()
	defer upstream.Close()

	br, mux := newTestBridge(t, upstream.URL, nil)
	defer br.Shutdown(nil)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/mcp/test/sse")
	if err != nil {
		t.Fatalf("GET /sse: %v", err)
	}
	defer resp.Body.Close()

	if resp.Header.Get("Content-Type") != "text/event-stream" {
		t.Errorf("Content-Type = %q", resp.Header.Get("Content-Type"))
	}

	// Read the initial endpoint event.
	scanner := bufio.NewScanner(resp.Body)
	var lines []string
	for scanner.Scan() {
		line := scanner.Text()
		lines = append(lines, line)
		if line == "" && len(lines) > 1 {
			break // end of first event
		}
	}

	if len(lines) < 2 {
		t.Fatalf("expected at least 2 lines, got %d: %v", len(lines), lines)
	}
	if lines[0] != "event: endpoint" {
		t.Errorf("first line = %q, want event: endpoint", lines[0])
	}
	if !strings.HasPrefix(lines[1], "data: /mcp/test/message?sessionId=") {
		t.Errorf("second line = %q, want data: /mcp/test/message?sessionId=...", lines[1])
	}
}

func TestJSONRoundTrip(t *testing.T) {
	upstream := mockUpstreamJSON()
	defer upstream.Close()

	br, mux := newTestBridge(t, upstream.URL, nil)
	defer br.Shutdown(nil)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	// Open SSE connection.
	resp, err := http.Get(ts.URL + "/mcp/test/sse")
	if err != nil {
		t.Fatalf("GET /sse: %v", err)
	}
	defer resp.Body.Close()

	// Read the endpoint event to get the session ID.
	scanner := bufio.NewScanner(resp.Body)
	var postURL string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: /mcp/test/message?sessionId=") {
			postURL = ts.URL + line[6:] // strip "data: " prefix
			break
		}
	}
	if postURL == "" {
		t.Fatal("could not extract POST URL from endpoint event")
	}

	// Read past the empty line ending the endpoint event.
	scanner.Scan()

	// POST a JSON-RPC message.
	msg := `{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}`
	postResp, err := http.Post(postURL, "application/json", strings.NewReader(msg))
	if err != nil {
		t.Fatalf("POST /message: %v", err)
	}
	if postResp.StatusCode != http.StatusAccepted {
		t.Errorf("POST status = %d, want 202", postResp.StatusCode)
	}
	postResp.Body.Close()

	// Read the response from the SSE stream.
	var data string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			data = line[6:]
			break
		}
	}
	if data == "" {
		t.Fatal("no data event received on SSE stream")
	}
	if !strings.Contains(data, `"result"`) {
		t.Errorf("unexpected response: %s", data)
	}
}

func TestSSEStreamResponse(t *testing.T) {
	upstream := mockUpstreamSSE()
	defer upstream.Close()

	br, mux := newTestBridge(t, upstream.URL, nil)
	defer br.Shutdown(nil)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/mcp/test/sse")
	if err != nil {
		t.Fatalf("GET /sse: %v", err)
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	var postURL string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: /mcp/test/message?sessionId=") {
			postURL = ts.URL + line[6:]
			break
		}
	}
	scanner.Scan() // skip empty line

	// POST a message — upstream will respond with SSE stream.
	postResp, err := http.Post(postURL, "application/json",
		strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"test"}`))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	postResp.Body.Close()

	// Collect data events from the SSE stream.
	var events []string
	deadline := time.After(5 * time.Second)
	for len(events) < 2 {
		done := make(chan struct{})
		var line string
		go func() {
			if scanner.Scan() {
				line = scanner.Text()
			}
			close(done)
		}()
		select {
		case <-done:
			if strings.HasPrefix(line, "data: ") {
				events = append(events, line[6:])
			}
		case <-deadline:
			t.Fatalf("timeout waiting for events, got %d", len(events))
		}
	}

	if len(events) != 2 {
		t.Errorf("expected 2 events, got %d", len(events))
	}
	if !strings.Contains(events[0], "first") {
		t.Errorf("event[0] = %q", events[0])
	}
	if !strings.Contains(events[1], "second") {
		t.Errorf("event[1] = %q", events[1])
	}
}

func TestBearerAuth(t *testing.T) {
	upstream := mockUpstreamAuthCheck("my-secret-token")
	defer upstream.Close()

	br, mux := newTestBridge(t, upstream.URL, &config.Auth{
		Type:  "bearer",
		Token: "my-secret-token",
	})
	defer br.Shutdown(nil)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/mcp/test/sse")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	var postURL string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: /mcp/test/message?sessionId=") {
			postURL = ts.URL + line[6:]
			break
		}
	}
	scanner.Scan()

	postResp, err := http.Post(postURL, "application/json",
		strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"test"}`))
	if err != nil {
		t.Fatal(err)
	}
	postResp.Body.Close()

	var data string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			data = line[6:]
			break
		}
	}
	if !strings.Contains(data, `"ok"`) {
		t.Errorf("expected ok response, got: %s", data)
	}
}

func TestUnknownSession(t *testing.T) {
	upstream := mockUpstreamJSON()
	defer upstream.Close()

	br, mux := newTestBridge(t, upstream.URL, nil)
	defer br.Shutdown(nil)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/mcp/test/message?sessionId=nonexistent",
		"application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("status = %d, want 404", resp.StatusCode)
	}
}

func TestMissingSessionID(t *testing.T) {
	upstream := mockUpstreamJSON()
	defer upstream.Close()

	br, mux := newTestBridge(t, upstream.URL, nil)
	defer br.Shutdown(nil)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	resp, err := http.Post(ts.URL+"/mcp/test/message",
		"application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", resp.StatusCode)
	}
}

func TestEmptyBody(t *testing.T) {
	upstream := mockUpstreamJSON()
	defer upstream.Close()

	br, mux := newTestBridge(t, upstream.URL, nil)
	defer br.Shutdown(nil)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	// Open SSE to get a valid session.
	resp, err := http.Get(ts.URL + "/mcp/test/sse")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	scanner := bufio.NewScanner(resp.Body)
	var postURL string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: /mcp/test/message?sessionId=") {
			postURL = ts.URL + line[6:]
			break
		}
	}

	postResp, err := http.Post(postURL, "application/json", bytes.NewReader(nil))
	if err != nil {
		t.Fatal(err)
	}
	defer postResp.Body.Close()
	if postResp.StatusCode != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", postResp.StatusCode)
	}
}
