package mcp

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"

	"github.com/nSimonFR/tiny-llm-gate/internal/auth"
)

// backendConn manages upstream StreamableHTTP requests for one bridge.
type backendConn struct {
	upstreamURL string
	auth        auth.Authenticator // nil = unauthenticated
	client      *http.Client
	logger      *slog.Logger
	ctx         context.Context // bridge-level context

	mu        sync.RWMutex
	sessionID string // Mcp-Session-Id learned from upstream
}

func newBackendConn(upstreamURL string, a auth.Authenticator, client *http.Client, ctx context.Context, logger *slog.Logger) *backendConn {
	return &backendConn{
		upstreamURL: upstreamURL,
		auth:        a,
		client:      client,
		ctx:         ctx,
		logger:      logger,
	}
}

// Send posts a JSON-RPC message to the upstream StreamableHTTP endpoint and
// relays the response(s) to outCh. The upstream may respond with a single
// JSON body or an SSE stream — both are handled.
func (bc *backendConn) Send(ctx context.Context, body []byte, outCh chan<- []byte) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, bc.upstreamURL, strings.NewReader(string(body)))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")

	bc.mu.RLock()
	if bc.sessionID != "" {
		req.Header.Set("Mcp-Session-Id", bc.sessionID)
	}
	bc.mu.RUnlock()

	if bc.auth != nil {
		if err := bc.auth.Apply(ctx, req); err != nil {
			return fmt.Errorf("apply auth: %w", err)
		}
	}

	resp, err := bc.client.Do(req)
	if err != nil {
		return fmt.Errorf("upstream request: %w", err)
	}
	defer resp.Body.Close()

	// Learn session ID from upstream.
	if sid := resp.Header.Get("Mcp-Session-Id"); sid != "" {
		bc.mu.Lock()
		bc.sessionID = sid
		bc.mu.Unlock()
	}

	ct := resp.Header.Get("Content-Type")
	switch {
	case strings.HasPrefix(ct, "text/event-stream"):
		return bc.readSSEStream(ctx, resp.Body, outCh)
	default:
		// Treat as single JSON-RPC response (including error bodies).
		data, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20)) // 1 MiB
		if err != nil {
			return fmt.Errorf("read response: %w", err)
		}
		if len(data) == 0 {
			return nil
		}
		select {
		case outCh <- data:
		case <-ctx.Done():
			return ctx.Err()
		}
		return nil
	}
}

// readSSEStream reads an SSE stream from the upstream and sends each data
// payload to outCh. Supports multi-line data fields per the SSE spec.
func (bc *backendConn) readSSEStream(ctx context.Context, r io.Reader, outCh chan<- []byte) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20) // up to 1 MiB lines

	var dataBuf strings.Builder
	for scanner.Scan() {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		line := scanner.Text()

		if strings.HasPrefix(line, "data: ") {
			if dataBuf.Len() > 0 {
				dataBuf.WriteByte('\n')
			}
			dataBuf.WriteString(line[6:])
			continue
		}
		if line == "data:" {
			// Empty data line.
			if dataBuf.Len() > 0 {
				dataBuf.WriteByte('\n')
			}
			continue
		}

		// Empty line = end of event. Dispatch if we have data.
		if line == "" && dataBuf.Len() > 0 {
			payload := []byte(dataBuf.String())
			dataBuf.Reset()
			select {
			case outCh <- payload:
			case <-ctx.Done():
				return ctx.Err()
			}
			continue
		}

		// Other SSE fields (event:, id:, retry:) — ignore for MCP bridging.
	}
	// Flush any trailing data (stream closed without final blank line).
	if dataBuf.Len() > 0 {
		payload := []byte(dataBuf.String())
		select {
		case outCh <- payload:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return scanner.Err()
}
