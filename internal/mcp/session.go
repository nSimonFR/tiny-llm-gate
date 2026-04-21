// Package mcp implements MCP (Model Context Protocol) transport bridging.
// It translates between MCP transport types (SSE, StreamableHTTP) without
// interpreting JSON-RPC message semantics — messages are opaque bytes.
package mcp

import (
	"context"
	"log/slog"
	"sync/atomic"
	"time"
)

// session tracks one SSE client connection and its upstream backend.
type session struct {
	id       string
	outCh    chan []byte // SSE frontend reads from this; capped at creation.
	backend  *backendConn
	ctx      context.Context
	cancel   context.CancelFunc
	logger   *slog.Logger
	lastSend atomic.Int64 // unix seconds of last SendToBackend call
}

const sessionChanCap = 64

func newSession(id string, bc *backendConn, logger *slog.Logger) *session {
	ctx, cancel := context.WithCancel(bc.ctx)
	s := &session{
		id:      id,
		outCh:   make(chan []byte, sessionChanCap),
		backend: bc,
		ctx:     ctx,
		cancel:  cancel,
		logger:  logger.With("session", id),
	}
	s.lastSend.Store(time.Now().Unix())
	return s
}

// SendToBackend forwards a JSON-RPC message to the upstream and relays
// responses back on outCh. Runs the backend call in a goroutine so the
// POST /message handler can return 202 immediately.
func (s *session) SendToBackend(body []byte) {
	s.lastSend.Store(time.Now().Unix())
	go func() {
		if err := s.backend.Send(s.ctx, body, s.outCh); err != nil {
			s.logger.Warn("backend send failed", "err", err)
		}
	}()
}

// OutCh returns the read side of the outbound message channel.
func (s *session) OutCh() <-chan []byte { return s.outCh }

// Close cancels the session context, which aborts in-flight backend requests.
func (s *session) Close() { s.cancel() }
