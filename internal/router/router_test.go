package router

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestExtractors(t *testing.T) {
	cases := []struct {
		name string
		fn   func([]byte) (Turn, error)
		body string
		want Turn
	}{
		{"chat", FromChat, `{"messages":[
			{"role":"system","content":"sys"},
			{"role":"user","content":[{"type":"text","text":"q1"},{"type":"image_url","image_url":{}}]},
			{"role":"assistant","content":null,"tool_calls":[{}]},
			{"role":"tool","content":"result"}]}`,
			Turn{System: "sys", Msgs: []Msg{{"user", "q1"}}}},
		{"responses", FromResponses, `{"instructions":"sys","input":[
			{"role":"developer","content":"dev"},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"q1"}]},
			{"type":"function_call","name":"f"},
			{"type":"function_call_output","output":"r"},
			{"role":"assistant","content":[{"type":"output_text","text":"a1"}]}]}`,
			Turn{System: "sys\ndev", Msgs: []Msg{{"user", "q1"}, {"assistant", "a1"}}}},
		{"responses string input", FromResponses, `{"input":"q"}`,
			Turn{Msgs: []Msg{{"user", "q"}}}},
		{"gemini", FromGemini, `{"systemInstruction":{"parts":[{"text":"sys"}]},"contents":[
			{"role":"user","parts":[{"text":"q1"}]},
			{"role":"model","parts":[{"functionCall":{}}]},
			{"role":"user","parts":[{"functionResponse":{}},{"text":"note"}]}]}`,
			Turn{System: "sys", Msgs: []Msg{{"user", "q1"}}}},
		{"messages", FromMessages, `{"system":[{"type":"text","text":"sys"}],"messages":[
			{"role":"user","content":"q1"},
			{"role":"assistant","content":[{"type":"text","text":"a1"},{"type":"tool_use","id":"t"}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"t"},{"type":"text","text":"<system-reminder>"}]}]}`,
			Turn{System: "sys", Msgs: []Msg{{"user", "q1"}, {"assistant", "a1"}}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.fn([]byte(tc.body))
			if err != nil {
				t.Fatal(err)
			}
			g, _ := json.Marshal(got)
			w, _ := json.Marshal(tc.want)
			if string(g) != string(w) {
				t.Errorf("got  %s\nwant %s", g, w)
			}
		})
	}
}

func TestState(t *testing.T) {
	turn := Turn{Msgs: []Msg{{"user", "old"}, {"assistant", strings.Repeat("p", 50)}, {"user", strings.Repeat("u", 100)}}}
	s := State(turn, 40)
	if !strings.Contains(s, "<user_message>\n"+strings.Repeat("u", 30)+"\n</user_message>") {
		t.Errorf("user text not cut to its budget: %s", s)
	}
	if !strings.Contains(s, "<previous_assistant_reply>\n"+strings.Repeat("p", 10)+"\n") {
		t.Errorf("previous reply not cut to its quarter: %s", s)
	}
	if strings.Contains(s, "old") {
		t.Error("state carries older turns")
	}
}

type stubJev struct {
	*httptest.Server
	hits atomic.Int32
	yes  float64
	gate chan struct{} // when set, each answer waits on it
}

func newStub(t *testing.T, yes float64) *stubJev {
	j := &stubJev{yes: yes}
	j.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		j.hits.Add(1)
		if j.gate != nil {
			<-j.gate
		}
		if r.Header.Get("Authorization") != "" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		_, _ = w.Write([]byte(`{"answers":{"a":{"noul":` + jsonFloat(j.yes) + `},"b":{"noul":0.2}}}`))
	}))
	t.Cleanup(j.Close)
	return j
}

func jsonFloat(f float64) string { b, _ := json.Marshal(f); return string(b) }

func newTestRouter(url string) *Router {
	return New(Config{
		Name: "smart", Endpoint: url, Model: "jev", Timeout: time.Second,
		Tiers: []string{"t0", "t1", "t2"}, OnError: "t2", MaxState: 1000, TTL: time.Hour, Entries: 2,
		Questions: []Question{{Name: "a", Escalate: 2, Threshold: 0.5}, {Name: "b", Escalate: 1, Threshold: 0.1}},
	}, http.DefaultClient)
}

func turn(user string) Turn { return Turn{Msgs: []Msg{{"user", user}}} }

func TestDecidePolicy(t *testing.T) {
	ctx := context.Background()
	if d := newTestRouter(newStub(t, 0.9).URL).Decide(ctx, turn("x")); d.Target != "t2" || d.Signals["a"] != 0.9 {
		t.Errorf("highest fired tier should win: %+v", d)
	}
	if d := newTestRouter(newStub(t, 0.3).URL).Decide(ctx, turn("x")); d.Target != "t1" {
		t.Errorf("only b fired: %+v", d)
	}
	if d := newTestRouter(newStub(t, 0.3).URL).Decide(ctx, Turn{}); d.Target != "t0" {
		t.Errorf("no user message should take tier 0: %+v", d)
	}
}

func TestDecideCache(t *testing.T) {
	j := newStub(t, 0.9)
	r := newTestRouter(j.URL)
	now := time.Now()
	r.now = func() time.Time { return now }
	ctx := context.Background()

	r.Decide(ctx, turn("a"))
	if d := r.Decide(ctx, turn("a")); !d.CacheHit || j.hits.Load() != 1 {
		t.Fatalf("repeat turn not cached: hit=%v calls=%d", d.CacheHit, j.hits.Load())
	}
	r.Decide(ctx, turn("b"))
	r.Decide(ctx, turn("c")) // evicts "a" (2 entries)
	r.Decide(ctx, turn("a"))
	if n := j.hits.Load(); n != 4 {
		t.Errorf("LRU eviction: %d calls, want 4", n)
	}
	now = now.Add(2 * time.Hour)
	r.Decide(ctx, turn("a"))
	if n := j.hits.Load(); n != 5 {
		t.Errorf("expired entry reused: %d calls, want 5", n)
	}
}

func TestDecideSharesInflight(t *testing.T) {
	j := newStub(t, 0.9)
	j.gate = make(chan struct{})
	r := newTestRouter(j.URL)
	var wg sync.WaitGroup
	for range 5 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if d := r.Decide(context.Background(), turn("same")); d.Target != "t2" {
				t.Errorf("target %s", d.Target)
			}
		}()
	}
	time.Sleep(50 * time.Millisecond)
	close(j.gate)
	wg.Wait()
	if n := j.hits.Load(); n != 1 {
		t.Errorf("%d classifier calls for concurrent same-turn requests, want 1", n)
	}
}

type badAuth struct{}

func (badAuth) Apply(_ context.Context, req *http.Request) error {
	req.Header.Set("Authorization", "Bearer nope")
	return nil
}

func TestDecideErrors(t *testing.T) {
	j := newStub(t, 0.9)
	r := newTestRouter(j.URL)
	r.cfg.Auth = badAuth{}
	if d := r.Decide(context.Background(), turn("x")); d.Target != "t2" || d.Err == nil {
		t.Errorf("401 should fall to on_error: %+v", d)
	}

	slow := newStub(t, 0.9)
	slow.gate = make(chan struct{})
	defer close(slow.gate)
	r = newTestRouter(slow.URL)
	r.cfg.Timeout = 20 * time.Millisecond
	if d := r.Decide(context.Background(), turn("x")); d.Target != "t2" || d.Err == nil {
		t.Errorf("timeout should fall to on_error: %+v", d)
	}
}
