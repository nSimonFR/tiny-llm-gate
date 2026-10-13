package responses

import (
	"strings"
	"testing"
)

func TestToChatSkipsCodexOnlyItems(t *testing.T) {
	body := `{"model":"m","input":[{"type":"additional_tools"},{"type":"message","role":"user","content":"hi"}],
"tools":[{"type":"custom","name":"apply_patch"},{"type":"function","name":"f","parameters":{}}]}`
	req, err := Parse([]byte(body))
	if err != nil {
		t.Fatal(err)
	}
	out, err := req.ToChat("up")
	if err != nil {
		t.Fatal(err)
	}
	if s := string(out); strings.Contains(s, "apply_patch") || !strings.Contains(s, `"name":"f"`) {
		t.Fatalf("unexpected body: %s", s)
	}
}
