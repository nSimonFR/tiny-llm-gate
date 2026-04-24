package auth

import (
	"context"
	"net/http"
	"os"
	"path/filepath"
	"testing"
)

func TestBearerApply(t *testing.T) {
	req, _ := http.NewRequest(http.MethodGet, "http://x/", nil)
	if err := (Bearer{Token: "abc"}).Apply(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if got := req.Header.Get("Authorization"); got != "Bearer abc" {
		t.Errorf("Authorization = %q", got)
	}
}

func TestBearerEmptyTokenNoop(t *testing.T) {
	req, _ := http.NewRequest(http.MethodGet, "http://x/", nil)
	if err := (Bearer{Token: ""}).Apply(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if got := req.Header.Get("Authorization"); got != "" {
		t.Errorf("expected no auth header, got %q", got)
	}
}

func TestFileBearerRereadsPerRequest(t *testing.T) {
	// Write initial token, build FileBearer, Apply(), verify.
	// Then rewrite file with a new token, Apply() again, verify the
	// request picks up the new value without rebuilding the Authenticator.
	dir := t.TempDir()
	path := filepath.Join(dir, "token")
	if err := os.WriteFile(path, []byte("first\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	b, err := Build(&AuthConfig{Type: "bearer", TokenFile: path})
	if err != nil {
		t.Fatalf("Build: %v", err)
	}

	req1, _ := http.NewRequest(http.MethodGet, "http://x/", nil)
	if err := b.Apply(context.Background(), req1); err != nil {
		t.Fatal(err)
	}
	if got := req1.Header.Get("Authorization"); got != "Bearer first" {
		t.Errorf("first request: Authorization = %q", got)
	}

	// Rotate the token on disk.
	if err := os.WriteFile(path, []byte("second\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	req2, _ := http.NewRequest(http.MethodGet, "http://x/", nil)
	if err := b.Apply(context.Background(), req2); err != nil {
		t.Fatal(err)
	}
	if got := req2.Header.Get("Authorization"); got != "Bearer second" {
		t.Errorf("rotated request: Authorization = %q (expected Bearer second)", got)
	}
}

func TestFileBearerMissingFileErrorsAtBuild(t *testing.T) {
	if _, err := Build(&AuthConfig{Type: "bearer", TokenFile: "/nonexistent/path"}); err == nil {
		t.Error("expected error for missing token_file at Build time")
	}
}

func TestFileBearerErrorsWhenFileDisappears(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "token")
	if err := os.WriteFile(path, []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}
	b, err := Build(&AuthConfig{Type: "bearer", TokenFile: path})
	if err != nil {
		t.Fatal(err)
	}
	// Delete the file — subsequent Apply() must return an error, not
	// silently send an empty bearer.
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	req, _ := http.NewRequest(http.MethodGet, "http://x/", nil)
	if err := b.Apply(context.Background(), req); err == nil {
		t.Error("expected error when token file disappears")
	}
}
