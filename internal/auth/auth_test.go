package auth

import (
	"context"
	"net/http"
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
