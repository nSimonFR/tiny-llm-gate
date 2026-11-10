package config

import "testing"

func TestExampleConfigParses(t *testing.T) {
	if _, err := Load("../../testdata/example-config.yaml"); err != nil {
		t.Fatal(err)
	}
}
