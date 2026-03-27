package llama_test

import (
	"os"
	"strings"
	"testing"

	llama "github.com/muthuishere/llama-bindings/bindings/go/llama"
)

// modelPath returns the test model path from the environment variable
// LLAMA_TEST_MODEL. Tests that require an actual model are skipped
// when the variable is not set.
func modelPath(t *testing.T) string {
	t.Helper()
	p := os.Getenv("LLAMA_TEST_MODEL")
	if p == "" {
		t.Skip("LLAMA_TEST_MODEL not set — skipping integration test")
	}
	return p
}

// ---------------------------------------------------------------------------
// Unit-style tests — no real model required
// ---------------------------------------------------------------------------

// TestLoad_EmptyPath ensures that an empty model path returns a descriptive
// error and does not panic.
func TestLoad_EmptyPath(t *testing.T) {
	_, err := llama.Load("")
	if err == nil {
		t.Fatal("expected error for empty model path, got nil")
	}
}

// TestLoad_InvalidPath ensures that a non-existent model path returns an
// error and does not crash the process.
func TestLoad_InvalidPath(t *testing.T) {
	_, err := llama.Load("/nonexistent/path/model.gguf")
	if err == nil {
		t.Fatal("expected error for invalid model path, got nil")
	}
}

// TestComplete_NilEngine ensures that calling Complete on a zero-value
// Engine returns an error without panicking.
func TestComplete_NilEngine(t *testing.T) {
	var e *llama.Engine
	_, err := e.Complete("hello")
	if err == nil {
		t.Fatal("expected error from nil engine, got nil")
	}
}

// TestClose_NilEngine ensures that closing a nil Engine is a no-op.
func TestClose_NilEngine(t *testing.T) {
	var e *llama.Engine
	if err := e.Close(); err != nil {
		t.Fatalf("unexpected error closing nil engine: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Integration tests — require LLAMA_TEST_MODEL to be set
// ---------------------------------------------------------------------------

// TestSmoke verifies the basic load → complete → close lifecycle.
func TestSmoke(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer func() {
		if err := engine.Close(); err != nil {
			t.Errorf("Close returned unexpected error: %v", err)
		}
	}()

	out, err := engine.Complete("Say hello in one short sentence.")
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
	if strings.TrimSpace(out) == "" {
		t.Fatal("expected non-empty completion")
	}
	t.Logf("completion: %s", out)
}

// TestFactual verifies that a factual prompt returns useful text.
func TestFactual(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	out, err := engine.Complete("Complete this: The capital of France is")
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
	if strings.TrimSpace(out) == "" {
		t.Fatal("expected non-empty completion for factual prompt")
	}
	t.Logf("factual completion: %s", out)
}

// TestEmptyPrompt verifies that an empty prompt is handled safely.
func TestEmptyPrompt(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	// Should not panic; result may be empty or an error.
	out, err := engine.Complete("")
	t.Logf("empty prompt result: %q err: %v", out, err)
}

// TestRepeatedCompletions verifies that the same engine handles multiple
// completion calls without crashing or corrupting state.
func TestRepeatedCompletions(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	for i := 0; i < 5; i++ {
		out, err := engine.Complete("Say hello.")
		if err != nil {
			t.Fatalf("Complete call %d failed: %v", i, err)
		}
		if strings.TrimSpace(out) == "" {
			t.Fatalf("Complete call %d returned empty result", i)
		}
		t.Logf("call %d: %s", i, out)
	}
}

// TestCreateDestroyCycle verifies that repeated create/destroy cycles do
// not crash.
func TestCreateDestroyCycle(t *testing.T) {
	path := modelPath(t)
	for i := 0; i < 3; i++ {
		engine, err := llama.Load(path)
		if err != nil {
			t.Fatalf("Load cycle %d failed: %v", i, err)
		}
		if err := engine.Close(); err != nil {
			t.Fatalf("Close cycle %d failed: %v", i, err)
		}
	}
}
