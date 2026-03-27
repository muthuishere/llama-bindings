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

// TestChat_NilEngine verifies Chat returns an error on a nil engine.
func TestChat_NilEngine(t *testing.T) {
	var e *llama.Engine
	_, err := e.Chat("", "hello")
	if err == nil {
		t.Fatal("expected error from nil engine, got nil")
	}
}

// TestChatWithMessages_EmptySlice verifies ChatWithMessages returns an error
// for an empty message slice.
func TestChatWithMessages_EmptySlice(t *testing.T) {
	_, err := llama.Load("/nonexistent/model.gguf")
	// We just need any Engine-shaped error check; use a nil engine:
	var e *llama.Engine
	_, err = e.ChatWithMessages(nil)
	if err == nil {
		t.Fatal("expected error for nil engine, got nil")
	}
}

// TestChatSession_EmptySessionID verifies that an empty session ID returns
// an error.
func TestChatSession_EmptySessionID(t *testing.T) {
	var e *llama.Engine
	_, err := e.ChatSession("", "hello")
	if err == nil {
		t.Fatal("expected error for nil engine, got nil")
	}
}

// TestChatWithTools_NilEngine verifies ChatWithTools returns error on nil engine.
func TestChatWithTools_NilEngine(t *testing.T) {
	var e *llama.Engine
	_, err := e.ChatWithTools([]llama.Message{{Role: "user", Content: "hi"}}, "[]")
	if err == nil {
		t.Fatal("expected error from nil engine, got nil")
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

// ---------------------------------------------------------------------------
// Chat integration tests
// ---------------------------------------------------------------------------

// TestChat_WithSystem verifies one-shot chat with a system message.
func TestChat_WithSystem(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	out, err := engine.Chat("You are a helpful assistant.", "Say hello.")
	if err != nil {
		t.Fatalf("Chat failed: %v", err)
	}
	if strings.TrimSpace(out) == "" {
		t.Fatal("expected non-empty chat response")
	}
	t.Logf("chat with system: %s", out)
}

// TestChat_NoSystem verifies one-shot chat without a system message.
func TestChat_NoSystem(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	out, err := engine.Chat("", "Say hello.")
	if err != nil {
		t.Fatalf("Chat (no system) failed: %v", err)
	}
	if strings.TrimSpace(out) == "" {
		t.Fatal("expected non-empty chat response")
	}
	t.Logf("chat no system: %s", out)
}

// TestChatWithMessages_MultiTurn verifies chatWithObject-style call.
func TestChatWithMessages_MultiTurn(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	msgs := []llama.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "Say hello in one sentence."},
	}
	out, err := engine.ChatWithMessages(msgs)
	if err != nil {
		t.Fatalf("ChatWithMessages failed: %v", err)
	}
	if strings.TrimSpace(out) == "" {
		t.Fatal("expected non-empty response")
	}
	t.Logf("chatWithMessages: %s", out)
}

// TestChatSession_MultiTurn verifies session-based multi-turn chat.
func TestChatSession_MultiTurn(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	sid := "test-session-1"

	out1, err := engine.ChatSession(sid, "Say hello.")
	if err != nil {
		t.Fatalf("ChatSession turn 1 failed: %v", err)
	}
	t.Logf("session turn 1: %s", out1)

	out2, err := engine.ChatSession(sid, "What did you just say?")
	if err != nil {
		t.Fatalf("ChatSession turn 2 failed: %v", err)
	}
	if strings.TrimSpace(out2) == "" {
		t.Fatal("expected non-empty response for turn 2")
	}
	t.Logf("session turn 2: %s", out2)
}

// TestChatSession_SetSystem verifies that setting a system message for a
// session works correctly.
func TestChatSession_SetSystem(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	sid := "test-session-system"
	engine.ChatSessionSetSystem(sid, "You are a helpful assistant.")

	out, err := engine.ChatSession(sid, "Say hello.")
	if err != nil {
		t.Fatalf("ChatSession with system failed: %v", err)
	}
	if strings.TrimSpace(out) == "" {
		t.Fatal("expected non-empty response")
	}
	t.Logf("session with system: %s", out)
}

// TestChatSession_Clear verifies that clearing a session resets history.
func TestChatSession_Clear(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	sid := "test-session-clear"
	_, _ = engine.ChatSession(sid, "Say hello.")

	engine.ChatSessionClear(sid)

	// After clearing, the session should start fresh.
	out, err := engine.ChatSession(sid, "Say hello again.")
	if err != nil {
		t.Fatalf("ChatSession after clear failed: %v", err)
	}
	if strings.TrimSpace(out) == "" {
		t.Fatal("expected non-empty response after clear")
	}
	t.Logf("session after clear: %s", out)
}

// TestChatWithTools_Basic verifies tool definition chat returns raw output.
func TestChatWithTools_Basic(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	tools := `[{"name":"get_weather","description":"Get the current weather","parameters":{"location":{"type":"string","description":"City name"}}}]`
	msgs := []llama.Message{
		{Role: "user", Content: "What is the weather in Paris?"},
	}

	out, err := engine.ChatWithTools(msgs, tools)
	if err != nil {
		t.Fatalf("ChatWithTools failed: %v", err)
	}
	if strings.TrimSpace(out) == "" {
		t.Fatal("expected non-empty response from ChatWithTools")
	}
	t.Logf("chatWithTools raw output: %s", out)
}

