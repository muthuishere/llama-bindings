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

func TestLoad_EmptyPath(t *testing.T) {
	_, err := llama.Load("")
	if err == nil {
		t.Fatal("expected error for empty model path, got nil")
	}
}

func TestLoad_InvalidPath(t *testing.T) {
	_, err := llama.Load("/nonexistent/path/model.gguf")
	if err == nil {
		t.Fatal("expected error for invalid model path, got nil")
	}
}

func TestComplete_NilEngine(t *testing.T) {
	var e *llama.Engine
	_, err := e.Complete("hello")
	if err == nil {
		t.Fatal("expected error from nil engine, got nil")
	}
}

func TestClose_NilEngine(t *testing.T) {
	var e *llama.Engine
	if err := e.Close(); err != nil {
		t.Fatalf("unexpected error closing nil engine: %v", err)
	}
}

func TestChat_NilEngine(t *testing.T) {
	var e *llama.Engine
	_, err := e.Chat("sid", llama.ChatRequest{UserMessage: "hi"})
	if err == nil {
		t.Fatal("expected error from nil engine, got nil")
	}
}

func TestChat_EmptySessionID(t *testing.T) {
	var e *llama.Engine
	_, err := e.Chat("", llama.ChatRequest{UserMessage: "hi"})
	if err == nil {
		t.Fatal("expected error for empty session ID, got nil")
	}
}

func TestChatWithObject_NilEngine(t *testing.T) {
	var e *llama.Engine
	_, err := e.ChatWithObject("sid", llama.ChatRequest{UserMessage: "hi"})
	if err == nil {
		t.Fatal("expected error from nil engine, got nil")
	}
}

func TestChatWithObject_EmptySessionID(t *testing.T) {
	var e *llama.Engine
	_, err := e.ChatWithObject("", llama.ChatRequest{UserMessage: "hi"})
	if err == nil {
		t.Fatal("expected error for empty session ID, got nil")
	}
}

// ---------------------------------------------------------------------------
// Integration tests — require LLAMA_TEST_MODEL
// ---------------------------------------------------------------------------

func TestSmoke(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	out, err := engine.Complete("Say hello in one short sentence.")
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
	if strings.TrimSpace(out) == "" {
		t.Fatal("expected non-empty completion")
	}
	t.Logf("completion: %s", out)
}

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

func TestEmptyPrompt(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	out, err := engine.Complete("")
	t.Logf("empty prompt result: %q err: %v", out, err)
}

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
	}
}

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

func TestChat_WithSystem_ReturnsChatMessage(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	msg, err := engine.Chat("sid-go-1", llama.ChatRequest{
		SystemMessage: "You are a helpful assistant.",
		UserMessage:   "Say hello.",
	})
	if err != nil {
		t.Fatalf("Chat failed: %v", err)
	}
	if msg.Role != "assistant" {
		t.Errorf("expected role=assistant, got %q", msg.Role)
	}
	if strings.TrimSpace(msg.Content) == "" {
		t.Fatal("expected non-empty content")
	}
	t.Logf("Chat response: role=%s content=%s", msg.Role, msg.Content)
}

func TestChat_NoSystem_ReturnsChatMessage(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	msg, err := engine.Chat("sid-go-2", llama.ChatRequest{
		UserMessage: "Say hello.",
	})
	if err != nil {
		t.Fatalf("Chat (no system) failed: %v", err)
	}
	if msg.Role != "assistant" {
		t.Errorf("expected role=assistant, got %q", msg.Role)
	}
	if strings.TrimSpace(msg.Content) == "" {
		t.Fatal("expected non-empty content")
	}
	t.Logf("Chat (no system) response: %s", msg.Content)
}

func TestChat_MultiTurn(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	sid := "sid-go-mt"

	turn1, err := engine.Chat(sid, llama.ChatRequest{
		SystemMessage: "You are a helpful assistant.",
		UserMessage:   "Say hello.",
	})
	if err != nil {
		t.Fatalf("Chat turn 1 failed: %v", err)
	}
	t.Logf("Turn 1: %s", turn1.Content)

	turn2, err := engine.Chat(sid, llama.ChatRequest{
		UserMessage: "What did you just say?",
	})
	if err != nil {
		t.Fatalf("Chat turn 2 failed: %v", err)
	}
	if strings.TrimSpace(turn2.Content) == "" {
		t.Fatal("expected non-empty content for turn 2")
	}
	t.Logf("Turn 2: %s", turn2.Content)
}

func TestChat_WithAssistantMessage(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	msg, err := engine.Chat("sid-go-asst", llama.ChatRequest{
		SystemMessage:    "You are helpful.",
		AssistantMessage: "I said hello earlier.",
		UserMessage:      "What did you say before?",
	})
	if err != nil {
		t.Fatalf("Chat with AssistantMessage failed: %v", err)
	}
	if strings.TrimSpace(msg.Content) == "" {
		t.Fatal("expected non-empty content")
	}
	t.Logf("Chat with AssistantMessage: %s", msg.Content)
}

func TestChat_WithToolMessage(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	msg, err := engine.Chat("sid-go-tool", llama.ChatRequest{
		SystemMessage: "You are a helpful assistant with tool access.",
		ToolMessage:   `{"weather": "sunny", "temperature": "22C"}`,
		UserMessage:   "What is the weather like?",
	})
	if err != nil {
		t.Fatalf("Chat with ToolMessage failed: %v", err)
	}
	if strings.TrimSpace(msg.Content) == "" {
		t.Fatal("expected non-empty content")
	}
	t.Logf("Chat with ToolMessage: %s", msg.Content)
}

func TestChatWithObject_ReturnsSchemaResponse(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	sid := "sid-go-obj"
	resp, err := engine.ChatWithObject(sid, llama.ChatRequest{
		SystemMessage: "You are a helpful assistant.",
		UserMessage:   "Say hello in one sentence.",
	})
	if err != nil {
		t.Fatalf("ChatWithObject failed: %v", err)
	}
	if resp.Role != "assistant" {
		t.Errorf("expected role=assistant, got %q", resp.Role)
	}
	if strings.TrimSpace(resp.Content) == "" {
		t.Fatal("expected non-empty content")
	}
	if resp.SessionID != sid {
		t.Errorf("expected sessionId=%q, got %q", sid, resp.SessionID)
	}
	if resp.MessageCount <= 0 {
		t.Errorf("expected messageCount > 0, got %d", resp.MessageCount)
	}
	t.Logf("ChatWithObject: role=%s content=%s sessionId=%s count=%d",
		resp.Role, resp.Content, resp.SessionID, resp.MessageCount)
}

func TestChatWithObject_MultiTurn_MessageCountGrows(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	sid := "sid-go-obj-mt"

	r1, err := engine.ChatWithObject(sid, llama.ChatRequest{UserMessage: "Hello."})
	if err != nil {
		t.Fatalf("ChatWithObject turn 1 failed: %v", err)
	}

	r2, err := engine.ChatWithObject(sid, llama.ChatRequest{UserMessage: "How are you?"})
	if err != nil {
		t.Fatalf("ChatWithObject turn 2 failed: %v", err)
	}

	if r2.MessageCount <= r1.MessageCount {
		t.Errorf("expected messageCount to grow: turn1=%d turn2=%d",
			r1.MessageCount, r2.MessageCount)
	}
	t.Logf("messageCount: turn1=%d turn2=%d", r1.MessageCount, r2.MessageCount)
}

func TestChatSessionClear_ResetsHistory(t *testing.T) {
	engine, err := llama.Load(modelPath(t))
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer engine.Close()

	sid := "sid-go-clear"
	_, err = engine.Chat(sid, llama.ChatRequest{UserMessage: "Say hello."})
	if err != nil {
		t.Fatalf("Chat before clear failed: %v", err)
	}

	engine.ChatSessionClear(sid)

	msg, err := engine.Chat(sid, llama.ChatRequest{UserMessage: "Say hello again."})
	if err != nil {
		t.Fatalf("Chat after clear failed: %v", err)
	}
	if strings.TrimSpace(msg.Content) == "" {
		t.Fatal("expected non-empty content after clear")
	}
	t.Logf("After clear: %s", msg.Content)
}
