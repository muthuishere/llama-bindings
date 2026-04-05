//go:build e2e

// Package llama_test contains end-to-end tests that exercise the full
// Go → purego → native bridge → llama.cpp stack with a real GGUF model.
//
// Prerequisites:
//   - Native bridge built:  task build-bridge
//   - Model downloaded:     task download-model
//
// Run:
//
//	go test -tags e2e ./tests/e2e/go/... -v
//	# or via Taskfile:
//	task e2e-go
//
// Model path defaults to tests/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
// Override with LLAMA_CHAT_MODEL and LLAMA_EMBED_MODEL env vars.
package e2e_test

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/muthuishere/llama-bindings/go/llama"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func chatModelPath(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("LLAMA_CHAT_MODEL"); p != "" {
		return p
	}
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..", "..", "..")
	return filepath.Join(root, "tests", "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")
}

func embedModelPath(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("LLAMA_EMBED_MODEL"); p != "" {
		return p
	}
	// Reuse the chat model for embedding in tests (Qwen2.5 supports both).
	return chatModelPath(t)
}

func newChatOrSkip(t *testing.T, opts llama.LoadOptions) *llama.ChatEngine {
	t.Helper()
	chat, err := llama.NewChat(chatModelPath(t), opts)
	if err != nil {
		t.Skipf("chat engine not available (build bridge + download model): %v", err)
	}
	t.Cleanup(chat.Close)
	return chat
}

func newEmbedOrSkip(t *testing.T, opts llama.LoadOptions) *llama.EmbedEngine {
	t.Helper()
	embed, err := llama.NewEmbed(embedModelPath(t), opts)
	if err != nil {
		t.Skipf("embed engine not available (build bridge + download model): %v", err)
	}
	t.Cleanup(embed.Close)
	return embed
}

// ---------------------------------------------------------------------------
// Chat — text mode
// ---------------------------------------------------------------------------

func TestE2EChatTextMode(t *testing.T) {
	chat := newChatOrSkip(t, llama.LoadOptions{})

	resp, err := chat.Chat(llama.ChatRequest{
		Messages: []llama.Message{
			{Role: "user", Content: "Say hello in exactly one word."},
		},
		ResponseMode: llama.ResponseModeText,
		Generation: llama.GenerationOptions{
			Temperature:     0.1,
			MaxOutputTokens: 32,
		},
	}, llama.ChatOptions{})

	if err != nil {
		t.Fatalf("chat error: %v", err)
	}
	if resp.Type != "assistant_text" {
		t.Fatalf("expected assistant_text, got %q", resp.Type)
	}
	if strings.TrimSpace(resp.Text) == "" {
		t.Fatal("expected non-empty text response")
	}
	t.Logf("Response: %s", resp.Text)
}

// ---------------------------------------------------------------------------
// Chat — JSON schema mode
// ---------------------------------------------------------------------------

func TestE2EChatSchemaMode(t *testing.T) {
	chat := newChatOrSkip(t, llama.LoadOptions{})

	resp, err := chat.Chat(llama.ChatRequest{
		Messages: []llama.Message{
			{Role: "user", Content: "Alice is 30 years old. Extract structured data."},
		},
		ResponseMode: llama.ResponseModeJSONSchema,
		Schema: &llama.Schema{
			Name: "person_extract",
			Schema: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"name": map[string]interface{}{"type": "string"},
					"age":  map[string]interface{}{"type": "integer"},
				},
				"required":             []string{"name", "age"},
				"additionalProperties": false,
			},
		},
		Generation: llama.GenerationOptions{
			Temperature:     0.0,
			MaxOutputTokens: 64,
		},
	}, llama.ChatOptions{})

	if err != nil {
		t.Fatalf("chat schema error: %v", err)
	}
	if resp.Type != "structured_json" {
		t.Fatalf("expected structured_json, got %q", resp.Type)
	}
	if resp.JSON == nil {
		t.Fatal("expected non-nil JSON field")
	}
	t.Logf("Structured JSON: %v", resp.JSON)
}

// ---------------------------------------------------------------------------
// Chat — tool call mode
// ---------------------------------------------------------------------------

func TestE2EChatToolCallMode(t *testing.T) {
	chat := newChatOrSkip(t, llama.LoadOptions{})

	resp, err := chat.Chat(llama.ChatRequest{
		Messages: []llama.Message{
			{Role: "user", Content: "What is the weather in Chennai?"},
		},
		ResponseMode: llama.ResponseModeToolCall,
		Tools: []llama.ToolDefinition{
			{
				Name:        "lookup_weather",
				Description: "Get current weather for a city",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"city": map[string]interface{}{"type": "string"},
					},
					"required":             []string{"city"},
					"additionalProperties": false,
				},
			},
		},
		ToolChoice: "auto",
		Generation: llama.GenerationOptions{
			Temperature:     0.1,
			MaxOutputTokens: 64,
		},
	}, llama.ChatOptions{})

	if err != nil {
		t.Fatalf("chat tool_call error: %v", err)
	}
	if resp.Type != "tool_call" {
		t.Fatalf("expected tool_call, got %q", resp.Type)
	}
	if len(resp.ToolCalls) == 0 {
		t.Fatal("expected at least one tool call")
	}
	t.Logf("Tool calls: %+v", resp.ToolCalls)
}

// ---------------------------------------------------------------------------
// Chat — observability events during real inference
// ---------------------------------------------------------------------------

func TestE2EChatObservabilityEvents(t *testing.T) {
	var events []string
	onEvent := func(e llama.Event) {
		events = append(events, e.Event)
	}

	chat := newChatOrSkip(t, llama.LoadOptions{OnEvent: onEvent})

	_, err := chat.Chat(llama.ChatRequest{
		Messages: []llama.Message{
			{Role: "user", Content: "Say hello."},
		},
		Generation: llama.GenerationOptions{Temperature: 0.1, MaxOutputTokens: 16},
	}, llama.ChatOptions{})
	if err != nil {
		t.Fatalf("chat error: %v", err)
	}

	if len(events) == 0 {
		t.Fatal("expected at least one observability event")
	}
	t.Logf("Events received: %v", events)
}

// ---------------------------------------------------------------------------
// Embed
// ---------------------------------------------------------------------------

func TestE2EEmbedReturnsNonEmptyVector(t *testing.T) {
	embed := newEmbedOrSkip(t, llama.LoadOptions{})

	vec, err := embed.Embed("semantic search example", llama.EmbedOptions{})
	if err != nil {
		t.Fatalf("embed error: %v", err)
	}
	if len(vec) == 0 {
		t.Fatal("expected non-empty float vector")
	}
	t.Logf("Vector length: %d, first value: %f", len(vec), vec[0])
}

func TestE2EEmbedRepeatedCallsConsistent(t *testing.T) {
	embed := newEmbedOrSkip(t, llama.LoadOptions{})

	const input = "hello world"
	var firstLen int
	for i := 0; i < 3; i++ {
		vec, err := embed.Embed(input, llama.EmbedOptions{})
		if err != nil {
			t.Fatalf("embed error on iteration %d: %v", i, err)
		}
		if len(vec) == 0 {
			t.Fatalf("empty vector on iteration %d", i)
		}
		if i == 0 {
			firstLen = len(vec)
		} else if len(vec) != firstLen {
			t.Fatalf("vector length changed: iteration 0 had %d, iteration %d has %d",
				firstLen, i, len(vec))
		}
	}
}
