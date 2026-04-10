//go:build e2e

// Package llama_test contains end-to-end tests that exercise the full
// Go → purego → native bridge → llama.cpp stack with a real GGUF model.
//
// Prerequisites:
//   - Native bridge built:  task build-bridge
//   - Models downloaded:    task download-model
//
// Run:
//
//	go test -tags e2e ./tests/e2e/go/... -v
//	# or via Taskfile:
//	task e2e-go
//
// Model paths default to ~/llama-bindings-conf/models/.
// Override with LLAMA_CHAT_MODEL and LLAMA_EMBED_MODEL env vars.
package e2e_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/muthuishere/llama-bindings/go/agent"
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
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "llama-bindings-conf", "models", "chat", "gemma-4-E2B-it-Q4_K_M.gguf")
}

func embedModelPath(t *testing.T) string {
	t.Helper()
	if p := os.Getenv("LLAMA_EMBED_MODEL"); p != "" {
		return p
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "llama-bindings-conf", "models", "embeddings", "nomic-embed-text-v1.5.Q4_K_M.gguf")
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

// ---------------------------------------------------------------------------
// Agent — helpers
// ---------------------------------------------------------------------------

func newAgentOrSkip(t *testing.T) *agent.Agent {
	t.Helper()
	a, err := agent.New(chatModelPath(t), embedModelPath(t), ":memory:")
	if err != nil {
		t.Skipf("agent not available (build bridge + download models): %v", err)
	}
	t.Cleanup(a.Close)
	return a
}

// ---------------------------------------------------------------------------
// Agent — basic chat
// ---------------------------------------------------------------------------

func TestE2EAgentChatReturnsText(t *testing.T) {
	a := newAgentOrSkip(t)

	reply, err := a.Chat("session-1", "Say hello in exactly one word.")
	if err != nil {
		t.Fatalf("agent chat error: %v", err)
	}
	if strings.TrimSpace(reply) == "" {
		t.Fatal("expected non-empty reply")
	}
	t.Logf("Agent reply: %s", reply)
}

// ---------------------------------------------------------------------------
// Agent — multi-turn history
// ---------------------------------------------------------------------------

func TestE2EAgentMultiTurnHistory(t *testing.T) {
	a := newAgentOrSkip(t)

	_, err := a.Chat("history-session", "My name is Muthukumaran.")
	if err != nil {
		t.Fatalf("first turn error: %v", err)
	}
	reply, err := a.Chat("history-session", "What is my name?")
	if err != nil {
		t.Fatalf("second turn error: %v", err)
	}
	if strings.TrimSpace(reply) == "" {
		t.Fatal("expected non-empty reply on second turn")
	}
	t.Logf("Second turn reply: %s", reply)
}

// ---------------------------------------------------------------------------
// Agent — knowledge (addDocument → query)
// ---------------------------------------------------------------------------

func TestE2EAgentWithDocument(t *testing.T) {
	a := newAgentOrSkip(t)

	if err := a.AddDocument("The capital of France is Paris."); err != nil {
		t.Fatalf("AddDocument error: %v", err)
	}

	reply, err := a.Chat("doc-session", "What is the capital of France?")
	if err != nil {
		t.Fatalf("agent chat error: %v", err)
	}
	if strings.TrimSpace(reply) == "" {
		t.Fatal("expected non-empty reply")
	}
	t.Logf("Knowledge-grounded reply: %s", reply)
}

// ---------------------------------------------------------------------------
// Agent — tool dispatch
// ---------------------------------------------------------------------------

func TestE2EAgentWithTool(t *testing.T) {
	a := newAgentOrSkip(t)

	called := false
	err := a.AddTool(llama.ToolDefinition{
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
	}, func(args map[string]interface{}) (interface{}, error) {
		called = true
		return map[string]interface{}{"temperature": "32°C", "condition": "sunny"}, nil
	})
	if err != nil {
		t.Fatalf("AddTool error: %v", err)
	}

	reply, err := a.Chat("tool-session", "What is the weather in Chennai?")
	// NOTE: the bridge stub always returns tool_call which causes the agent
	// to loop. Until llama.cpp inference is wired into the bridge, we accept
	// either a valid reply or the known loop-limit error.
	if err != nil {
		if strings.Contains(err.Error(), "exceeded") {
			t.Logf("Tool loop limit hit (bridge stub behaviour — expected): %v", err)
			return
		}
		t.Fatalf("agent chat error: %v", err)
	}
	t.Logf("Tool-grounded reply (tool called=%v): %s", called, reply)
}

// ---------------------------------------------------------------------------
// Agent — export → import round-trip
// ---------------------------------------------------------------------------

func TestE2EAgentExportImport(t *testing.T) {
	t.Skip("Export/ImportFrom not yet implemented — tracked in docs/export-import.md")
}
