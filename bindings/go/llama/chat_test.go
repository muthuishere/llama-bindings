package llama_test

import (
	"testing"

	"github.com/muthuishere/llama-bindings/go/llama"
)

// TestChatEngineCloseIdempotent verifies that calling Close twice does not panic.
func TestChatEngineCloseIdempotent(t *testing.T) {
	chat, err := llama.NewChat("testdata/dummy.gguf", llama.LoadOptions{})
	if err != nil {
		t.Skipf("model not available: %v", err)
	}
	chat.Close()
	chat.Close() // must not panic
}

// TestChatAfterClose returns ENGINE_CLOSED error.
func TestChatAfterClose(t *testing.T) {
	chat, err := llama.NewChat("testdata/dummy.gguf", llama.LoadOptions{})
	if err != nil {
		t.Skipf("model not available: %v", err)
	}
	chat.Close()

	_, err = chat.Chat(llama.ChatRequest{
		Messages: []llama.Message{{Role: "user", Content: "hello"}},
	}, llama.ChatOptions{})

	if err == nil {
		t.Fatal("expected error after close, got nil")
	}
	ll, ok := err.(*llama.LlamaError)
	if !ok {
		t.Fatalf("expected *llama.LlamaError, got %T", err)
	}
	if ll.Code != llama.ErrCodeEngineClosed {
		t.Fatalf("expected ENGINE_CLOSED, got %s", ll.Code)
	}
}

// TestChatInvalidModelPath verifies NewChat returns an error for a bad path.
func TestChatInvalidModelPath(t *testing.T) {
	_, err := llama.NewChat("/nonexistent/model.gguf", llama.LoadOptions{})
	if err == nil {
		t.Fatal("expected error for invalid model path")
	}
}

// TestChatMissingMessages verifies that an empty Messages slice is rejected.
func TestChatMissingMessages(t *testing.T) {
	chat, err := llama.NewChat("testdata/dummy.gguf", llama.LoadOptions{})
	if err != nil {
		t.Skipf("model not available: %v", err)
	}
	defer chat.Close()

	_, err = chat.Chat(llama.ChatRequest{}, llama.ChatOptions{})
	if err == nil {
		t.Fatal("expected error for missing messages")
	}
}

// TestChatTextMode exercises the text response mode.
func TestChatTextMode(t *testing.T) {
	chat, err := llama.NewChat("testdata/dummy.gguf", llama.LoadOptions{})
	if err != nil {
		t.Skipf("model not available: %v", err)
	}
	defer chat.Close()

	resp, err := chat.Chat(llama.ChatRequest{
		Messages: []llama.Message{
			{Role: "user", Content: "Say hello in one short sentence."},
		},
		ResponseMode: llama.ResponseModeText,
		Generation: llama.GenerationOptions{
			Temperature:     0.2,
			MaxOutputTokens: 64,
		},
	}, llama.ChatOptions{})
	if err != nil {
		t.Fatalf("chat error: %v", err)
	}
	if resp.Type != "assistant_text" {
		t.Fatalf("expected assistant_text, got %s", resp.Type)
	}
	if resp.Text == "" {
		t.Fatal("expected non-empty text response")
	}
}

// TestChatSchemaMode exercises the json_schema response mode.
func TestChatSchemaMode(t *testing.T) {
	chat, err := llama.NewChat("testdata/dummy.gguf", llama.LoadOptions{})
	if err != nil {
		t.Skipf("model not available: %v", err)
	}
	defer chat.Close()

	resp, err := chat.Chat(llama.ChatRequest{
		Messages: []llama.Message{
			{Role: "user", Content: "John is 32 years old. Extract structured data."},
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
		Generation: llama.GenerationOptions{Temperature: 0.0, MaxOutputTokens: 64},
	}, llama.ChatOptions{})
	if err != nil {
		t.Fatalf("chat error: %v", err)
	}
	if resp.Type != "structured_json" {
		t.Fatalf("expected structured_json, got %s", resp.Type)
	}
}

// TestChatToolCallMode exercises the tool_call response mode.
func TestChatToolCallMode(t *testing.T) {
	chat, err := llama.NewChat("testdata/dummy.gguf", llama.LoadOptions{})
	if err != nil {
		t.Skipf("model not available: %v", err)
	}
	defer chat.Close()

	resp, err := chat.Chat(llama.ChatRequest{
		Messages: []llama.Message{
			{Role: "user", Content: "What is the weather in Chennai?"},
		},
		ResponseMode: llama.ResponseModeToolCall,
		Tools: []llama.ToolDefinition{
			{
				Name:        "lookup_weather",
				Description: "Get weather by city",
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
		Generation: llama.GenerationOptions{Temperature: 0.2, MaxOutputTokens: 64},
	}, llama.ChatOptions{})
	if err != nil {
		t.Fatalf("chat error: %v", err)
	}
	if resp.Type != "tool_call" {
		t.Fatalf("expected tool_call, got %s", resp.Type)
	}
}

// TestChatEventCallback verifies that load and chat events are emitted.
func TestChatEventCallback(t *testing.T) {
	var events []string
	onEvent := func(e llama.Event) {
		events = append(events, e.Event)
	}

	chat, err := llama.NewChat("testdata/dummy.gguf", llama.LoadOptions{OnEvent: onEvent})
	if err != nil {
		t.Skipf("model not available: %v", err)
	}
	defer chat.Close()

	_, _ = chat.Chat(llama.ChatRequest{
		Messages: []llama.Message{{Role: "user", Content: "hi"}},
	}, llama.ChatOptions{})

	if len(events) == 0 {
		t.Fatal("expected at least one event, got none")
	}
}
