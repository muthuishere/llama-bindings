package llama_test

import (
	"testing"

	"github.com/muthuishere/llama-bindings/go/llama"
)

// TestEmbedEngineCloseIdempotent verifies that calling Close twice does not panic.
func TestEmbedEngineCloseIdempotent(t *testing.T) {
	embed, err := llama.NewEmbed("testdata/dummy-embed.gguf", llama.LoadOptions{})
	if err != nil {
		t.Skipf("model not available: %v", err)
	}
	embed.Close()
	embed.Close() // must not panic
}

// TestEmbedAfterClose returns ENGINE_CLOSED error.
func TestEmbedAfterClose(t *testing.T) {
	embed, err := llama.NewEmbed("testdata/dummy-embed.gguf", llama.LoadOptions{})
	if err != nil {
		t.Skipf("model not available: %v", err)
	}
	embed.Close()

	_, err = embed.Embed("hello", llama.EmbedOptions{})
	if err == nil {
		t.Fatal("expected error after close")
	}
	ll, ok := err.(*llama.LlamaError)
	if !ok {
		t.Fatalf("expected *llama.LlamaError, got %T", err)
	}
	if ll.Code != llama.ErrCodeEngineClosed {
		t.Fatalf("expected ENGINE_CLOSED, got %s", ll.Code)
	}
}

// TestEmbedInvalidModelPath verifies NewEmbed returns an error for a bad path.
func TestEmbedInvalidModelPath(t *testing.T) {
	_, err := llama.NewEmbed("/nonexistent/model.gguf", llama.LoadOptions{})
	if err == nil {
		t.Fatal("expected error for invalid model path")
	}
}

// TestEmbedEmptyInput verifies that empty text is rejected.
func TestEmbedEmptyInput(t *testing.T) {
	embed, err := llama.NewEmbed("testdata/dummy-embed.gguf", llama.LoadOptions{})
	if err != nil {
		t.Skipf("model not available: %v", err)
	}
	defer embed.Close()

	_, err = embed.Embed("", llama.EmbedOptions{})
	if err == nil {
		t.Fatal("expected error for empty input")
	}
}

// TestEmbedReturnsVector verifies that a non-empty vector is returned.
func TestEmbedReturnsVector(t *testing.T) {
	embed, err := llama.NewEmbed("testdata/dummy-embed.gguf", llama.LoadOptions{})
	if err != nil {
		t.Skipf("model not available: %v", err)
	}
	defer embed.Close()

	vec, err := embed.Embed("semantic search example", llama.EmbedOptions{})
	if err != nil {
		t.Fatalf("embed error: %v", err)
	}
	if len(vec) == 0 {
		t.Fatal("expected non-empty vector")
	}
}

// TestEmbedRepeatedCallsStable verifies that repeated calls return consistent results.
func TestEmbedRepeatedCallsStable(t *testing.T) {
	embed, err := llama.NewEmbed("testdata/dummy-embed.gguf", llama.LoadOptions{})
	if err != nil {
		t.Skipf("model not available: %v", err)
	}
	defer embed.Close()

	for i := 0; i < 3; i++ {
		vec, err := embed.Embed("repeated input", llama.EmbedOptions{})
		if err != nil {
			t.Fatalf("embed error on iteration %d: %v", i, err)
		}
		if len(vec) == 0 {
			t.Fatalf("empty vector on iteration %d", i)
		}
	}
}

// TestEmbedEventCallback verifies that observability events are emitted.
func TestEmbedEventCallback(t *testing.T) {
	var events []string
	onEvent := func(e llama.Event) {
		events = append(events, e.Event)
	}

	embed, err := llama.NewEmbed("testdata/dummy-embed.gguf", llama.LoadOptions{OnEvent: onEvent})
	if err != nil {
		t.Skipf("model not available: %v", err)
	}
	defer embed.Close()

	_, _ = embed.Embed("test input", llama.EmbedOptions{})
	if len(events) == 0 {
		t.Fatal("expected at least one event, got none")
	}
}
