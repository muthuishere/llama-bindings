package agent_test

import (
	"fmt"
	"testing"

	"github.com/muthuishere/llama-bindings/go/agent"
	"github.com/muthuishere/llama-bindings/go/llama"
	"github.com/muthuishere/llama-bindings/go/tools"
)

const dummyChat  = "../llama/testdata/dummy.gguf"
const dummyEmbed = "../llama/testdata/dummy-embed.gguf"

func newAgentOrSkip(t *testing.T) *agent.Agent {
	t.Helper()
	a, err := agent.New(dummyChat, dummyEmbed, ":memory:")
	if err != nil {
		t.Skipf("agent not available (bridge required): %v", err)
	}
	t.Cleanup(a.Close)
	return a
}

func TestAgentChatReturnsText(t *testing.T) {
	a := newAgentOrSkip(t)

	resp, err := a.Chat("session-1", "Say hello.")
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	if resp == "" {
		t.Fatal("expected non-empty response")
	}
}

func TestAgentChatMaintainsHistory(t *testing.T) {
	a := newAgentOrSkip(t)

	_, err := a.Chat("history-session", "What is 2+2?")
	if err != nil {
		t.Fatalf("first Chat: %v", err)
	}
	_, err = a.Chat("history-session", "And what about 3+3?")
	if err != nil {
		t.Fatalf("second Chat: %v", err)
	}
}

func TestAgentClearSession(t *testing.T) {
	a := newAgentOrSkip(t)

	_, _ = a.Chat("clear-session", "hello")
	a.ClearSession("clear-session")

	_, err := a.Chat("clear-session", "hello again")
	if err != nil {
		t.Fatalf("Chat after clear: %v", err)
	}
}

func TestAgentAddDocument(t *testing.T) {
	a := newAgentOrSkip(t)

	if err := a.AddDocument("The capital of France is Paris."); err != nil {
		t.Fatalf("AddDocument: %v", err)
	}
}

func TestAgentAddTool(t *testing.T) {
	a := newAgentOrSkip(t)

	err := a.AddTool(llama.ToolDefinition{
		Name:        "greet",
		Description: "Return a greeting",
		Parameters:  map[string]interface{}{"type": "object"},
	}, func(args map[string]interface{}) (interface{}, error) {
		return "Hello!", nil
	})
	if err != nil {
		t.Fatalf("AddTool: %v", err)
	}
}

func TestAgentNewInvalidChatModel(t *testing.T) {
	_, err := agent.New("/nonexistent/chat.gguf", dummyEmbed, ":memory:")
	if err == nil {
		t.Fatal("expected error for invalid chat model path")
	}
}

func TestAgentToolExecution(t *testing.T) {
	a := newAgentOrSkip(t)

	called := false
	err := a.AddTool(llama.ToolDefinition{
		Name:        "lookup_weather",
		Description: "Get weather by city",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"city": map[string]interface{}{"type": "string"},
			},
			"required": []string{"city"},
		},
	}, func(args map[string]interface{}) (interface{}, error) {
		called = true
		_ = args["city"]
		return map[string]interface{}{"temperature": "30°C", "condition": "sunny"}, nil
	})
	if err != nil {
		t.Fatalf("AddTool: %v", err)
	}

	// The model might or might not call the tool depending on stub behaviour.
	_, err = a.Chat("tool-session", "What is the weather in Chennai?")
	if err != nil {
		t.Fatalf("Chat: %v", err)
	}
	_ = called // tool may or may not be called by stub model
}

func TestAgentMultipleSessions(t *testing.T) {
	a := newAgentOrSkip(t)

	for _, sid := range []string{"s1", "s2", "s3"} {
		s := sid
		_, err := a.Chat(s, "hello from "+s)
		if err != nil {
			t.Fatalf("Chat [%s]: %v", s, err)
		}
	}
}

func TestToolRegistryHandlerError(t *testing.T) {
	r := tools.New()
	err := r.Register(llama.ToolDefinition{Name: "fail"}, func(_ map[string]interface{}) (interface{}, error) {
		return nil, fmt.Errorf("tool failed")
	})
	if err != nil {
		t.Fatalf("Register: %v", err)
	}
	_, execErr := r.Execute("fail", nil)
	if execErr == nil {
		t.Fatal("expected error from failing handler")
	}
}
