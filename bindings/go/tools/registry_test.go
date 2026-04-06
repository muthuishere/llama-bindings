package tools_test

import (
	"testing"

	"github.com/muthuishere/llama-bindings/go/llama"
	"github.com/muthuishere/llama-bindings/go/tools"
)

func TestRegistryRegisterAndExecute(t *testing.T) {
	r := tools.New()

	def := llama.ToolDefinition{
		Name:        "add",
		Description: "Add two numbers",
		Parameters:  map[string]interface{}{"type": "object"},
	}
	if err := r.Register(def, func(args map[string]interface{}) (interface{}, error) {
		a, _ := args["a"].(float64)
		b, _ := args["b"].(float64)
		return a + b, nil
	}); err != nil {
		t.Fatalf("Register: %v", err)
	}

	result, err := r.Execute("add", map[string]interface{}{"a": 2.0, "b": 3.0})
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	if result.(float64) != 5.0 {
		t.Fatalf("expected 5, got %v", result)
	}
}

func TestRegistryUnknownTool(t *testing.T) {
	r := tools.New()
	_, err := r.Execute("nonexistent", nil)
	if err == nil {
		t.Fatal("expected error for unknown tool")
	}
}

func TestRegistryDefinitions(t *testing.T) {
	r := tools.New()
	for _, name := range []string{"a", "b", "c"} {
		n := name
		_ = r.Register(llama.ToolDefinition{Name: n}, func(_ map[string]interface{}) (interface{}, error) {
			return n, nil
		})
	}
	defs := r.Definitions()
	if len(defs) != 3 {
		t.Fatalf("expected 3 definitions, got %d", len(defs))
	}
	if defs[0].Name != "a" || defs[1].Name != "b" || defs[2].Name != "c" {
		t.Fatalf("unexpected order: %v", defs)
	}
}

func TestRegistryEmptyName(t *testing.T) {
	r := tools.New()
	err := r.Register(llama.ToolDefinition{Name: ""}, func(_ map[string]interface{}) (interface{}, error) {
		return nil, nil
	})
	if err == nil {
		t.Fatal("expected error for empty tool name")
	}
}

func TestRegistryNilHandler(t *testing.T) {
	r := tools.New()
	err := r.Register(llama.ToolDefinition{Name: "test"}, nil)
	if err == nil {
		t.Fatal("expected error for nil handler")
	}
}

func TestRegistryHas(t *testing.T) {
	r := tools.New()
	if r.Has("foo") {
		t.Fatal("expected false for unregistered tool")
	}
	_ = r.Register(llama.ToolDefinition{Name: "foo"}, func(_ map[string]interface{}) (interface{}, error) {
		return nil, nil
	})
	if !r.Has("foo") {
		t.Fatal("expected true after registering foo")
	}
}
