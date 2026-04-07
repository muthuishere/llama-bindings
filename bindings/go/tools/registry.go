// Package tools provides a ToolRegistry that maps tool names to their
// definitions and Go handler functions.
//
// The registry is safe for concurrent read access; writes (Register) should
// be done before the Agent starts handling requests.
package tools

import (
	"fmt"
	"sync"

	"github.com/muthuishere/llama-bindings/go/llama"
)

// Handler is a function that executes a tool call.
// args contains the decoded JSON arguments from the model.
// The return value is serialised back to JSON and forwarded to the model.
type Handler func(args map[string]interface{}) (interface{}, error)

// Registry maps tool names to their definitions and handlers.
type Registry struct {
	mu       sync.RWMutex
	tools    map[string]entry
	ordered  []llama.ToolDefinition // preserves registration order
}

type entry struct {
	def     llama.ToolDefinition
	handler Handler
}

// New creates an empty ToolRegistry.
func New() *Registry {
	return &Registry{tools: make(map[string]entry)}
}

// Register adds (or replaces) a tool.
// def.Name must be non-empty and unique; handler must not be nil.
func (r *Registry) Register(def llama.ToolDefinition, handler Handler) error {
	if def.Name == "" {
		return fmt.Errorf("tools: tool name must not be empty")
	}
	if handler == nil {
		return fmt.Errorf("tools: handler for %q must not be nil", def.Name)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.tools[def.Name]; !exists {
		r.ordered = append(r.ordered, def)
	} else {
		// Update existing entry in ordered slice.
		for i := range r.ordered {
			if r.ordered[i].Name == def.Name {
				r.ordered[i] = def
				break
			}
		}
	}
	r.tools[def.Name] = entry{def: def, handler: handler}
	return nil
}

// Definitions returns all registered tool definitions in registration order.
func (r *Registry) Definitions() []llama.ToolDefinition {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]llama.ToolDefinition, len(r.ordered))
	copy(out, r.ordered)
	return out
}

// Execute calls the handler registered for name with the provided args.
// Returns an error if the tool is not found or the handler returns one.
func (r *Registry) Execute(name string, args map[string]interface{}) (interface{}, error) {
	r.mu.RLock()
	e, ok := r.tools[name]
	r.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("tools: unknown tool %q", name)
	}
	return e.handler(args)
}

// Has reports whether a tool with the given name is registered.
func (r *Registry) Has(name string) bool {
	r.mu.RLock()
	_, ok := r.tools[name]
	r.mu.RUnlock()
	return ok
}
