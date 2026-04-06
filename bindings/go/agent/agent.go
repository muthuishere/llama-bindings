// Package agent provides an agentic chat loop that combines:
//   - ChatEngine   – language model inference
//   - EmbedEngine  – text embedding for RAG
//   - KnowledgeStore – persistent vector + FTS knowledge base
//   - ToolRegistry  – callable tool dispatch
//
// The Agent maintains per-session conversation history in memory and injects
// relevant knowledge context into the system prompt on every turn.
package agent

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/muthuishere/llama-bindings/go/knowledge"
	"github.com/muthuishere/llama-bindings/go/llama"
	"github.com/muthuishere/llama-bindings/go/tools"
)

const maxToolIterations = 10

// Agent orchestrates chat, embedding, knowledge retrieval, and tool execution.
// Create with New; always call Close when done.
type Agent struct {
	chat  *llama.ChatEngine
	embed *llama.EmbedEngine
	store *knowledge.Store
	reg   *tools.Registry

	mu       sync.Mutex
	sessions map[string][]llama.Message // sessionID → history
}

// New creates an Agent that owns its ChatEngine, EmbedEngine, and
// KnowledgeStore. storagePath is passed to the KnowledgeStore (use ":memory:"
// for an ephemeral store).
func New(chatModelPath, embedModelPath, storagePath string) (*Agent, error) {
	chat, err := llama.NewChat(chatModelPath, llama.LoadOptions{})
	if err != nil {
		return nil, fmt.Errorf("agent: load chat model: %w", err)
	}

	embed, err := llama.NewEmbed(embedModelPath, llama.LoadOptions{})
	if err != nil {
		chat.Close()
		return nil, fmt.Errorf("agent: load embed model: %w", err)
	}

	store, err := knowledge.New(storagePath)
	if err != nil {
		chat.Close()
		embed.Close()
		return nil, fmt.Errorf("agent: open knowledge store: %w", err)
	}

	return &Agent{
		chat:     chat,
		embed:    embed,
		store:    store,
		reg:      tools.New(),
		sessions: make(map[string][]llama.Message),
	}, nil
}

// AddDocument embeds text using the EmbedEngine and stores it in the
// KnowledgeStore.
func (a *Agent) AddDocument(text string) error {
	vec, err := a.embed.Embed(text, llama.EmbedOptions{})
	if err != nil {
		return fmt.Errorf("agent: embed document: %w", err)
	}
	return a.store.Add(text, vec)
}

// AddTool registers a callable tool. The handler receives decoded JSON
// arguments and must return a JSON-serialisable result.
func (a *Agent) AddTool(def llama.ToolDefinition, handler tools.Handler) error {
	return a.reg.Register(def, handler)
}

// Chat sends a user message for the given session and returns the assistant's
// final text reply.
//
// The agent:
//  1. Embeds the user message and retrieves relevant knowledge context.
//  2. Injects context into the system prompt.
//  3. Appends the user message to the session history.
//  4. Loops up to maxToolIterations times: if the model returns a tool call,
//     executes it and feeds the result back; otherwise returns the text reply.
func (a *Agent) Chat(sessionID, message string) (string, error) {
	// 1. Retrieve knowledge context.
	context, err := a.retrieveContext(message)
	if err != nil {
		return "", fmt.Errorf("agent: retrieve context: %w", err)
	}

	// 2. Build system prompt.
	systemPrompt := buildSystemPrompt(context)

	// 3. Update session history with user message.
	a.mu.Lock()
	history := a.sessions[sessionID]
	history = append(history, llama.Message{Role: "user", Content: message})
	a.sessions[sessionID] = history
	a.mu.Unlock()

	// 4. Agentic loop.
	msgs := buildMessages(systemPrompt, history)
	toolDefs := a.reg.Definitions()

	for i := 0; i < maxToolIterations; i++ {
		req := llama.ChatRequest{
			Messages:     msgs,
			ResponseMode: pickResponseMode(toolDefs),
			Tools:        toolDefs,
			ToolChoice:   "auto",
		}

		resp, err := a.chat.Chat(req, llama.ChatOptions{})
		if err != nil {
			return "", fmt.Errorf("agent: chat inference: %w", err)
		}

		switch resp.Type {
		case "assistant_text":
			text := resp.Text
			// Save assistant reply to history.
			a.mu.Lock()
			a.sessions[sessionID] = append(a.sessions[sessionID],
				llama.Message{Role: "assistant", Content: text})
			a.mu.Unlock()
			return text, nil

		case "tool_call":
			if len(resp.ToolCalls) == 0 {
				return "", fmt.Errorf("agent: model returned tool_call with no tool calls")
			}
			// Execute each tool call and collect results.
			assistantMsg := llama.Message{Role: "assistant", Content: toolCallsToContent(resp.ToolCalls)}
			msgs = append(msgs, assistantMsg)

			for _, tc := range resp.ToolCalls {
				args, _ := toStringMap(tc.Arguments)
				result, execErr := a.reg.Execute(tc.Name, args)
				var resultStr string
				if execErr != nil {
					resultStr = fmt.Sprintf(`{"error": %q}`, execErr.Error())
				} else {
					b, _ := json.Marshal(result)
					resultStr = string(b)
				}
				msgs = append(msgs, llama.Message{
					Role:     "tool",
					Content:  resultStr,
					ToolName: tc.Name,
				})
			}
			// Continue the loop with the tool results in context.

		case "structured_json":
			// Treat as text reply.
			b, _ := json.Marshal(resp.JSON)
			text := string(b)
			a.mu.Lock()
			a.sessions[sessionID] = append(a.sessions[sessionID],
				llama.Message{Role: "assistant", Content: text})
			a.mu.Unlock()
			return text, nil

		default:
			return "", fmt.Errorf("agent: unexpected response type %q", resp.Type)
		}
	}

	return "", fmt.Errorf("agent: exceeded %d tool call iterations", maxToolIterations)
}

// ClearSession removes the conversation history for the given session ID.
func (a *Agent) ClearSession(sessionID string) {
	a.mu.Lock()
	delete(a.sessions, sessionID)
	a.mu.Unlock()
}

// Close releases all resources owned by the Agent (engines and knowledge store).
func (a *Agent) Close() {
	a.store.Close()
	a.embed.Close()
	a.chat.Close()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

func (a *Agent) retrieveContext(text string) ([]knowledge.Document, error) {
	vec, err := a.embed.Embed(text, llama.EmbedOptions{})
	if err != nil {
		// Embedding failure is non-fatal; return empty context.
		return nil, nil //nolint:nilerr
	}
	docs, err := a.store.Search(vec, text, 5)
	if err != nil {
		return nil, err
	}
	return docs, nil
}

func buildSystemPrompt(docs []knowledge.Document) string {
	if len(docs) == 0 {
		return "You are a helpful AI assistant."
	}
	var sb strings.Builder
	sb.WriteString("You are a helpful AI assistant.\n\n")
	sb.WriteString("Relevant context:\n")
	for _, d := range docs {
		sb.WriteString("- ")
		sb.WriteString(d.Text)
		sb.WriteByte('\n')
	}
	return sb.String()
}

func buildMessages(systemPrompt string, history []llama.Message) []llama.Message {
	msgs := make([]llama.Message, 0, len(history)+1)
	msgs = append(msgs, llama.Message{Role: "system", Content: systemPrompt})
	msgs = append(msgs, history...)
	return msgs
}

func pickResponseMode(defs []llama.ToolDefinition) llama.ResponseMode {
	if len(defs) > 0 {
		return llama.ResponseModeToolCall
	}
	return llama.ResponseModeText
}

func toolCallsToContent(calls []llama.ToolCall) string {
	b, _ := json.Marshal(calls)
	return string(b)
}

func toStringMap(v interface{}) (map[string]interface{}, bool) {
	if v == nil {
		return make(map[string]interface{}), true
	}
	// Already a map.
	if m, ok := v.(map[string]interface{}); ok {
		return m, true
	}
	// JSON round-trip.
	b, err := json.Marshal(v)
	if err != nil {
		return make(map[string]interface{}), false
	}
	var m map[string]interface{}
	if err := json.Unmarshal(b, &m); err != nil {
		return make(map[string]interface{}), false
	}
	return m, true
}
