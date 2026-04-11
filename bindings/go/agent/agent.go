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
	"archive/zip"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

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

	chatModelPath  string
	embedModelPath string
	storagePath    string

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
		chat:           chat,
		embed:          embed,
		store:          store,
		reg:            tools.New(),
		chatModelPath:  chatModelPath,
		embedModelPath: embedModelPath,
		storagePath:    storagePath,
		sessions:       make(map[string][]llama.Message),
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
	justCalledTool := false // after tool execution, switch to text mode for final answer

	for i := 0; i < maxToolIterations; i++ {
		// After a tool call + result, ask the model for a text answer.
		useTools := toolDefs
		mode := pickResponseMode(toolDefs)
		if justCalledTool {
			useTools = nil
			mode = llama.ResponseModeText
			justCalledTool = false
		}

		req := llama.ChatRequest{
			Messages:     msgs,
			ResponseMode: mode,
			Tools:        useTools,
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
			// Continue the loop — next iteration uses text mode.
			justCalledTool = true

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
// Export / Import
// ─────────────────────────────────────────────────────────────────────────────

// exportManifest is serialised as manifest.json inside the export ZIP.
type exportManifest struct {
	Version      string `json:"version"`
	CreatedAt    string `json:"created_at"`
	ChatModel    string `json:"chat_model"`
	EmbedModel   string `json:"embed_model"`
	StoragePath  string `json:"storage_path"`
	DocCount     int    `json:"doc_count"`
	EmbeddingDim int    `json:"embedding_dim"`
}

// exportDoc is one entry in the knowledge.json array.
type exportDoc struct {
	Text      string    `json:"text"`
	Embedding []float32 `json:"embedding"`
}

// Export writes a complete Agent bundle to a ZIP archive at zipPath.
func (a *Agent) Export(zipPath string) error {
	docs, err := a.store.All()
	if err != nil {
		return fmt.Errorf("agent: export: %w", err)
	}

	// Build manifest.
	embDim := 0
	if len(docs) > 0 {
		embDim = len(docs[0].Embedding)
	}
	manifest := exportManifest{
		Version:      "1",
		CreatedAt:    time.Now().UTC().Format(time.RFC3339),
		ChatModel:    filepath.Base(a.chatModelPath),
		EmbedModel:   filepath.Base(a.embedModelPath),
		StoragePath:  a.storagePath,
		DocCount:     len(docs),
		EmbeddingDim: embDim,
	}

	manifestJSON, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		return fmt.Errorf("agent: export: marshal manifest: %w", err)
	}

	// Build knowledge.json.
	knowledgeDocs := make([]exportDoc, len(docs))
	for i, d := range docs {
		knowledgeDocs[i] = exportDoc{Text: d.Text, Embedding: d.Embedding}
	}
	knowledgeJSON, err := json.Marshal(knowledgeDocs)
	if err != nil {
		return fmt.Errorf("agent: export: marshal knowledge: %w", err)
	}

	// Create the ZIP file.
	f, err := os.Create(zipPath)
	if err != nil {
		return fmt.Errorf("agent: export: create zip: %w", err)
	}
	defer f.Close()

	zw := zip.NewWriter(f)
	defer zw.Close()

	// Write manifest.json.
	if err := writeZipEntry(zw, "manifest.json", manifestJSON); err != nil {
		return err
	}

	// Write knowledge.json.
	if err := writeZipEntry(zw, "knowledge.json", knowledgeJSON); err != nil {
		return err
	}

	// Write knowledge.db — copy the file if it's a real path.
	dbPath := a.store.Path()
	if dbPath != "" && dbPath != ":memory:" {
		dbBytes, err := os.ReadFile(dbPath)
		if err != nil {
			return fmt.Errorf("agent: export: read db: %w", err)
		}
		if err := writeZipEntry(zw, "knowledge.db", dbBytes); err != nil {
			return err
		}
	} else {
		// In-memory store: write empty placeholder.
		if err := writeZipEntry(zw, "knowledge.db", []byte{}); err != nil {
			return err
		}
	}

	return nil
}

// ImportFrom restores an Agent from a previously exported ZIP bundle.
func ImportFrom(zipPath, chatModelPath, embedModelPath, storagePath string) (*Agent, error) {
	zr, err := zip.OpenReader(zipPath)
	if err != nil {
		return nil, fmt.Errorf("agent: import: open zip: %w", err)
	}
	defer zr.Close()

	// Read manifest.json.
	manifestData, err := readZipEntry(&zr.Reader, "manifest.json")
	if err != nil {
		return nil, fmt.Errorf("agent: import: %w", err)
	}
	var manifest exportManifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		return nil, fmt.Errorf("agent: import: parse manifest: %w", err)
	}
	if manifest.Version != "1" {
		return nil, fmt.Errorf("agent: import: unsupported version %q", manifest.Version)
	}

	// Default storage path.
	if storagePath == "" {
		storagePath = "agent.db"
	}

	// Extract knowledge.db if present and non-empty.
	dbData, err := readZipEntry(&zr.Reader, "knowledge.db")
	if err != nil {
		return nil, fmt.Errorf("agent: import: %w", err)
	}
	if len(dbData) > 0 {
		if err := os.WriteFile(storagePath, dbData, 0644); err != nil {
			return nil, fmt.Errorf("agent: import: write db: %w", err)
		}
	}

	// If no DB data was exported (in-memory store), we need to rebuild from
	// knowledge.json.
	if len(dbData) == 0 {
		knowledgeData, err := readZipEntry(&zr.Reader, "knowledge.json")
		if err != nil {
			return nil, fmt.Errorf("agent: import: %w", err)
		}
		var docs []exportDoc
		if err := json.Unmarshal(knowledgeData, &docs); err != nil {
			return nil, fmt.Errorf("agent: import: parse knowledge: %w", err)
		}
		// Create a fresh store and insert all docs.
		store, err := knowledge.New(storagePath)
		if err != nil {
			return nil, fmt.Errorf("agent: import: create store: %w", err)
		}
		for _, d := range docs {
			if err := store.Add(d.Text, d.Embedding); err != nil {
				store.Close()
				return nil, fmt.Errorf("agent: import: add doc: %w", err)
			}
		}
		store.Close()
	}

	// Now create a normal Agent pointing at the restored DB.
	return New(chatModelPath, embedModelPath, storagePath)
}

func writeZipEntry(zw *zip.Writer, name string, data []byte) error {
	w, err := zw.Create(name)
	if err != nil {
		return fmt.Errorf("agent: export: create %s: %w", name, err)
	}
	_, err = w.Write(data)
	if err != nil {
		return fmt.Errorf("agent: export: write %s: %w", name, err)
	}
	return nil
}

func readZipEntry(zr *zip.Reader, name string) ([]byte, error) {
	for _, f := range zr.File {
		if f.Name == name {
			rc, err := f.Open()
			if err != nil {
				return nil, fmt.Errorf("open %s: %w", name, err)
			}
			defer rc.Close()
			return io.ReadAll(rc)
		}
	}
	return nil, fmt.Errorf("missing %s in archive", name)
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
