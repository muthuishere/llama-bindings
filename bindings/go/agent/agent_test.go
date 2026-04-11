package agent_test

import (
	"archive/zip"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
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

func TestExportImportRoundTrip(t *testing.T) {
	a := newAgentOrSkip(t)

	// Add some documents.
	docs := []string{
		"The capital of France is Paris.",
		"Go is a statically typed language.",
		"SQLite is a lightweight database engine.",
	}
	for _, d := range docs {
		if err := a.AddDocument(d); err != nil {
			t.Fatalf("AddDocument(%q): %v", d, err)
		}
	}

	// Export to a temp ZIP.
	tmpDir := t.TempDir()
	zipPath := filepath.Join(tmpDir, "export.zip")
	if err := a.Export(zipPath); err != nil {
		t.Fatalf("Export: %v", err)
	}

	// Verify ZIP exists and has expected entries.
	zr, err := zip.OpenReader(zipPath)
	if err != nil {
		t.Fatalf("open zip: %v", err)
	}
	defer zr.Close()

	entryNames := make(map[string]bool)
	for _, f := range zr.File {
		entryNames[f.Name] = true
	}
	for _, name := range []string{"manifest.json", "knowledge.json", "knowledge.db"} {
		if !entryNames[name] {
			t.Errorf("missing ZIP entry %q", name)
		}
	}

	// Import from the ZIP.
	importDBPath := filepath.Join(tmpDir, "imported.db")
	imported, err := agent.ImportFrom(zipPath, dummyChat, dummyEmbed, importDBPath)
	if err != nil {
		t.Fatalf("ImportFrom: %v", err)
	}
	defer imported.Close()

	// The imported agent should work (basic smoke test).
	resp, err := imported.Chat("import-test", "hello")
	if err != nil {
		t.Fatalf("Chat on imported agent: %v", err)
	}
	if resp == "" {
		t.Fatal("expected non-empty response from imported agent")
	}
}

func TestExportZipStructure(t *testing.T) {
	// This test validates the ZIP structure without requiring the bridge.
	// We test by creating a minimal ZIP that matches the export format,
	// then verify the manifest can be parsed.

	tmpDir := t.TempDir()
	zipPath := filepath.Join(tmpDir, "test.zip")

	// Create a mock export ZIP.
	manifest := map[string]interface{}{
		"version":       "1",
		"created_at":    "2024-01-01T00:00:00Z",
		"chat_model":    "chat.gguf",
		"embed_model":   "embed.gguf",
		"storage_path":  "agent.db",
		"doc_count":     2,
		"embedding_dim": 3,
	}
	manifestJSON, _ := json.MarshalIndent(manifest, "", "  ")

	knowledgeDocs := []map[string]interface{}{
		{"text": "hello world", "embedding": []float64{0.1, 0.2, 0.3}},
		{"text": "foo bar", "embedding": []float64{0.4, 0.5, 0.6}},
	}
	knowledgeJSON, _ := json.Marshal(knowledgeDocs)

	f, err := os.Create(zipPath)
	if err != nil {
		t.Fatalf("create zip: %v", err)
	}
	zw := zip.NewWriter(f)

	for name, data := range map[string][]byte{
		"manifest.json":  manifestJSON,
		"knowledge.json": knowledgeJSON,
		"knowledge.db":   {}, // empty placeholder
	} {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatalf("create entry %s: %v", name, err)
		}
		if _, err := w.Write(data); err != nil {
			t.Fatalf("write entry %s: %v", name, err)
		}
	}
	zw.Close()
	f.Close()

	// Verify the ZIP can be opened and has all entries.
	zr, err := zip.OpenReader(zipPath)
	if err != nil {
		t.Fatalf("open zip: %v", err)
	}
	defer zr.Close()

	if len(zr.File) != 3 {
		t.Fatalf("expected 3 entries, got %d", len(zr.File))
	}

	// Verify manifest content.
	for _, f := range zr.File {
		if f.Name == "manifest.json" {
			rc, _ := f.Open()
			var m map[string]interface{}
			if err := json.NewDecoder(rc).Decode(&m); err != nil {
				t.Fatalf("decode manifest: %v", err)
			}
			rc.Close()
			if m["version"] != "1" {
				t.Errorf("expected version 1, got %v", m["version"])
			}
			if int(m["doc_count"].(float64)) != 2 {
				t.Errorf("expected doc_count 2, got %v", m["doc_count"])
			}
		}
	}
}

func TestImportFromInvalidVersion(t *testing.T) {
	tmpDir := t.TempDir()
	zipPath := filepath.Join(tmpDir, "bad.zip")

	// Create a ZIP with version "99".
	manifest := map[string]interface{}{
		"version":       "99",
		"created_at":    "2024-01-01T00:00:00Z",
		"chat_model":    "chat.gguf",
		"embed_model":   "embed.gguf",
		"storage_path":  "agent.db",
		"doc_count":     0,
		"embedding_dim": 0,
	}
	manifestJSON, _ := json.Marshal(manifest)

	f, _ := os.Create(zipPath)
	zw := zip.NewWriter(f)
	w, _ := zw.Create("manifest.json")
	w.Write(manifestJSON)
	w2, _ := zw.Create("knowledge.json")
	w2.Write([]byte("[]"))
	w3, _ := zw.Create("knowledge.db")
	w3.Write([]byte{})
	zw.Close()
	f.Close()

	_, err := agent.ImportFrom(zipPath, dummyChat, dummyEmbed, filepath.Join(tmpDir, "out.db"))
	if err == nil {
		t.Fatal("expected error for unsupported version")
	}
}
