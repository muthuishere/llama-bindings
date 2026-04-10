// examples/agent-team/go/main.go
//
// Knowledge Assistant HTTP server — Go example for llama-bindings.
//
// Exposes:
//   POST /chat  — NDJSON streaming: each line is {"token":"..."} then {"done":true}
//   GET  /      — Chat UI (index.html from the same directory)
//
// Prerequisites:
//   task build-bridge      (builds the native C bridge)
//   task download-model    (fetches Gemma 4 E2B + nomic-embed)
//
// Run:
//   cd examples/agent-team/go
//   go mod tidy            (first time — downloads gin)
//   go run main.go
//   open http://localhost:8080
//
// Override model paths:
//   LLAMA_CHAT_MODEL=/path/chat.gguf LLAMA_EMBED_MODEL=/path/embed.gguf go run main.go

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/muthuishere/llama-bindings/go/agent"
	"github.com/muthuishere/llama-bindings/go/llama"
)

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

var (
	globalAgent *agent.Agent
	agentMu     sync.Mutex
)

// ---------------------------------------------------------------------------
// Model paths
// ---------------------------------------------------------------------------

func chatModelPath() string {
	if p := os.Getenv("LLAMA_CHAT_MODEL"); p != "" {
		return p
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "llama-bindings-conf", "models", "chat", "gemma-4-E2B-it-Q4_K_M.gguf")
}

func embedModelPath() string {
	if p := os.Getenv("LLAMA_EMBED_MODEL"); p != "" {
		return p
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "llama-bindings-conf", "models", "embeddings", "nomic-embed-text-v1.5.Q4_K_M.gguf")
}

// ---------------------------------------------------------------------------
// Agent setup
// ---------------------------------------------------------------------------

func setupAgent() (*agent.Agent, error) {
	a, err := agent.New(chatModelPath(), embedModelPath(), ":memory:")
	if err != nil {
		return nil, fmt.Errorf("create agent: %w", err)
	}

	docs := []string{
		"llama-bindings is a cross-language library that wraps llama.cpp with a unified API for Go, Java, and Browser JavaScript.",
		"The Go binding uses purego for dynamic loading — no CGO required. Import path: github.com/muthuishere/llama-bindings/go/llama.",
		"The Java binding uses Project Panama FFM (JDK 22) instead of JNI. Artifact: com.example.llama:llama-java.",
		"The Browser JS binding compiles llama.cpp to WebAssembly via Emscripten. Package: @llama-bindings/js-browser.",
		"The Agent layer combines ChatEngine, EmbedEngine, KnowledgeStore, and ToolRegistry into a single orchestrated loop.",
		"Build all targets with: task build. Run all tests with: task test. Download models with: task download-model.",
		"The chat model is Gemma 4 E2B (2.3B effective parameters, native tool calling tokens, Apache 2.0 license).",
		"The embed model is nomic-embed-text-v1.5 (137M parameters, 80 MB Q4_K_M, purpose-built for semantic search).",
	}
	for _, doc := range docs {
		if err := a.AddDocument(doc); err != nil {
			a.Close()
			return nil, fmt.Errorf("AddDocument: %w", err)
		}
	}

	err = a.AddTool(llama.ToolDefinition{
		Name:        "calculate",
		Description: "Perform basic arithmetic. Supports add, subtract, multiply, divide, sqrt.",
		Parameters: map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"operation": map[string]interface{}{
					"type":        "string",
					"enum":        []string{"add", "subtract", "multiply", "divide", "sqrt"},
					"description": "The arithmetic operation to perform",
				},
				"a": map[string]interface{}{"type": "number", "description": "First operand"},
				"b": map[string]interface{}{"type": "number", "description": "Second operand (not used for sqrt)"},
			},
			"required":             []string{"operation", "a"},
			"additionalProperties": false,
		},
	}, func(args map[string]interface{}) (interface{}, error) {
		op, _ := args["operation"].(string)
		aVal, _ := toFloat(args["a"])
		bVal, _ := toFloat(args["b"])
		switch op {
		case "add":
			return map[string]interface{}{"result": aVal + bVal}, nil
		case "subtract":
			return map[string]interface{}{"result": aVal - bVal}, nil
		case "multiply":
			return map[string]interface{}{"result": aVal * bVal}, nil
		case "divide":
			if bVal == 0 {
				return nil, fmt.Errorf("division by zero")
			}
			return map[string]interface{}{"result": aVal / bVal}, nil
		case "sqrt":
			return map[string]interface{}{"result": math.Sqrt(aVal)}, nil
		default:
			return nil, fmt.Errorf("unknown operation: %s", op)
		}
	})
	if err != nil {
		a.Close()
		return nil, fmt.Errorf("AddTool: %w", err)
	}

	return a, nil
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

type chatRequest struct {
	Session string `json:"session" binding:"required"`
	Message string `json:"message" binding:"required"`
}

func writeNDJSON(w http.ResponseWriter, v interface{}) {
	line, _ := json.Marshal(v)
	w.Write(append(line, '\n'))
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

func handleChat(c *gin.Context) {
	var req chatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	c.Header("Content-Type", "application/x-ndjson")
	c.Header("Cache-Control", "no-cache")
	c.Header("X-Accel-Buffering", "no")

	w := c.Writer

	agentMu.Lock()
	reply, err := globalAgent.Chat(req.Session, req.Message)
	agentMu.Unlock()

	if err != nil {
		// Bridge stub always returns tool_call → agent hits iteration limit.
		// Surface a friendly message until real inference is wired in.
		if strings.Contains(err.Error(), "exceeded") {
			reply = "(stub mode) The bridge is not yet wired to real llama.cpp inference. " +
				"Once wired, this query would return a real model response."
		} else {
			writeNDJSON(w, map[string]interface{}{"error": err.Error(), "done": true})
			return
		}
	}

	// Stream word by word with a short delay to simulate token streaming.
	words := strings.Fields(reply)
	for i, word := range words {
		token := word
		if i < len(words)-1 {
			token += " "
		}
		writeNDJSON(w, map[string]string{"token": token})
		time.Sleep(25 * time.Millisecond)
	}

	writeNDJSON(w, map[string]interface{}{"done": true, "reply": reply})
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	log.Println("=== llama-bindings · Agent Team Example (Go / Gin) ===")
	log.Println("Loading agent...")

	a, err := setupAgent()
	if err != nil {
		log.Fatalf("Agent setup failed: %v", err)
	}
	globalAgent = a
	defer a.Close()

	log.Println("Agent ready. 8 documents loaded. Calculator tool registered.")

	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()

	r.StaticFile("/", "./index.html")
	r.POST("/chat", handleChat)

	log.Println("Listening on http://localhost:8080")
	if err := r.Run(":8080"); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func toFloat(v interface{}) (float64, error) {
	switch x := v.(type) {
	case float64:
		return x, nil
	case float32:
		return float64(x), nil
	case int:
		return float64(x), nil
	case string:
		return strconv.ParseFloat(x, 64)
	}
	return 0, fmt.Errorf("cannot convert %T to float64", v)
}
