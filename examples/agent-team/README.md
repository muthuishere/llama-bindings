# Agent Team Example

A **Knowledge Assistant** that shows how to use the llama-bindings agent layer
across all three target languages: Go, Java, and Browser JS.

Each example creates the same agent:
- **Knowledge base** — 8 documents about the llama-bindings project loaded via `addDocument`
- **Tool** — a `calculate` function (+, -, ×, ÷, √) registered via `addTool`
- **Multi-turn conversation** — scripted demo questions that exercise knowledge retrieval and tool dispatch

---

## Prerequisites

```sh
# 1. Build the native C bridge
task build-bridge

# 2. Download models (Gemma 4 E2B + nomic-embed-text-v1.5)
task download-model
```

---

## Go

```sh
task example-go
# or directly:
cd examples/agent-team/go
LLAMA_CHAT_MODEL=~/llama-bindings-conf/models/chat/gemma-4-E2B-it-Q4_K_M.gguf \
LLAMA_EMBED_MODEL=~/llama-bindings-conf/models/embeddings/nomic-embed-text-v1.5.Q4_K_M.gguf \
go run main.go
```

References the library via `go.mod` `replace` directive — no publish needed.

---

## Java

```sh
# Install llama-java to local Maven repo first (one-time):
cd bindings/java
mvn install -DskipTests -Dbridge.lib.dir=../../bridge/build

# Then run the example:
task example-java
# or directly:
cd examples/agent-team/java
mvn package -Dbridge.lib.dir=../../../bridge/build
java --enable-preview \
     -Djava.library.path=../../../bridge/build \
     -jar target/agent-team-java-1.0-SNAPSHOT-jar-with-dependencies.jar
```

---

## Browser JS

```sh
task example-js
# Then open: http://localhost:3000/examples/agent-team/js-browser/
```

Serves the repo root so `/bindings/js-browser/src/` imports resolve.
Without the WASM build (`task build-wasm`) the stub module runs —
full API surface is exercised with deterministic responses.

With the real WASM build, set model URLs in the browser via:
```js
window.LLAMA_CHAT_MODEL  = 'http://localhost:3000/path/to/gemma-4-E2B-it-Q4_K_M.gguf';
window.LLAMA_EMBED_MODEL = 'http://localhost:3000/path/to/nomic-embed-text-v1.5.Q4_K_M.gguf';
```

---

## What the conversation covers

| Turn | Tests |
|---|---|
| "What is llama-bindings?" | Knowledge retrieval |
| "How does the Go binding avoid CGO?" | Specific doc retrieval |
| "What is the square root of 144?" | Tool dispatch (calculate) |
| "What chat model is used?" | Knowledge + factual answer |
| "How do I build everything?" | Knowledge retrieval |
