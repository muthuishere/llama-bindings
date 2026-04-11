# E2E Test Specification

Unified end-to-end tests for all three example servers (Go, Java, JS browser).
Run via `task e2e-examples`. Outputs a JSON report with pass/fail and timing.

## Test Matrix

| Test | Go (:8080) | Java (:8081) | JS Browser (Playwright) |
|------|------------|--------------|-------------------------|
| `chat` | POST /chat | POST /chat | Agent.chat() in browser |
| `chat_tool` | POST /chat (calculator) | POST /chat (calculator) | Agent.chat() with tool |
| `embed` | POST /embed | POST /embed | addDocument (uses embed internally) |
| `chat_schema` | POST /chat-schema | POST /chat-schema | Structured prompt via chat |

## API Tests (Go + Java)

The example servers expose `POST /chat` with NDJSON streaming.
Tests send requests and collect the streamed response.

### Test 1: `chat` — Knowledge retrieval
```json
{"session": "e2e-chat", "message": "What is llama-bindings?"}
```
**Pass criteria:** Response contains "cross-language" or "library" or "llama.cpp".

### Test 2: `chat_tool` — Calculator tool call
```json
{"session": "e2e-tool", "message": "What is the square root of 144?"}
```
**Pass criteria:** Response contains "12".

### Test 3: `embed` — Embedding vector
```json
{"text": "The capital of France is Paris."}
```
**Pass criteria:** Response contains "embedding" and "dim" fields, and dim > 0.

### Test 4: `chat_schema` — Structured JSON output
```json
{"session": "e2e-schema", "message": "Extract: John is 30 years old", "schema": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}}
```
**Pass criteria:** Response contains "John" or "john" or "30" or valid JSON with name/age fields.

## JS Browser Tests (Playwright)

Run against `http://localhost:3000/examples/agent-team/js-browser/`.
The agent uses stub mode (no WASM inference).

### Test 1: `chat` — Agent chat returns response
Click "What is llama-bindings?" chip, verify agent response appears.

### Test 2: `chat_tool` — Calculator suggestion
Click "What is the square root of 144?" chip, verify response appears.

### Test 3: `page_load` — Agent initializes
Verify "Agent ready" status text appears within 10 seconds.

### Test 4: `embed` — addDocument uses embed engine
Verify status text contains "8 documents" (addDocument calls embed internally during init).

### Test 5: `chat_schema` — Structured prompt
Send "List the name and age: John is 30" and verify a non-empty agent response.

## JSON Report Format

```json
{
  "timestamp": "2026-04-11T03:00:00Z",
  "model": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
  "results": [
    {
      "target": "go",
      "test": "chat",
      "status": "pass",
      "duration_ms": 11372,
      "response": "llama-bindings is a cross-language library..."
    },
    {
      "target": "go",
      "test": "chat_tool",
      "status": "pass",
      "duration_ms": 8893,
      "response": "The square root of 144 is 12."
    },
    {
      "target": "go",
      "test": "embed",
      "status": "pass",
      "duration_ms": 523,
      "response": "dim=768"
    },
    {
      "target": "go",
      "test": "chat_schema",
      "status": "pass",
      "duration_ms": 9200,
      "response": "{\"name\":\"John\",\"age\":30}"
    },
    {
      "target": "java",
      "test": "chat",
      "status": "pass",
      "duration_ms": 67630,
      "response": "llama-bindings is a cross-language library..."
    },
    {
      "target": "java",
      "test": "chat_tool",
      "status": "pass",
      "duration_ms": 12400,
      "response": "The square root of 144 is 12."
    },
    {
      "target": "java",
      "test": "embed",
      "status": "pass",
      "duration_ms": 610,
      "response": "dim=768"
    },
    {
      "target": "java",
      "test": "chat_schema",
      "status": "pass",
      "duration_ms": 15300,
      "response": "{\"name\":\"John\",\"age\":30}"
    },
    {
      "target": "js-browser",
      "test": "playwright",
      "status": "pass",
      "duration_ms": 5200,
      "response": "Playwright tests passed"
    }
  ],
  "summary": {
    "total": 9,
    "passed": 9,
    "failed": 0
  }
}
```

## Runner

`tests/e2e/run-e2e-examples.sh` — starts servers, runs tests, outputs JSON.
Servers are started in background and killed on exit.
