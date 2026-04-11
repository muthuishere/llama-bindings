# E2E Test Specification

Unified end-to-end tests for all three example servers (Go, Java, JS browser).
Run via `task e2e-examples`. Outputs a JSON report with pass/fail and timing.

## Test Matrix

| Test | Go (:8080) | Java (:8081) | JS Browser (Playwright) |
|------|------------|--------------|-------------------------|
| `chat` | POST /chat | POST /chat | Agent.chat() in browser |
| `chat_tool` | POST /chat (calculator) | POST /chat (calculator) | Agent.chat() with tool |
| `embed` | POST /embed (if exposed) | POST /embed | LlamaEmbed in browser |

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
Not exposed by example servers. Tested via binding unit tests.
For the report, include the unit test result for embed.

## JS Browser Tests (Playwright)

Run against `http://localhost:3000/examples/agent-team/js-browser/`.
The agent uses stub mode (no WASM inference).

### Test 1: `chat` — Agent chat returns response
Click "What is llama-bindings?" chip, verify agent response appears.

### Test 2: `chat_tool` — Calculator suggestion
Click "What is the square root of 144?" chip, verify response appears.

### Test 3: `page_load` — Agent initializes
Verify "Agent ready" status text appears within 10 seconds.

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
      "target": "java",
      "test": "chat",
      "status": "pass",
      "duration_ms": 67630,
      "response": "llama-bindings is a cross-language library..."
    },
    {
      "target": "js-browser",
      "test": "page_load",
      "status": "pass",
      "duration_ms": 1200,
      "response": "Agent ready"
    }
  ],
  "summary": {
    "total": 7,
    "passed": 7,
    "failed": 0
  }
}
```

## Runner

`tests/e2e/run-e2e-examples.sh` — starts servers, runs tests, outputs JSON.
Servers are started in background and killed on exit.
