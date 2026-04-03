# Testing

## Running all tests

```sh
task test          # Go + Java + JS
task test-go
task test-java
task test-js
```

## Test categories

### Build tests
Verify that a fresh clone can build all targets. Run after every llama.cpp
upgrade (`task upgrade`).

### Smoke tests
Check that a valid model loads and a minimal request succeeds.  
Tests in this category use real model files stored in `tests/models/`.

### Functional tests — chat
- Text mode: messages → `assistant_text` response
- Schema mode: messages + schema → `structured_json` response
- Tool mode: messages + tools → `tool_call` response

### Functional tests — embed
- Input string → non-empty float vector
- Repeated calls return stable results

### Validation tests
- Malformed JSON request rejected
- Missing `messages` field rejected
- Invalid schema rejected
- Bad model path rejected
- Empty embed input rejected

### Lifecycle tests
- Create → use → destroy cycle
- Repeated create/destroy cycles
- Operations after `close()` return `ENGINE_CLOSED` error

### Cross-language parity tests
Verify that the same conceptual request yields conceptually consistent
results across Go, Java, and browser JS. Located in `tests/integration/`.

### Callback/observability tests
- Load events (`create_start`, `create_success`) emitted
- Chat events (`infer_start`, `infer_success`) emitted
- Embed events (`embed_infer_start`, `embed_complete`) emitted

## Test fixtures

Fixtures live in `tests/fixtures/` and are shared across all language targets.

| Fixture | File |
|---------|------|
| Text mode chat | `tests/fixtures/chat_requests/text_mode.json` |
| Schema mode chat | `tests/fixtures/chat_requests/schema_mode.json` |
| Tool call mode chat | `tests/fixtures/chat_requests/tool_mode.json` |
| Person extract schema | `tests/fixtures/schemas/person_extract.json` |
| Weather tool definition | `tests/fixtures/tool_requests/lookup_weather.json` |
| Embedding prompt | `tests/fixtures/prompts/embed_basic.txt` |

## Model files

Integration tests that require real model files look in `tests/models/`.
These files are not committed to the repository.

```
tests/models/
  chat-model.gguf      ← any compatible GGUF chat model
  embed-model.gguf     ← any compatible GGUF embedding model
```

Tests that cannot find a model file skip gracefully rather than failing.
