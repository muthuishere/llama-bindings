# Design: Tiny Cross-Language llama.cpp Bridge

## 1. Goal

Build a small, stable wrapper layer on top of llama.cpp for one simple use case:

- load a GGUF model
- send a prompt
- get back a completion string

The wrapper supports Go, Java, and JavaScript, but the core logic exists only once.

---

## 2. Architecture

```
llama.cpp upstream
      ↓
our C bridge  (stable API we own)
      ↓
Go binding | Java binding | JS binding
```

The bridge is the blast-radius boundary. Only `bridge/src/llama_bridge.c` knows
the llama.cpp internals. Every other layer is insulated from upstream churn.

---

## 3. Public C API

```c
typedef void* llama_engine_t;

llama_engine_t llama_engine_create(const char* model_path);
char*          llama_engine_complete(llama_engine_t engine, const char* prompt);
void           llama_engine_free_string(char* str);
void           llama_engine_destroy(llama_engine_t engine);
```

This is the only contract that language bindings depend on.

---

## 4. Design Principles

| Principle            | What it means                                                     |
|----------------------|-------------------------------------------------------------------|
| One core             | No inference logic in Go, Java, or JS                            |
| Stable contract      | Bindings never call raw llama.cpp APIs                           |
| Upgrade isolation    | Only `llama_bridge.c` absorbs upstream API changes               |
| Opaque handles       | Bindings receive `void*`; they never see model/context internals |
| Minimal surface      | Three functions. That's all.                                     |

---

## 5. Folder Structure

```
project-root/
  Taskfile.yml              # build/update automation
  third_party/llama.cpp/    # vendored upstream (Taskfile-controlled)
  bridge/
    include/llama_bridge.h  # public stable header
    src/
      llama_bridge.c        # only file that knows llama.cpp internals
      llama_bridge_internal.h
  bindings/
    go/llama/               # CGO wrapper
    java/                   # JNA wrapper
    js/                     # Node.js ffi-napi wrapper
  tests/prompts/            # shared test fixtures
  docs/                     # this document and others
  scripts/                  # shell utilities
```

---

## 6. Internal Engine Structure

```c
struct llama_engine {
    struct llama_model*   model;
    struct llama_context* ctx;
};
```

This struct is **never** exposed to language bindings.

---

## 7. Runtime Flow

1. Caller calls `create(modelPath)` → bridge initialises backend, loads model, creates context
2. Caller calls `complete(prompt)` → bridge tokenises, decodes, samples, returns string
3. Caller calls `destroy()` → bridge frees context, model, engine struct

---

## 8. Non-Goals (v1)

- Streaming tokens
- Embeddings
- Chat history / session memory
- Tool calling
- Structured JSON mode
- Batching / concurrency
- Server or API gateway

---

## 9. Language Binding Shapes

### Go

```go
engine, err := llama.Load("model.gguf")
defer engine.Close()
out, err := engine.Complete("Say hello.")
```

### Java

```java
try (LlamaEngine engine = LlamaEngine.load("model.gguf")) {
    String out = engine.complete("Say hello.");
}
```

### JavaScript

```js
const engine = Llama.load('model.gguf');
const out    = engine.complete('Say hello.');
engine.close();
```

---

## 10. Upgrade Path

When llama.cpp releases a breaking API change:

1. Run `task update` to pull the latest llama.cpp
2. Run `task build-llama` to rebuild upstream
3. Edit **only** `bridge/src/llama_bridge.c` to adapt to the new API
4. Run `task build-bridge`
5. Run `task test` to verify all bindings still pass

See `docs/upgrade.md` for the step-by-step procedure.
