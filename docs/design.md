# Architecture and Design

## Overview

```
https://github.com/ggml-org/llama.cpp   ← upstream
             ↑
       llama internals
             ↑
       bridge native layer   (bridge/)
             ↑
  ┌──────────┼──────────┐
  │          │          │
 Go        Java    Browser JS/WASM
```

## Core principle

Only the bridge layer knows about the upstream llama.cpp repository
(`https://github.com/ggml-org/llama.cpp`).  
No language binding may call llama.cpp internals directly.

## Bridge philosophy

The bridge (`bridge/`) is the central compatibility layer. It:

- Isolates upstream llama.cpp changes
- Owns memory and lifecycle for all engine handles
- Normalises request/response behaviour via JSON
- Exposes a tiny, stable C ABI (see `bridge/include/llama_bridge.h`)
- Centralises observability and event emission

The bridge never executes tools, behaves like an HTTP server, or exposes
raw token/logit internals.

## Stable native ABI

```c
/* Chat */
llama_chat_engine_t  llama_chat_create(model_path, on_event, user_data);
char*                llama_chat_infer_json(engine, request_json);
void                 llama_chat_destroy(engine);

/* Embed */
llama_embed_engine_t llama_embed_create(model_path, on_event, user_data);
float*               llama_embed_infer(engine, input_text, *out_len);
void                 llama_embed_destroy(engine);

/* Memory */
void llama_bridge_string_free(char* s);
void llama_bridge_float_free(float* p);
```

ABI changes are rare and deliberate. Upstream llama.cpp changes are absorbed
inside the bridge implementation files, not in the ABI.

## Engine separation

`ChatEngine` and `EmbedEngine` are completely separate. A chat engine never
provides embeddings; an embed engine never provides chat. They may use
different model files.

## Language runtime models

| Target | Stack |
|--------|-------|
| Go | Go → CGO → native bridge → llama.cpp |
| Java | Java → Project Panama FFM (JDK 21) → native bridge → llama.cpp |
| Browser JS | Browser JS → WASM (Emscripten) → llama.cpp |

## Request/response model

All chat inference flows through a JSON request/response boundary at the
bridge ABI. The bridge parses the request, runs inference, and returns a
normalised JSON response. Language bindings serialise/deserialise this JSON;
they never interpret raw model output.
