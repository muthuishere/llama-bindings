# Observability

## Overview

Observability is first-class. The bridge emits structured JSON events for every
significant lifecycle moment. Language bindings surface these as typed callbacks.

## Event callback

Every engine accepts an optional callback at creation time:

```c
// C (bridge ABI)
typedef void (*llama_event_cb)(const char* event_json, void* user_data);
```

```go
// Go
type EventCallback func(event Event)
```

```java
// Java
public interface EventListener { void onEvent(Event event); }
```

```js
// Browser JS
{ onEvent(event) { console.log(event); } }
```

## Event payload shape

```json
{
  "event":       "chat_infer_start",
  "engine_type": "chat",
  "stage":       "generation",
  "progress":    42,
  "message":     "Generating response",
  "usage": {
    "prompt_tokens": 120,
    "completion_tokens": 18,
    "total_tokens": 138
  },
  "partial_text":  "Hello, here is",
  "timestamp_ms":  1712345678901
}
```

Not every field is present in every event.

## Canonical event names

### Load events
| Event | Meaning |
|-------|---------|
| `chat_engine_create_start` | Model load begun |
| `chat_engine_create_success` | Model loaded |
| `chat_engine_create_failure` | Model load failed |
| `embed_engine_create_start` | Embed model load begun |
| `embed_engine_create_success` | Embed model loaded |
| `embed_engine_create_failure` | Embed model load failed |

### Chat events
| Event | Meaning |
|-------|---------|
| `chat_infer_start` | Inference started |
| `chat_progress` | Progress update |
| `chat_partial_text` | Partial token output |
| `chat_tool_candidate` | Tool call candidate identified |
| `chat_schema_validation` | Schema validation step |
| `chat_complete` | Inference complete |
| `chat_infer_success` | Successful inference |
| `chat_infer_failure` | Inference failed |

### Embed events
| Event | Meaning |
|-------|---------|
| `embed_infer_start` | Embedding started |
| `embed_progress` | Progress update |
| `embed_complete` | Embedding complete |
| `embed_infer_success` | Successful embedding |
| `embed_infer_failure` | Embedding failed |

### Lifecycle events
| Event | Meaning |
|-------|---------|
| `engine_destroy` | Engine closed |
| `engine_error` | Unexpected engine error |

## Metrics summary

| Metric | Source |
|--------|--------|
| Model load duration ms | `create_start` → `create_success` timestamps |
| Request duration ms | `infer_start` → `infer_success` timestamps |
| Prompt / completion / total tokens | `usage` field on completion events |
| Finish reason | `finish_reason` in response |
| Active engine count | tracked by the application via create/destroy events |
