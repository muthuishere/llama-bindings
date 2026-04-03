# Java Binding

Java library for [llama.cpp](https://github.com/ggml-org/llama.cpp) chat and embedding inference.

**Runtime:** Java → Project Panama FFM (JDK 21) → native bridge → llama.cpp

## Maven coordinates

```xml
<dependency>
  <groupId>com.example.llama</groupId>
  <artifactId>llama-java</artifactId>
  <version>0.1.0</version>
</dependency>
```

## Prerequisites

- JDK 21+
- Maven ≥ 3.9
- Native bridge built (`task build-bridge`)
- JVM started with `--enable-native-access=ALL-UNNAMED` and `-Djava.library.path=<bridge/build>`

## Quick start

### Chat

```java
import com.example.llama.*;
import com.example.llama.model.*;
import java.util.List;

try (var chat = ChatEngine.load("chat-model.gguf", new LoadOptions(event -> {
    System.out.println("Event: " + event.event);
}))) {
    var request = ChatRequest.builder()
        .messages(List.of(
            ChatMessage.system("You are helpful."),
            ChatMessage.user("Extract person data.")
        ))
        .responseMode("json_schema")
        .schema(Map.of("name", "person_extract", "schema", Map.of("type", "object")))
        .generation(new GenerationOptions(0.0f, 128, 1.0f, 40))
        .build();

    ChatResponse resp = chat.chat(request, new ChatOptions());
    System.out.println(resp.type);  // "structured_json"
}
```

### Embed

```java
try (var embed = EmbedEngine.load("embed-model.gguf", new LoadOptions())) {
    float[] vec = embed.embed("hello world", new EmbedOptions());
    System.out.println("Vector length: " + vec.length);
}
```

## API reference

| Class | Key method |
|-------|------------|
| `ChatEngine` | `load(path, LoadOptions)` → `ChatEngine` |
| `ChatEngine` | `chat(ChatRequest, ChatOptions)` → `ChatResponse` |
| `EmbedEngine` | `load(path, LoadOptions)` → `EmbedEngine` |
| `EmbedEngine` | `embed(text, EmbedOptions)` → `float[]` |
| `LlamaException` | `getCode()` — bridge error code |

## Tests

```sh
task test-java
# or
cd bindings/java && mvn test -Dbridge.lib.dir=../../bridge/build
```

## Build

```sh
task build-java
```

Upstream llama.cpp: https://github.com/ggml-org/llama.cpp
