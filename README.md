# llama-bindings

Cross-language library suite on top of [llama.cpp](https://github.com/ggml-org/llama.cpp).

**Two engines. One bridge philosophy. Three targets.**

| Target | Runtime | Package |
|--------|---------|---------|
| Go | CGO → native bridge → llama.cpp | `github.com/muthuishere/llama-bindings/go` |
| Java | Project Panama FFM → native bridge → llama.cpp | `com.example.llama:llama-java` |
| Browser JS | WASM → llama.cpp | `@llama-bindings/js-browser` |

---

## Quick start

### Go

```go
import "github.com/muthuishere/llama-bindings/go/llama"

chat, err := llama.NewChat("chat-model.gguf", llama.LoadOptions{})
if err != nil { log.Fatal(err) }
defer chat.Close()

resp, err := chat.Chat(llama.ChatRequest{
    Messages: []llama.Message{{Role: "user", Content: "Hello!"}},
}, llama.ChatOptions{Temperature: 0.7})
```

### Java

```java
try (var chat = ChatEngine.load("chat-model.gguf", new LoadOptions())) {
    var resp = chat.chat(request, new ChatOptions());
}
```

### Browser JavaScript

```js
const chat = await LlamaChat.load("chat-model.gguf");
const resp = await chat.chat({ messages: [{ role: "user", content: "Hello!" }] });
chat.close();
```

---

## Repository structure

```
third_party/llama.cpp/    ← upstream: https://github.com/ggml-org/llama.cpp
bridge/                   ← native C bridge (shared by all targets)
bindings/
  go/                     ← Go library
  java/                   ← Java library (JDK 21 / Panama FFM)
  js-browser/             ← Browser JS / WASM library
wasm/                     ← WASM build artefacts and glue
tests/                    ← shared fixtures and integration tests
docs/                     ← design, build, observability, publishing docs
Taskfile.yml              ← build / test / publish orchestration
```

---

## Build

```sh
# 1. Clone upstream llama.cpp (https://github.com/ggml-org/llama.cpp)
task init

# 2. Build everything
task build

# 3. Test everything
task test
```

See [docs/build.md](docs/build.md) for full instructions.

---

## Upgrade llama.cpp

```sh
task upgrade
```

This pulls the latest commit from `https://github.com/ggml-org/llama.cpp` and
rebuilds the bridge. Only bridge implementation files should ever need changes;
the public ABI and all language bindings remain stable.

---

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/design.md](docs/design.md) | Architecture and bridge philosophy |
| [docs/build.md](docs/build.md) | Build system and Taskfile reference |
| [docs/observability.md](docs/observability.md) | Events, metrics, and callbacks |
| [docs/testing.md](docs/testing.md) | Testing strategy and fixtures |
| [docs/publishing.md](docs/publishing.md) | Publish flows for Go, Java, and npm |
| [docs/upgrade.md](docs/upgrade.md) | Upgrading llama.cpp safely |