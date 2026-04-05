# Go Binding

Go library for [llama.cpp](https://github.com/ggml-org/llama.cpp) chat and embedding inference.

**Runtime:** Go → purego → native bridge → llama.cpp

## Installation

```sh
go get github.com/muthuishere/llama-bindings/go/llama
```

## Prerequisites

- Go ≥ 1.21
- Native bridge built (`task build-bridge`)
- `task build-bridge` run first so `libllama_bridge` exists in `bridge/build/`

## Quick start

### Chat

```go
import "github.com/muthuishere/llama-bindings/go/llama"

chat, err := llama.NewChat("chat-model.gguf", llama.LoadOptions{
    OnEvent: func(e llama.Event) { fmt.Println(e.Event) },
})
if err != nil {
    return err
}
defer chat.Close()

resp, err := chat.Chat(llama.ChatRequest{
    Messages: []llama.Message{
        {Role: "system", Content: "You are helpful."},
        {Role: "user",   Content: "Extract person data."},
    },
    ResponseMode: llama.ResponseModeJSONSchema,
    Schema: &llama.Schema{
        Name: "person_extract",
        Schema: map[string]interface{}{
            "type": "object",
            "properties": map[string]interface{}{
                "name": map[string]interface{}{"type": "string"},
                "age":  map[string]interface{}{"type": "integer"},
            },
            "required": []string{"name", "age"},
        },
    },
}, llama.ChatOptions{Temperature: 0.0, MaxOutputTokens: 128})
```

### Embed

```go
embed, err := llama.NewEmbed("embed-model.gguf", llama.LoadOptions{})
if err != nil {
    return err
}
defer embed.Close()

vec, err := embed.Embed("hello world", llama.EmbedOptions{})
// vec is []float32
```

## API reference

| Symbol | Description |
|--------|-------------|
| `NewChat(path, LoadOptions)` | Create a chat engine |
| `(*ChatEngine).Chat(ChatRequest, ChatOptions)` | Run chat inference |
| `(*ChatEngine).Close()` | Release engine |
| `NewEmbed(path, LoadOptions)` | Create an embedding engine |
| `(*EmbedEngine).Embed(text, EmbedOptions)` | Generate embedding vector |
| `(*EmbedEngine).Close()` | Release engine |

## Tests

```sh
task test-go
# or
cd bindings/go && go test ./...
```

## Build

```sh
task build-go
```

Upstream llama.cpp: https://github.com/ggml-org/llama.cpp
