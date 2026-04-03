// Package llama provides Go bindings to the llama.cpp bridge library.
//
// # Architecture
//
// Go → CGO → native bridge → llama.cpp
//
// Only the bridge layer knows about upstream llama.cpp internals.
// This package exposes a stable, idiomatic Go API.
//
// # Quick start
//
//	chat, err := llama.NewChat("chat-model.gguf", llama.LoadOptions{})
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer chat.Close()
//
//	resp, err := chat.Chat(llama.ChatRequest{
//	    Messages: []llama.Message{
//	        {Role: "user", Content: "Hello!"},
//	    },
//	}, llama.ChatOptions{Temperature: 0.7})
//
//	embed, err := llama.NewEmbed("embed-model.gguf", llama.LoadOptions{})
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer embed.Close()
//
//	vec, err := embed.Embed("hello world", llama.EmbedOptions{})
package llama
