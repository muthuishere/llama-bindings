// Package llama provides a thin Go binding to the llama_bridge C library.
//
// Usage:
//
//	engine, err := llama.Load("model.gguf")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer engine.Close()
//
//	// Simple completion
//	result, err := engine.Complete("Say hello.")
//
//	// One-shot chat
//	result, err = engine.Chat("You are helpful.", "What is 2+2?")
//
//	// Multi-message chat (chatWithObject)
//	result, err = engine.ChatWithMessages([]llama.Message{
//	    {Role: "system",    Content: "You are helpful."},
//	    {Role: "user",      Content: "What is 2+2?"},
//	})
//
//	// Session-based multi-turn chat
//	result, err = engine.ChatSession("sid-1", "Hello!")
//	result, err = engine.ChatSession("sid-1", "What did I just say?")
//
//	// Chat with tool definitions
//	tools := `[{"name":"add","description":"Add two numbers","parameters":{"a":{"type":"number"},"b":{"type":"number"}}}]`
//	result, err = engine.ChatWithTools([]llama.Message{{Role:"user",Content:"Add 3 and 4"}}, tools)
//
// Build requirements:
//
//	CGO_CFLAGS="-I<bridge/include> -I<llama.cpp/include>"
//	CGO_LDFLAGS="-L<build> -lllama_bridge -lllama"
package llama

/*
#cgo CFLAGS: -I../../../bridge/include
#include "llama_bridge.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"unsafe"
)

// Message represents a single chat message with a role and content.
// Role must be one of "system", "user", or "assistant".
type Message struct {
	Role    string
	Content string
}

// Engine wraps an opaque llama_engine_t handle.
// Create one with Load and release it with Close.
type Engine struct {
	handle C.llama_engine_t
}

// Load creates a new Engine by loading a GGUF model from modelPath.
// Returns an error if the model cannot be loaded.
func Load(modelPath string) (*Engine, error) {
	if modelPath == "" {
		return nil, errors.New("llama: model path must not be empty")
	}

	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	handle := C.llama_engine_create(cPath)
	if handle == nil {
		return nil, errors.New("llama: failed to create engine (check model path and logs)")
	}

	return &Engine{handle: handle}, nil
}

// Complete sends prompt to the engine and returns the generated completion text.
// Returns an error if the engine handle is invalid or inference fails.
func (e *Engine) Complete(prompt string) (string, error) {
	if e == nil || e.handle == nil {
		return "", errors.New("llama: engine is not initialised")
	}

	cPrompt := C.CString(prompt)
	defer C.free(unsafe.Pointer(cPrompt))

	cResult := C.llama_engine_complete(e.handle, cPrompt)
	if cResult == nil {
		return "", errors.New("llama: completion failed")
	}
	defer C.llama_engine_free_string(cResult)

	return C.GoString(cResult), nil
}

// Chat performs a one-shot chat with an optional system message and a user
// message.  The model's built-in chat template is applied automatically.
// Pass an empty string for systemMsg to omit the system message.
func (e *Engine) Chat(systemMsg, userMsg string) (string, error) {
	if e == nil || e.handle == nil {
		return "", errors.New("llama: engine is not initialised")
	}

	cSystem := C.CString(systemMsg)
	defer C.free(unsafe.Pointer(cSystem))
	cUser := C.CString(userMsg)
	defer C.free(unsafe.Pointer(cUser))

	cResult := C.llama_engine_chat(e.handle, cSystem, cUser)
	if cResult == nil {
		return "", errors.New("llama: chat failed")
	}
	defer C.llama_engine_free_string(cResult)
	return C.GoString(cResult), nil
}

// ChatWithMessages sends an explicit ordered slice of Messages to the engine
// (the chatWithObject equivalent).  Role values must be "system", "user", or
// "assistant".  The model's chat template is applied to the full message list.
func (e *Engine) ChatWithMessages(messages []Message) (string, error) {
	if e == nil || e.handle == nil {
		return "", errors.New("llama: engine is not initialised")
	}
	if len(messages) == 0 {
		return "", errors.New("llama: messages must not be empty")
	}

	n := len(messages)

	// Allocate C string arrays for roles and contents.
	cRoles := make([]*C.char, n)
	cConts := make([]*C.char, n)
	for i, m := range messages {
		cRoles[i] = C.CString(m.Role)
		cConts[i] = C.CString(m.Content)
	}
	defer func() {
		for i := 0; i < n; i++ {
			C.free(unsafe.Pointer(cRoles[i]))
			C.free(unsafe.Pointer(cConts[i]))
		}
	}()

	cResult := C.llama_engine_chat_with_messages(
		e.handle,
		(**C.char)(unsafe.Pointer(&cRoles[0])),
		(**C.char)(unsafe.Pointer(&cConts[0])),
		C.int(n),
	)
	if cResult == nil {
		return "", errors.New("llama: chat_with_messages failed")
	}
	defer C.llama_engine_free_string(cResult)
	return C.GoString(cResult), nil
}

// ChatSession performs a multi-turn chat using a named session.
// The engine maintains conversation history keyed by sessionID.
// Each call appends userMsg to the session and returns the next assistant turn.
// Call ChatSessionSetSystem before the first turn to set a system prompt.
func (e *Engine) ChatSession(sessionID, userMsg string) (string, error) {
	if e == nil || e.handle == nil {
		return "", errors.New("llama: engine is not initialised")
	}
	if sessionID == "" {
		return "", errors.New("llama: sessionID must not be empty")
	}

	cID := C.CString(sessionID)
	defer C.free(unsafe.Pointer(cID))
	cMsg := C.CString(userMsg)
	defer C.free(unsafe.Pointer(cMsg))

	cResult := C.llama_engine_chat_session(e.handle, cID, cMsg)
	if cResult == nil {
		return "", errors.New("llama: chat session failed")
	}
	defer C.llama_engine_free_string(cResult)
	return C.GoString(cResult), nil
}

// ChatSessionSetSystem sets (or replaces) the system message for a named
// session.  Call this before the first ChatSession call when a system prompt
// is needed.  Pass an empty string to clear the existing system message.
func (e *Engine) ChatSessionSetSystem(sessionID, systemMsg string) {
	if e == nil || e.handle == nil {
		return
	}
	cID  := C.CString(sessionID)
	defer C.free(unsafe.Pointer(cID))
	cSys := C.CString(systemMsg)
	defer C.free(unsafe.Pointer(cSys))

	C.llama_engine_chat_session_set_system(e.handle, cID, cSys)
}

// ChatSessionClear removes all history for the named session, including the
// system message.  The session slot is released for reuse.
func (e *Engine) ChatSessionClear(sessionID string) {
	if e == nil || e.handle == nil {
		return
	}
	cID := C.CString(sessionID)
	defer C.free(unsafe.Pointer(cID))

	C.llama_engine_chat_session_clear(e.handle, cID)
}

// ChatWithTools sends messages together with a JSON tool-definition list.
// toolsJSON must be a JSON array of tool objects in OpenAI-compatible format,
// e.g.:
//
//	[{"name":"get_weather","description":"...","parameters":{...}}]
//
// The tool definitions are injected into the system message so that the model
// can reason about available tools.  The raw model output is returned — the
// caller is responsible for parsing and executing any tool calls the model
// emits.
func (e *Engine) ChatWithTools(messages []Message, toolsJSON string) (string, error) {
	if e == nil || e.handle == nil {
		return "", errors.New("llama: engine is not initialised")
	}
	if len(messages) == 0 {
		return "", errors.New("llama: messages must not be empty")
	}

	n := len(messages)

	cRoles := make([]*C.char, n)
	cConts := make([]*C.char, n)
	for i, m := range messages {
		cRoles[i] = C.CString(m.Role)
		cConts[i] = C.CString(m.Content)
	}
	defer func() {
		for i := 0; i < n; i++ {
			C.free(unsafe.Pointer(cRoles[i]))
			C.free(unsafe.Pointer(cConts[i]))
		}
	}()

	cTools := C.CString(toolsJSON)
	defer C.free(unsafe.Pointer(cTools))

	cResult := C.llama_engine_chat_with_tools(
		e.handle,
		(**C.char)(unsafe.Pointer(&cRoles[0])),
		(**C.char)(unsafe.Pointer(&cConts[0])),
		C.int(n),
		cTools,
	)
	if cResult == nil {
		return "", errors.New("llama: chat_with_tools failed")
	}
	defer C.llama_engine_free_string(cResult)
	return C.GoString(cResult), nil
}

// Close destroys the engine and frees all associated resources.
// After Close, the Engine must not be used.
func (e *Engine) Close() error {
	if e == nil || e.handle == nil {
		return nil
	}
	C.llama_engine_destroy(e.handle)
	e.handle = nil
	return nil
}

