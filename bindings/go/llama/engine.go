// Package llama provides a thin Go binding to the llama_bridge C library.
//
// Usage:
//
// engine, err := llama.Load("model.gguf")
//
//	if err != nil {
//	   log.Fatal(err)
//	}
//
// defer engine.Close()
//
// // Raw completion (no chat template)
// result, err := engine.Complete("Say hello.")
//
// // Session-based chat — pass an ordered slice of Message values,
// // get back a Message{Role:"assistant", Content:"..."}.
//
//	msg, err := engine.Chat("sid-1", []llama.Message{
//	   {Role: "system", Content: "You are helpful."},
//	   {Role: "user",   Content: "What is 2+2?"},
//	})
//
// fmt.Println(msg.Content)
//
// // Multi-turn: only pass the new messages each turn.
//
//	msg, err = engine.Chat("sid-1", []llama.Message{
//	   {Role: "user", Content: "Tell me more."},
//	})
//
// // Inject a tool response before the user message.
//
//	msg, err = engine.Chat("sid-1", []llama.Message{
//	   {Role: "tool", Content: `{"result": 42}`},
//	   {Role: "user", Content: "What was the result?"},
//	})
//
// // Richer schema response — returns a map with role, content,
// // sessionId, and messageCount.
//
//	obj, err := engine.ChatWithObject("sid-1", []llama.Message{
//	   {Role: "user", Content: "Hello."},
//	})
//
// fmt.Println(obj["role"], obj["content"], obj["sessionId"], obj["messageCount"])
//
// // Clear session history
// engine.ChatSessionClear("sid-1")
//
// Build requirements:
//
// CGO_CFLAGS="-I<bridge/include> -I<llama.cpp/include>"
// CGO_LDFLAGS="-L<build> -lllama_bridge -lllama"
package llama

/*
#cgo CFLAGS: -I../../../bridge/include
#include "llama_bridge.h"
#include <stdlib.h>
*/
import "C"

import (
	"encoding/json"
	"errors"
	"unsafe"
)

// Message is the standard chat message type — a role+content pair.
// It is used both as the element type of the input slice passed to
// Chat and ChatWithObject, and as the response type returned by Chat.
//
// Supported roles: "system", "user", "assistant", "tool".
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
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

// Complete sends prompt to the engine and returns the generated completion text
// without applying a chat template.
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

// Chat performs a session-based chat turn and returns the assistant reply
// as a Message{Role:"assistant", Content:"..."}.
//
// messages is an ordered slice of Message values to append to the session
// history for this turn.  Supported roles: "system", "user", "assistant",
// "tool".  A "system" entry sets or replaces the session system prompt.
//
// The model's built-in chat template is applied to the full session history.
// The assistant response is automatically appended to the session history.
func (e *Engine) Chat(sessionID string, messages []Message) (Message, error) {
	if e == nil || e.handle == nil {
		return Message{}, errors.New("llama: engine is not initialised")
	}
	if sessionID == "" {
		return Message{}, errors.New("llama: sessionID must not be empty")
	}

	data, err := json.Marshal(messages)
	if err != nil {
		return Message{}, errors.New("llama: failed to serialize messages")
	}

	cSID := C.CString(sessionID)
	cMsgs := C.CString(string(data))
	defer C.free(unsafe.Pointer(cSID))
	defer C.free(unsafe.Pointer(cMsgs))

	cResult := C.llama_engine_chat_messages(e.handle, cSID, cMsgs)
	if cResult == nil {
		return Message{}, errors.New("llama: chat failed")
	}
	defer C.llama_engine_free_string(cResult)

	var msg Message
	if err := json.Unmarshal([]byte(C.GoString(cResult)), &msg); err != nil {
		return Message{}, errors.New("llama: failed to parse chat response")
	}
	return msg, nil
}

// ChatWithObject performs a session-based chat turn and returns the response
// as a map representing the JSON object.  The map always contains:
//
// "role"         → string  (always "assistant")
// "content"      → string  (the generated reply)
// "sessionId"    → string  (the session identifier)
// "messageCount" → float64 (total messages in the session after this turn)
//
// See Chat for details on how messages are applied to the session.
func (e *Engine) ChatWithObject(sessionID string, messages []Message) (map[string]interface{}, error) {
	if e == nil || e.handle == nil {
		return nil, errors.New("llama: engine is not initialised")
	}
	if sessionID == "" {
		return nil, errors.New("llama: sessionID must not be empty")
	}

	data, err := json.Marshal(messages)
	if err != nil {
		return nil, errors.New("llama: failed to serialize messages")
	}

	cSID := C.CString(sessionID)
	cMsgs := C.CString(string(data))
	defer C.free(unsafe.Pointer(cSID))
	defer C.free(unsafe.Pointer(cMsgs))

	cResult := C.llama_engine_chat_with_object_messages(e.handle, cSID, cMsgs)
	if cResult == nil {
		return nil, errors.New("llama: chatWithObject failed")
	}
	defer C.llama_engine_free_string(cResult)

	var result map[string]interface{}
	if err := json.Unmarshal([]byte(C.GoString(cResult)), &result); err != nil {
		return nil, errors.New("llama: failed to parse chatWithObject response")
	}
	return result, nil
}

// ChatSessionClear removes all history for the named session (including the
// system message).  The session slot is released for reuse.
func (e *Engine) ChatSessionClear(sessionID string) {
	if e == nil || e.handle == nil || sessionID == "" {
		return
	}
	cID := C.CString(sessionID)
	defer C.free(unsafe.Pointer(cID))
	C.llama_engine_chat_session_clear(e.handle, cID)
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
