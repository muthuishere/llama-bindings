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
// // Session-based chat — returns a ChatMessage
//
//	msg, err := engine.Chat("sid-1", llama.ChatRequest{
//	   SystemMessage: "You are helpful.",
//	   UserMessage:   "What is 2+2?",
//	})
//
// fmt.Println(msg.Content)
//
// // Session-based chat with schema response (includes session metadata)
//
//	resp, err := engine.ChatWithObject("sid-1", llama.ChatRequest{
//	   UserMessage: "Tell me more.",
//	})
//
// fmt.Printf("role=%s content=%s session=%s count=%d\n",
//
//	resp.Role, resp.Content, resp.SessionID, resp.MessageCount)
//
// // Inject a tool response into the conversation
//
//	msg, err = engine.Chat("sid-1", llama.ChatRequest{
//	   ToolMessage: `{"result": 42}`,
//	   UserMessage: "What was the result?",
//	})
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

// ChatRequest is the input to Chat and ChatWithObject.
// UserMessage is the main required field; all others are optional.
type ChatRequest struct {
	// SystemMessage sets or replaces the session's system prompt.
	// Pass "" to leave the existing system prompt unchanged.
	SystemMessage string

	// UserMessage is the user turn for this call.
	UserMessage string

	// AssistantMessage injects a prior assistant turn into the session
	// before UserMessage (useful for few-shot context or correction).
	AssistantMessage string

	// ToolMessage injects a tool-use response (role "tool") into the
	// session before UserMessage.
	ToolMessage string
}

// ChatMessage is the response returned by Chat.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatResponse is the richer response returned by ChatWithObject.
// It includes session metadata in addition to the assistant reply.
type ChatResponse struct {
	Role         string `json:"role"`
	Content      string `json:"content"`
	SessionID    string `json:"sessionId"`
	MessageCount int    `json:"messageCount"`
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

// Chat performs a session-based chat turn and returns a ChatMessage.
//
// The engine maintains conversation history keyed by sessionID.
// The ChatRequest may contain any combination of SystemMessage, UserMessage,
// AssistantMessage, and ToolMessage; all non-empty fields are appended to the
// session in that order before running inference.
//
// The model's built-in chat template is applied to the full session history.
// The assistant response is automatically appended to the session history.
func (e *Engine) Chat(sessionID string, req ChatRequest) (ChatMessage, error) {
	if e == nil || e.handle == nil {
		return ChatMessage{}, errors.New("llama: engine is not initialised")
	}
	if sessionID == "" {
		return ChatMessage{}, errors.New("llama: sessionID must not be empty")
	}

	cSID := C.CString(sessionID)
	cSys := C.CString(req.SystemMessage)
	cUser := C.CString(req.UserMessage)
	cAsst := C.CString(req.AssistantMessage)
	cTool := C.CString(req.ToolMessage)
	defer func() {
		C.free(unsafe.Pointer(cSID))
		C.free(unsafe.Pointer(cSys))
		C.free(unsafe.Pointer(cUser))
		C.free(unsafe.Pointer(cAsst))
		C.free(unsafe.Pointer(cTool))
	}()

	cResult := C.llama_engine_chat(e.handle, cSID, cSys, cUser, cAsst, cTool)
	if cResult == nil {
		return ChatMessage{}, errors.New("llama: chat failed")
	}
	defer C.llama_engine_free_string(cResult)

	var msg ChatMessage
	if err := json.Unmarshal([]byte(C.GoString(cResult)), &msg); err != nil {
		return ChatMessage{}, errors.New("llama: failed to parse chat response")
	}
	return msg, nil
}

// ChatWithObject performs a session-based chat turn and returns a ChatResponse
// that includes session metadata (SessionID and MessageCount) in addition to
// the assistant reply.
//
// See Chat for details on how the request fields are handled.
func (e *Engine) ChatWithObject(sessionID string, req ChatRequest) (ChatResponse, error) {
	if e == nil || e.handle == nil {
		return ChatResponse{}, errors.New("llama: engine is not initialised")
	}
	if sessionID == "" {
		return ChatResponse{}, errors.New("llama: sessionID must not be empty")
	}

	cSID := C.CString(sessionID)
	cSys := C.CString(req.SystemMessage)
	cUser := C.CString(req.UserMessage)
	cAsst := C.CString(req.AssistantMessage)
	cTool := C.CString(req.ToolMessage)
	defer func() {
		C.free(unsafe.Pointer(cSID))
		C.free(unsafe.Pointer(cSys))
		C.free(unsafe.Pointer(cUser))
		C.free(unsafe.Pointer(cAsst))
		C.free(unsafe.Pointer(cTool))
	}()

	cResult := C.llama_engine_chat_with_object(
		e.handle, cSID, cSys, cUser, cAsst, cTool)
	if cResult == nil {
		return ChatResponse{}, errors.New("llama: chatWithObject failed")
	}
	defer C.llama_engine_free_string(cResult)

	var resp ChatResponse
	if err := json.Unmarshal([]byte(C.GoString(cResult)), &resp); err != nil {
		return ChatResponse{}, errors.New("llama: failed to parse chatWithObject response")
	}
	return resp, nil
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
