package llama

/*
#cgo CFLAGS: -I${SRCDIR}/../../../bridge/include
#cgo LDFLAGS: -L${SRCDIR}/../../../bridge/build -lllama_bridge

#include "llama_bridge.h"
#include <stdlib.h>

extern void goEmbedEventCallback(const char* event_json, void* user_data);
*/
import "C"
import (
	"encoding/json"
	"sync"
	"unsafe"
)

var (
	embedCallbackMu sync.Mutex
	embedCallbacks  = make(map[uintptr]EventCallback)
)

//export goEmbedEventCallback
func goEmbedEventCallback(eventJSON *C.char, userData unsafe.Pointer) {
	key := uintptr(userData)
	embedCallbackMu.Lock()
	cb, ok := embedCallbacks[key]
	embedCallbackMu.Unlock()
	if !ok || cb == nil {
		return
	}
	var evt Event
	if err := json.Unmarshal([]byte(C.GoString(eventJSON)), &evt); err == nil {
		cb(evt)
	}
}

// EmbedEngine provides text-embedding inference backed by a loaded GGUF model.
// Create with NewEmbed; always call Close when done.
type EmbedEngine struct {
	mu      sync.Mutex
	handle  C.llama_embed_engine_t
	onEvent EventCallback
	closed  bool
}

// NewEmbed creates an EmbedEngine loaded from modelPath.
func NewEmbed(modelPath string, opts LoadOptions) (*EmbedEngine, error) {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	dummy := new(C.char)
	key := uintptr(unsafe.Pointer(dummy))

	embedCallbackMu.Lock()
	if opts.OnEvent != nil {
		embedCallbacks[key] = opts.OnEvent
	}
	embedCallbackMu.Unlock()

	var cb C.llama_event_cb
	if opts.OnEvent != nil {
		cb = C.llama_event_cb(C.goEmbedEventCallback)
	}

	handle := C.llama_embed_create(cPath, cb, unsafe.Pointer(dummy))
	if handle == nil {
		embedCallbackMu.Lock()
		delete(embedCallbacks, key)
		embedCallbackMu.Unlock()
		return nil, &LlamaError{Code: ErrCodeModelLoadFailed, Message: "llama_embed_create returned NULL"}
	}

	return &EmbedEngine{
		handle:  handle,
		onEvent: opts.OnEvent,
	}, nil
}

// Embed generates an embedding vector for the given text.
// Returns a float32 slice whose length equals the model's embedding dimension.
func (e *EmbedEngine) Embed(text string, opts EmbedOptions) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return nil, &LlamaError{Code: ErrCodeEngineClosed, Message: "engine is closed"}
	}
	if text == "" {
		return nil, &LlamaError{Code: ErrCodeInvalidRequest, Message: "input text must not be empty"}
	}

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	var outLen C.int
	cVec := C.llama_embed_infer(e.handle, cText, &outLen)
	if cVec == nil {
		return nil, &LlamaError{Code: ErrCodeInferenceFailed, Message: "bridge returned NULL vector"}
	}
	defer C.llama_bridge_float_free(cVec)

	n := int(outLen)
	if n <= 0 {
		return nil, &LlamaError{Code: ErrCodeInferenceFailed, Message: "bridge returned empty vector"}
	}

	// Copy the C float array into a Go slice.
	result := make([]float32, n)
	cSlice := (*[1 << 28]C.float)(unsafe.Pointer(cVec))[:n:n]
	for i := 0; i < n; i++ {
		result[i] = float32(cSlice[i])
	}

	return result, nil
}

// Close releases the native engine and all associated resources.
func (e *EmbedEngine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return
	}
	e.closed = true
	C.llama_embed_destroy(e.handle)
	e.handle = nil
}
