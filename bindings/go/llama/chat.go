package llama

/*
#cgo CFLAGS: -I${SRCDIR}/../../../bridge/include
#cgo LDFLAGS: -L${SRCDIR}/../../../bridge/build -lllama_bridge

#include "llama_bridge.h"
#include <stdlib.h>

// Forward declaration for the Go-exported CGO callback shim.
extern void goEventCallback(const char* event_json, void* user_data);
*/
import "C"
import (
	"encoding/json"
	"sync"
	"unsafe"
)

// ChatEngine provides chat inference backed by a loaded GGUF model.
// Create with NewChat; always call Close when done.
type ChatEngine struct {
	mu      sync.Mutex
	handle  C.llama_chat_engine_t
	onEvent EventCallback
	closed  bool
}

// callbackRegistry maps uintptr(handle) → EventCallback so CGO can find it.
var (
	callbackMu       sync.Mutex
	chatCallbacks    = make(map[uintptr]EventCallback)
)

//export goEventCallback
func goEventCallback(eventJSON *C.char, userData unsafe.Pointer) {
	key := uintptr(userData)
	callbackMu.Lock()
	cb, ok := chatCallbacks[key]
	callbackMu.Unlock()
	if !ok || cb == nil {
		return
	}
	var evt Event
	if err := json.Unmarshal([]byte(C.GoString(eventJSON)), &evt); err == nil {
		cb(evt)
	}
}

// NewChat creates a ChatEngine loaded from modelPath.
// opts.OnEvent is optional; pass LoadOptions{} for no callback.
func NewChat(modelPath string, opts LoadOptions) (*ChatEngine, error) {
	cPath := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cPath))

	// Allocate a temporary engine struct to obtain a stable key before the
	// real handle is known.  We use a dummy allocation as the CGO user_data.
	dummy := new(C.char)
	key := uintptr(unsafe.Pointer(dummy))

	callbackMu.Lock()
	if opts.OnEvent != nil {
		chatCallbacks[key] = opts.OnEvent
	}
	callbackMu.Unlock()

	var cb C.llama_event_cb
	if opts.OnEvent != nil {
		cb = C.llama_event_cb(C.goEventCallback)
	}

	handle := C.llama_chat_create(cPath, cb, unsafe.Pointer(dummy))
	if handle == nil {
		callbackMu.Lock()
		delete(chatCallbacks, key)
		callbackMu.Unlock()
		return nil, &LlamaError{Code: ErrCodeModelLoadFailed, Message: "llama_chat_create returned NULL"}
	}

	return &ChatEngine{
		handle:  handle,
		onEvent: opts.OnEvent,
	}, nil
}

// Chat runs a chat inference request and returns the normalized response.
func (e *ChatEngine) Chat(req ChatRequest, opts ChatOptions) (*ChatResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return nil, &LlamaError{Code: ErrCodeEngineClosed, Message: "engine is closed"}
	}

	// Merge generation options from ChatOptions into the request.
	if opts.Temperature != 0 || opts.MaxOutputTokens != 0 ||
		opts.TopP != 0 || opts.TopK != 0 || opts.Stop != nil {
		req.Generation = GenerationOptions{
			Temperature:     opts.Temperature,
			MaxOutputTokens: opts.MaxOutputTokens,
			TopP:            opts.TopP,
			TopK:            opts.TopK,
			Stop:            opts.Stop,
		}
	}

	// Set default response mode.
	if req.ResponseMode == "" {
		req.ResponseMode = ResponseModeText
	}

	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, &LlamaError{Code: ErrCodeInvalidRequest, Message: err.Error()}
	}

	cReq := C.CString(string(reqJSON))
	defer C.free(unsafe.Pointer(cReq))

	cResp := C.llama_chat_infer_json(e.handle, cReq)
	if cResp == nil {
		return nil, &LlamaError{Code: ErrCodeInferenceFailed, Message: "bridge returned NULL"}
	}
	defer C.llama_bridge_string_free(cResp)

	var resp ChatResponse
	if err := json.Unmarshal([]byte(C.GoString(cResp)), &resp); err != nil {
		return nil, &LlamaError{Code: ErrCodeInternalBridgeError, Message: err.Error()}
	}

	if resp.Type == "error" && resp.Error != nil {
		return nil, newLlamaError(resp.Error.Code, resp.Error.Message)
	}

	return &resp, nil
}

// Close releases the native engine and all associated resources.
// After Close, any call to Chat will return ErrCodeEngineClosed.
func (e *ChatEngine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return
	}
	e.closed = true
	C.llama_chat_destroy(e.handle)
	e.handle = nil
}
