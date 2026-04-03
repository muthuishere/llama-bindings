package llama

import (
"encoding/json"
"runtime"
"sync"
"sync/atomic"
"unsafe"

"github.com/ebitengine/purego"
)

// ChatEngine provides chat inference backed by a loaded GGUF model.
// Create with NewChat; always call Close when done.
type ChatEngine struct {
mu      sync.Mutex
handle  uintptr
cbKey   uintptr // stable registry key for this engine's callback
onEvent EventCallback
closed  bool
}

// chatNextKey is an atomic counter used to generate stable, unique callback keys.
// Using a counter (not a pointer address) avoids GC-related instability.
var chatNextKey uint64

// callbackRegistry maps stable cbKey → EventCallback.
var (
callbackMu    sync.RWMutex
chatCallbacks = make(map[uintptr]EventCallback)
)

// chatEventCallback is a C-callable function pointer created by purego.NewCallback.
// It receives observability events from the bridge and dispatches to Go callbacks.
// userData is the stable cbKey assigned before calling llama_chat_create.
var chatEventCallback = purego.NewCallback(func(eventJSONPtr unsafe.Pointer, userData uintptr) {
callbackMu.RLock()
cb, ok := chatCallbacks[userData]
callbackMu.RUnlock()
if !ok || cb == nil {
return
}
jsonStr := readCString(eventJSONPtr)
var evt Event
if err := json.Unmarshal([]byte(jsonStr), &evt); err == nil {
cb(evt)
}
})

// NewChat creates a ChatEngine loaded from modelPath.
// opts.OnEvent is optional; pass LoadOptions{} for no callback.
func NewChat(modelPath string, opts LoadOptions) (*ChatEngine, error) {
if err := checkBridge(); err != nil {
return nil, err
}

// Register the callback BEFORE calling llama_chat_create so that any events
// emitted during model loading (e.g. open_model_file) are captured.
var cbKey uintptr
var cbPtr uintptr
if opts.OnEvent != nil {
cbKey = uintptr(atomic.AddUint64(&chatNextKey, 1))
callbackMu.Lock()
chatCallbacks[cbKey] = opts.OnEvent
callbackMu.Unlock()
cbPtr = chatEventCallback
}

pathBuf := cStringBytes(modelPath)
handle := llamaChatCreate(bufPtr(pathBuf), cbPtr, cbKey)
runtime.KeepAlive(pathBuf)

if handle == 0 {
if cbKey != 0 {
callbackMu.Lock()
delete(chatCallbacks, cbKey)
callbackMu.Unlock()
}
return nil, &LlamaError{Code: ErrCodeModelLoadFailed, Message: "llama_chat_create returned NULL"}
}

return &ChatEngine{
handle:  handle,
cbKey:   cbKey,
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

if req.ResponseMode == "" {
req.ResponseMode = ResponseModeText
}

reqJSON, err := json.Marshal(req)
if err != nil {
return nil, &LlamaError{Code: ErrCodeInvalidRequest, Message: err.Error()}
}

reqBuf := cStringBytes(string(reqJSON))
respPtr := llamaChatInferJson(e.handle, bufPtr(reqBuf))
runtime.KeepAlive(reqBuf)

if respPtr == nil {
return nil, &LlamaError{Code: ErrCodeInferenceFailed, Message: "bridge returned NULL"}
}

respStr := readCString(respPtr)
llamaStringFree(respPtr)

var resp ChatResponse
if err := json.Unmarshal([]byte(respStr), &resp); err != nil {
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

if e.cbKey != 0 {
callbackMu.Lock()
delete(chatCallbacks, e.cbKey)
callbackMu.Unlock()
}

llamaChatDestroy(e.handle)
e.handle = 0
}
