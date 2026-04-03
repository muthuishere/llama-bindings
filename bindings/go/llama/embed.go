package llama

import (
"encoding/json"
"runtime"
"sync"
"sync/atomic"
"unsafe"

"github.com/ebitengine/purego"
)

// EmbedEngine provides text-embedding inference backed by a loaded GGUF model.
// Create with NewEmbed; always call Close when done.
type EmbedEngine struct {
mu      sync.Mutex
handle  uintptr
cbKey   uintptr // stable registry key for this engine's callback
onEvent EventCallback
closed  bool
}

// embedNextKey is an atomic counter used to generate stable, unique callback keys.
var embedNextKey uint64

var (
embedCallbackMu sync.RWMutex
embedCallbacks  = make(map[uintptr]EventCallback)
)

// embedEventCallback is a C-callable function pointer created by purego.NewCallback.
// userData is the stable cbKey assigned before calling llama_embed_create.
var embedEventCallback = purego.NewCallback(func(eventJSONPtr unsafe.Pointer, userData uintptr) {
embedCallbackMu.RLock()
cb, ok := embedCallbacks[userData]
embedCallbackMu.RUnlock()
if !ok || cb == nil {
return
}
jsonStr := readCString(eventJSONPtr)
var evt Event
if err := json.Unmarshal([]byte(jsonStr), &evt); err == nil {
cb(evt)
}
})

// NewEmbed creates an EmbedEngine loaded from modelPath.
func NewEmbed(modelPath string, opts LoadOptions) (*EmbedEngine, error) {
if err := checkBridge(); err != nil {
return nil, err
}

// Register callback BEFORE calling llama_embed_create so that load-time
// events (e.g. open_model_file) are captured.
var cbKey uintptr
var cbPtr uintptr
if opts.OnEvent != nil {
cbKey = uintptr(atomic.AddUint64(&embedNextKey, 1))
embedCallbackMu.Lock()
embedCallbacks[cbKey] = opts.OnEvent
embedCallbackMu.Unlock()
cbPtr = embedEventCallback
}

pathBuf := cStringBytes(modelPath)
handle := llamaEmbedCreate(bufPtr(pathBuf), cbPtr, cbKey)
runtime.KeepAlive(pathBuf)

if handle == 0 {
if cbKey != 0 {
embedCallbackMu.Lock()
delete(embedCallbacks, cbKey)
embedCallbackMu.Unlock()
}
return nil, &LlamaError{Code: ErrCodeModelLoadFailed, Message: "llama_embed_create returned NULL"}
}

return &EmbedEngine{
handle:  handle,
cbKey:   cbKey,
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

textBuf := cStringBytes(text)

var outLen int32
vecPtr := llamaEmbedInfer(e.handle, bufPtr(textBuf), unsafe.Pointer(&outLen))
runtime.KeepAlive(textBuf)

if vecPtr == nil {
return nil, &LlamaError{Code: ErrCodeInferenceFailed, Message: "bridge returned NULL vector"}
}

n := int(outLen)
if n <= 0 {
llamaFloatFree(vecPtr)
return nil, &LlamaError{Code: ErrCodeInferenceFailed, Message: "bridge returned empty vector"}
}

result := readFloats(vecPtr, n)
llamaFloatFree(vecPtr)

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

if e.cbKey != 0 {
embedCallbackMu.Lock()
delete(embedCallbacks, e.cbKey)
embedCallbackMu.Unlock()
}

llamaEmbedDestroy(e.handle)
e.handle = 0
}
