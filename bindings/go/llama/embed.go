package llama

import (
"encoding/json"
"runtime"
"sync"
"unsafe"

"github.com/ebitengine/purego"
)

// EmbedEngine provides text-embedding inference backed by a loaded GGUF model.
// Create with NewEmbed; always call Close when done.
type EmbedEngine struct {
mu      sync.Mutex
handle  uintptr
onEvent EventCallback
closed  bool

// cbPtr holds the purego callback function pointer for the lifetime of the engine.
cbPtr uintptr
}

var (
embedCallbackMu sync.RWMutex
embedCallbacks  = make(map[uintptr]EventCallback)
)

// embedEventCallback is a C-callable function pointer created by purego.NewCallback.
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

pathBuf := cStringBytes(modelPath)

var cbPtr uintptr
if opts.OnEvent != nil {
cbPtr = embedEventCallback
}

handle := llamaEmbedCreate(bufPtr(pathBuf), cbPtr, 0)
runtime.KeepAlive(pathBuf)

if handle == 0 {
return nil, &LlamaError{Code: ErrCodeModelLoadFailed, Message: "llama_embed_create returned NULL"}
}

if opts.OnEvent != nil {
embedCallbackMu.Lock()
embedCallbacks[handle] = opts.OnEvent
embedCallbackMu.Unlock()
}

return &EmbedEngine{
handle:  handle,
onEvent: opts.OnEvent,
cbPtr:   cbPtr,
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

embedCallbackMu.Lock()
delete(embedCallbacks, e.handle)
embedCallbackMu.Unlock()

llamaEmbedDestroy(e.handle)
e.handle = 0
}
