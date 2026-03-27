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
//	result, err := engine.Complete("Say hello.")
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Println(result)
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
