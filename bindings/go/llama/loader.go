package llama

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"unsafe"

"github.com/ebitengine/purego"
)

// ─────────────────────────────────────────────────────────────────────────────
// Package-level bridge function variables.
// Populated by init() and called by ChatEngine / EmbedEngine.
//
// Parameter / return types use unsafe.Pointer for string buffers and C-heap
// pointers so that go vet is satisfied — no uintptr arithmetic conversions.
// Opaque engine handles use uintptr (used as map keys for callbacks).
// ─────────────────────────────────────────────────────────────────────────────

var (
bridgeOnce sync.Once
bridgeErr  error // non-nil when the native library could not be loaded

// Chat
llamaChatCreate    func(modelPath unsafe.Pointer, onEvent, userData uintptr) uintptr
llamaChatInferJson func(engine uintptr, requestJson unsafe.Pointer) unsafe.Pointer
llamaChatDestroy   func(engine uintptr)

// Embed
llamaEmbedCreate  func(modelPath unsafe.Pointer, onEvent, userData uintptr) uintptr
llamaEmbedInfer   func(engine uintptr, inputText, outLen unsafe.Pointer) unsafe.Pointer
llamaEmbedDestroy func(engine uintptr)

// Memory management
llamaStringFree func(s unsafe.Pointer)
llamaFloatFree  func(p unsafe.Pointer)
)

func init() {
bridgeOnce.Do(func() {
lib, err := loadBridgeLibrary()
if err != nil {
bridgeErr = err
return
}
purego.RegisterLibFunc(&llamaChatCreate, lib, "llama_chat_create")
purego.RegisterLibFunc(&llamaChatInferJson, lib, "llama_chat_infer_json")
purego.RegisterLibFunc(&llamaChatDestroy, lib, "llama_chat_destroy")
purego.RegisterLibFunc(&llamaEmbedCreate, lib, "llama_embed_create")
purego.RegisterLibFunc(&llamaEmbedInfer, lib, "llama_embed_infer")
purego.RegisterLibFunc(&llamaEmbedDestroy, lib, "llama_embed_destroy")
purego.RegisterLibFunc(&llamaStringFree, lib, "llama_bridge_string_free")
purego.RegisterLibFunc(&llamaFloatFree, lib, "llama_bridge_float_free")
})
}

// checkBridge returns a LlamaError if the native bridge failed to load.
func checkBridge() error {
if bridgeErr != nil {
return &LlamaError{
Code:    ErrCodeModelLoadFailed,
Message: "native bridge unavailable: " + bridgeErr.Error(),
}
}
return nil
}

// loadBridgeLibrary tries the following sources in order:
//  1. Embedded prebuilt binary (bundled at compile time via go:embed).
//  2. Explicit / repo-local developer build paths.
//  3. System library path (for developer builds where the loader search path
//     already includes libllama_bridge).
func loadBridgeLibrary() (uintptr, error) {
	// 1. Try embedded prebuilt
	data := embeddedNativeLib()
	if len(data) > 0 {
		tmpPath, cleanup, err := extractLibToTemp(data, nativeLibName())
		if err == nil {
			lib, err := openLibrary(tmpPath)
			if err == nil {
				_ = cleanup // temp file persists until process exit; OS cleans up
				return lib, nil
			}
			cleanup()
		}
	}

	// 2. Try explicit / repo-local developer build locations.
	for _, path := range developerLibraryCandidates() {
		lib, err := openLibrary(path)
		if err == nil {
			return lib, nil
		}
	}

	// 3. Try well-known system library names.
	for _, name := range systemLibraryCandidates() {
		lib, err := openLibrary(name)
		if err == nil {
			return lib, nil
		}
	}

	return 0, fmt.Errorf("libllama_bridge not found — run `task build-bridge`, set LLAMA_BRIDGE_PATH, or use a published release that bundles prebuilt natives")
}

func developerLibraryCandidates() []string {
	libName := nativeLibName()
	var candidates []string

	if p := os.Getenv("LLAMA_BRIDGE_PATH"); p != "" {
		candidates = append(candidates, p)
	}

	addBridgeBuildPaths := func(base string) {
		if base == "" {
			return
		}
		base = filepath.Clean(base)
		candidates = append(candidates, filepath.Join(base, libName))
		for dir := base; ; dir = filepath.Dir(dir) {
			candidates = append(candidates, filepath.Join(dir, "bridge", "build", libName))
			parent := filepath.Dir(dir)
			if parent == dir {
				break
			}
		}
	}

	if cwd, err := os.Getwd(); err == nil {
		addBridgeBuildPaths(cwd)
	}
	if exe, err := os.Executable(); err == nil {
		addBridgeBuildPaths(filepath.Dir(exe))
	}

	seen := make(map[string]struct{}, len(candidates))
	deduped := make([]string, 0, len(candidates))
	for _, path := range candidates {
		if path == "" {
			continue
		}
		if _, ok := seen[path]; ok {
			continue
		}
		seen[path] = struct{}{}
		deduped = append(deduped, path)
	}
	return deduped
}

// extractLibToTemp writes data to a uniquely named temporary file that already
// carries the correct extension (e.g. .dylib, .so, .dll).
// The pattern "llama_bridge_*_<libName>" keeps the unique random infix while
// preserving the extension, so no renaming is needed.
func extractLibToTemp(data []byte, libName string) (string, func(), error) {
f, err := os.CreateTemp(os.TempDir(), "llama_bridge_*_"+libName)
if err != nil {
return "", nil, err
}
path := f.Name()
cleanup := func() { os.Remove(path) }

if _, err := f.Write(data); err != nil {
f.Close()
cleanup()
return "", nil, err
}
if err := f.Close(); err != nil {
cleanup()
return "", nil, err
}
if err := os.Chmod(path, 0755); err != nil {
cleanup()
return "", nil, err
}

return path, cleanup, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// C-interop helpers (no CGO needed)
// ─────────────────────────────────────────────────────────────────────────────

// cStringBytes returns a NUL-terminated byte slice for s.
// The caller must keep the slice alive via runtime.KeepAlive for any C call
// that uses the pointer obtained from bufPtr(b).
func cStringBytes(s string) []byte {
b := make([]byte, len(s)+1)
copy(b, s)
return b
}

// bufPtr returns a pointer to the first byte of b as unsafe.Pointer.
// b must be kept alive via runtime.KeepAlive after any C call using this pointer.
func bufPtr(b []byte) unsafe.Pointer {
if len(b) == 0 {
return nil
}
return unsafe.Pointer(&b[0])
}

// readCString copies a NUL-terminated C string at ptr into a Go string.
// ptr is C-owned memory; the GC has no stake in it.
func readCString(ptr unsafe.Pointer) string {
if ptr == nil {
return ""
}
// Cast to a fixed-size array — the standard Go FFI idiom that avoids
// uintptr arithmetic in unsafe.Pointer conversions (approved by go vet).
const maxLen = 64 * 1024 // 64 KB is ample for any JSON response
cptr := (*[maxLen]byte)(ptr)
n := 0
for cptr[n] != 0 {
n++
}
return string(cptr[:n])
}

// maxEmbedDim is the maximum number of float32 values we will read from a
// C-heap float array. 1<<22 = 4 M floats = 16 MB; well above any embedding
// dimension in current models while staying within safe allocation bounds.
const maxEmbedDim = 1 << 22

// readFloats copies n float32 values from the C array at ptr into a Go slice.
// Returns nil if n is out of the expected range.
func readFloats(ptr unsafe.Pointer, n int) []float32 {
if n <= 0 || n > maxEmbedDim {
return nil
}
cptr := (*[maxEmbedDim]float32)(ptr)
result := make([]float32, n)
copy(result, cptr[:n])
return result
}
