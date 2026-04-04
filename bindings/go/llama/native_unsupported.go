//go:build !((linux && (amd64 || arm64)) || (darwin && (amd64 || arm64)) || (windows && amd64))

package llama

// embeddedNativeLib returns nil on platforms that don't have a prebuilt binary
// bundled. The loader will fall back to searching the system library path.
func embeddedNativeLib() []byte { return nil }
func nativeLibName() string     { return "" }
