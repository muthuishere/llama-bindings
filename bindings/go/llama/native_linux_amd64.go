//go:build linux && amd64

package llama

import _ "embed"

//go:embed prebuilt/linux_amd64/libllama_bridge.so
var embeddedLib []byte

func embeddedNativeLib() []byte { return embeddedLib }
func nativeLibName() string     { return "libllama_bridge.so" }
