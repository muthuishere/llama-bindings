//go:build darwin && arm64

package llama

import _ "embed"

//go:embed prebuilt/darwin_arm64/libllama_bridge.dylib
var embeddedLib []byte

func embeddedNativeLib() []byte { return embeddedLib }
func nativeLibName() string     { return "libllama_bridge.dylib" }
