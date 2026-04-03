//go:build windows && amd64

package llama

import _ "embed"

//go:embed prebuilt/windows_amd64/llama_bridge.dll
var embeddedLib []byte

func embeddedNativeLib() []byte { return embeddedLib }
func nativeLibName() string     { return "llama_bridge.dll" }
