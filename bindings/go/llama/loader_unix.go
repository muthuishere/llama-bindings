//go:build !windows

package llama

import "github.com/ebitengine/purego"

// openLibrary opens a shared library by path on POSIX systems.
func openLibrary(path string) (uintptr, error) {
	return purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}

// systemLibraryCandidates returns names that openLibrary will search for on
// POSIX when no embedded prebuilt is found.  The dynamic linker resolves them
// against LD_LIBRARY_PATH / DYLD_LIBRARY_PATH / default search paths.
func systemLibraryCandidates() []string {
	return []string{
		"libllama_bridge.so",
		"./libllama_bridge.so",
		"libllama_bridge.dylib",
		"./libllama_bridge.dylib",
	}
}
