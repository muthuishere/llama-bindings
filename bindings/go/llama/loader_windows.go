//go:build windows

package llama

import (
	"fmt"
	"syscall"
)

// openLibrary loads a DLL by path on Windows.
func openLibrary(path string) (uintptr, error) {
	dll, err := syscall.LoadDLL(path)
	if err != nil {
		return 0, fmt.Errorf("LoadDLL(%q): %w", path, err)
	}
	return uintptr(dll.Handle), nil
}

// systemLibraryCandidates returns DLL names searched on Windows when no
// embedded prebuilt is found.  Windows searches PATH automatically.
func systemLibraryCandidates() []string {
	return []string{
		"llama_bridge.dll",
		".\\llama_bridge.dll",
	}
}
