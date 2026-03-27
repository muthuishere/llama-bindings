# Build Guide

## Prerequisites

| Tool        | Minimum version | Purpose                          |
|-------------|-----------------|----------------------------------|
| `task`      | 3.x             | Task runner (Taskfile.yml)       |
| `cmake`     | 3.14            | Build system for llama.cpp       |
| `cc` / `gcc`| any             | C compiler for the bridge        |
| `go`        | 1.21            | Go binding and tests             |
| `java`      | 17              | Java binding and tests           |
| `mvn`       | 3.8             | Maven (Java build tool)          |
| `node`      | 18              | JavaScript binding and tests     |
| `npm`       | 9               | Node package manager             |

---

## Quick Start

```bash
# 1. Clone llama.cpp into third_party/
task init

# 2. Build llama.cpp and the C bridge
task build-llama
task build-bridge

# 3. Build and test each language binding
task build-go
task build-java
task build-js

# or run everything at once
task test
```

---

## Step-by-Step

### Step 1 — Initialise

```bash
task init
```

Clones `https://github.com/ggerganov/llama.cpp` into `third_party/llama.cpp`.
Safe to re-run; skips the clone if the directory already exists.

### Step 2 — Configure

```bash
task configure
```

Runs `cmake` to generate build files in `build/`.

### Step 3 — Build llama.cpp

```bash
task build-llama
```

Compiles the llama.cpp shared library (`libllama.so` / `.dylib`).

### Step 4 — Build the bridge

```bash
task build-bridge
```

Compiles `bridge/src/llama_bridge.c` into `libllama_bridge.so`.
Links against `libllama`.

### Step 5 — Build and test each binding

```bash
task build-go    # CGO build + go test
task build-java  # mvn test
task build-js    # npm install + npm test
```

---

## Environment Variables

| Variable                | Default          | Description                              |
|-------------------------|------------------|------------------------------------------|
| `LLAMA_TEST_MODEL`      | (unset)          | Path to GGUF model for integration tests |
| `LLAMA_BRIDGE_LIB_DIR`  | `<root>/build`   | Directory containing libllama_bridge     |

Set `LLAMA_TEST_MODEL` to enable the integration smoke tests:

```bash
LLAMA_TEST_MODEL=/path/to/model.gguf task test
```

---

## Clean

```bash
task clean
```

Removes the `build/` directory. `third_party/llama.cpp` is left intact.

---

## Troubleshooting

### Library not found at runtime

```
error while loading shared libraries: libllama_bridge.so
```

Add the build directory to the runtime linker path:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/build
```

On macOS use `DYLD_LIBRARY_PATH` instead.

### CGO compilation fails

Ensure the `CGO_CFLAGS` and `CGO_LDFLAGS` variables point to the correct
include and library directories.  The `task build-go` task sets these
automatically.

### JNA UnsatisfiedLinkError

Ensure `LLAMA_BRIDGE_LIB_DIR` is set to the directory containing
`libllama_bridge.so`, or pass `-DLLAMA_BRIDGE_LIB_DIR=<path>` to Maven.
