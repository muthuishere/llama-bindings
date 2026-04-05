# Build Guide

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| [task](https://taskfile.dev) | ≥ 3 | Orchestration |
| git | any | Clone llama.cpp |
| cmake | ≥ 3.18 | Configure/build bridge |
| C compiler | gcc/clang | Bridge native build |
| Go | ≥ 1.21 | Go binding |
| JDK | 21 | Java binding (Panama FFM) |
| Maven | ≥ 3.9 | Java build/test/publish |
| Node.js | ≥ 18 | JS browser binding |
| npm | ≥ 9 | JS package management |
| Emscripten | ≥ 3.1 | WASM build |

## Upstream llama.cpp

The upstream repository is **https://github.com/ggml-org/llama.cpp**.

It is cloned into `third_party/llama.cpp-src/` by `task init` and is **not**
committed to this repository.

## Step-by-step

```sh
# 1. Clone llama.cpp (https://github.com/ggml-org/llama.cpp) and prepare dirs
task init

# 2. Build llama.cpp natively
task build-llama

# 3. Build the native bridge
task build-bridge

# 4. (Optional) Build the WASM target — requires Emscripten
task build-wasm

# 5. Build language bindings
task build-go
task build-java
task build-js    # requires build-wasm first
```

Or build everything at once:

```sh
task build
```

## Per-target details

### Bridge

The bridge is a shared C library (`libllama_bridge`). CMake configuration
lives in `bridge/CMakeLists.txt`. The build requires `third_party/llama.cpp-src`
to be built first (`task build-llama`).

### Go

```sh
cd bindings/go
go build ./...
```

The Go binding uses `purego`, not CGO. For local developer builds, run
`task build-bridge` first so `libllama_bridge` exists in `bridge/build/`.

### Java

```sh
cd bindings/java
mvn package -DskipTests -Dbridge.lib.dir=../../bridge/build
```

The Maven property `bridge.lib.dir` points to the directory containing
`libllama_bridge`.

### Browser JS / WASM

```sh
task build-wasm   # produces wasm/build/llama_bridge.{js,wasm}
task build-js     # copies artefacts to bindings/js-browser/dist/
```

## Build output locations

| Artefact | Location |
|----------|---------|
| Native bridge shared lib | `bridge/build/libllama_bridge.{so,dylib,dll}` |
| WASM module | `wasm/build/llama_bridge.{js,wasm}` |
| JS dist | `bindings/js-browser/dist/` |
| Java JAR | `bindings/java/target/llama-java-*.jar` |
