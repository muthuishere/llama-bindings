# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

llama-bindings is a cross-language binding library for [llama.cpp](https://github.com/ggml-org/llama.cpp). It provides Go, Java, and Browser JS/WASM bindings through a single shared C bridge layer. The bridge is the only code that touches llama.cpp internals — all language bindings talk exclusively to the bridge's stable C ABI defined in `bridge/include/llama_bridge.h`.

## Build Commands

All orchestration uses [Task](https://taskfile.dev). Run `task --list` to see everything.

```bash
task init             # Clone llama.cpp into third_party/llama.cpp-src/
task build-bridge     # Build llama.cpp + native bridge (libllama_bridge)
task build            # Build all targets (bridge + Go + Java + JS/WASM)

task test             # Run all unit tests (Go + Java + JS)
task test-go          # Go unit tests only
task test-java        # Java unit tests only (Maven)
task test-js          # JS unit tests only (Jest)

task download-model   # Fetch Gemma 4 E2B + nomic-embed to ~/llama-bindings-conf/models/
task e2e              # Run all e2e tests (requires build-bridge + download-model)
task e2e-go           # Go e2e only
task e2e-java         # Java e2e only
task e2e-js           # Playwright browser e2e only

task upgrade          # Pull latest llama.cpp + rebuild bridge
```

### Running a single test

```bash
# Go — single test function
cd bindings/go/llama && go test -run TestChatTextMode -v -count=1

# Go — e2e (requires models)
cd tests/e2e/go && LLAMA_CHAT_MODEL=~/llama-bindings-conf/models/chat/gemma-4-E2B-it-Q4_K_M.gguf \
  LLAMA_EMBED_MODEL=~/llama-bindings-conf/models/embeddings/nomic-embed-text-v1.5.Q4_K_M.gguf \
  go test -tags e2e -run TestE2EChatText -v -count=1

# Java — single test class
cd bindings/java && mvn -B test -Dtest=ChatEngineTest -Dbridge.lib.dir=../../bridge/build

# Java — e2e
cd bindings/java && mvn -B test -Pe2e -Dtest=E2ETest -Dbridge.lib.dir=../../bridge/build \
  -Djava.library.path=../../bridge/build \
  -DLLAMA_CHAT_MODEL=~/llama-bindings-conf/models/chat/gemma-4-E2B-it-Q4_K_M.gguf \
  -DLLAMA_EMBED_MODEL=~/llama-bindings-conf/models/embeddings/nomic-embed-text-v1.5.Q4_K_M.gguf

# JS — single test file
cd bindings/js-browser && npx jest test/chat.test.js
```

### CMake rebuild from scratch

If the CMake cache is stale (e.g. after moving directories or upgrading llama.cpp):

```bash
rm -rf bridge/build third_party/llama.cpp-src/build
task build-bridge
```

## Architecture

```
llama.cpp (upstream, cloned into third_party/llama.cpp-src/)
    ↑
bridge/  — C/C++ bridge layer, stable ABI (bridge/include/llama_bridge.h)
    ↑
┌───┼───────────┬──────────────────┐
Go (purego)   Java (Panama FFM)   JS (Emscripten WASM)
```

### Bridge layer (`bridge/`)

- `include/llama_bridge.h` — public C ABI. Six functions: create/infer/destroy for chat and embed, plus two free helpers.
- `src/llama_bridge.cpp` — engine create/destroy, llama backend init
- `src/llama_bridge_chat.cpp` — chat inference (JSON in → JSON out)
- `src/llama_bridge_embed.cpp` — embedding inference
- `src/llama_bridge_json.c` — lightweight JSON helpers
- `src/llama_bridge_observe.c` — event emission to callbacks

All requests/responses cross the bridge as JSON strings. This normalizes behavior across all languages.

### Go binding (`bindings/go/llama/`)

Uses **purego** (no CGO). `loader.go` searches for the native library in this order: embedded prebuilt → `LLAMA_BRIDGE_PATH` env → repo-relative paths → system library path. Prebuilt binaries for Darwin/Linux/Windows are in `bindings/go/llama/prebuilt/`.

### Java binding (`bindings/java/`)

Uses **Project Panama FFM** (JDK 21+, no JNI). `NativeLibrary.java` handles FFM loading. Maven builds require `-Dbridge.lib.dir=` pointing to the built bridge. Tests need `--enable-native-access=ALL-UNNAMED` (configured in pom.xml surefire plugin).

### JS/WASM binding (`bindings/js-browser/`)

Bridge compiled to WASM via Emscripten. `scripts/build.mjs` copies WASM artifacts from `wasm/build/` to `dist/`. Unit tests use Jest with a deterministic stub module (no WASM required). E2E tests use Playwright against Chromium.

## Test Structure

- **Unit tests** live alongside each binding (`bindings/{go,java,js-browser}/`)
- **E2E tests** live in `tests/e2e/{go,java,js-browser}/` — require real models and built bridge
- **Shared fixtures** in `tests/fixtures/` — JSON request templates, schemas, tool definitions
- Go e2e uses build tag `//go:build e2e`; Java e2e uses `@Tag("e2e")` + Maven profile `-Pe2e`

## Models

Models are stored outside the repo at `~/llama-bindings-conf/models/`. Two models:
- **Chat:** `chat/gemma-4-E2B-it-Q4_K_M.gguf` (~3.1 GB) — tool-calling capable
- **Embed:** `embeddings/nomic-embed-text-v1.5.Q4_K_M.gguf` (~80 MB)

Override with env vars: `LLAMA_CHAT_MODEL`, `LLAMA_EMBED_MODEL`.

## Key Constraints

- **Bridge is the only llama.cpp touchpoint.** Never call llama.cpp APIs from binding code. When upgrading llama.cpp, only `bridge/src/` files should need changes.
- **ChatEngine and EmbedEngine are independent.** They don't share state, models, or contexts.
- **Bridge source is C++** (`.cpp` files) despite the C ABI header. The `extern "C"` linkage ensures ABI stability.
- **Go binding has no CGO dependency.** The purego approach means `go build` works without a C compiler for consumers using prebuilt binaries.
- **Java requires JDK 21+** for Project Panama FFM. The pom.xml targets JDK 22.
