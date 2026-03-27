# llama-bindings

A tiny, stable cross-language bridge to [llama.cpp](https://github.com/ggerganov/llama.cpp).

Load a GGUF model, send a prompt, get a completion — in Go, Java, or JavaScript —
with the inference logic living only once (in a C bridge).

---

## Architecture

```
llama.cpp upstream
      ↓
C bridge  (bridge/src/llama_bridge.c)   ← only this file knows llama.cpp
      ↓
Go binding | Java binding | JS binding
```

---

## Quick Start

```bash
# Requires: task, cmake, go, java 17, maven, node 18, npm

task init          # clone llama.cpp
task build-llama   # build llama.cpp
task build-bridge  # build C bridge

LLAMA_TEST_MODEL=/path/to/model.gguf task test   # run all tests
```

---

## Documentation

- [docs/design.md](docs/design.md) — architecture and design decisions
- [docs/build.md](docs/build.md)   — detailed build instructions
- [docs/upgrade.md](docs/upgrade.md) — how to update to a newer llama.cpp

---

## Project Structure

```
bridge/include/llama_bridge.h     public stable C API
bridge/src/llama_bridge.c         implementation (only file that imports llama.cpp)
bindings/go/llama/                Go CGO binding
bindings/java/                    Java JNA binding
bindings/js/                      Node.js ffi-napi binding
tests/prompts/                    shared prompt fixtures
docs/                             documentation
scripts/verify-artifacts.sh       build artifact verification
Taskfile.yml                      build / update / test automation
```