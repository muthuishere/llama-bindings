# wasm/build/

WebAssembly build output directory.

Populated by `task build-wasm`, which compiles the bridge + llama.cpp
([https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp))
to WebAssembly via Emscripten.

Generated files:
- `llama_bridge.js`   — Emscripten JS wrapper (copied to `bindings/js-browser/dist/`)
- `llama_bridge.wasm` — compiled WASM binary (copied to `bindings/js-browser/dist/`)

These files are not committed to the repository.
