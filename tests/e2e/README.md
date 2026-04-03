# End-to-End Tests

Full-stack e2e tests that run real inference against a
[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF)
GGUF model across all three language targets.

## Prerequisites

1. Native bridge built: `task build-bridge`
2. Model downloaded: `task download-model`

## Run all e2e tests

```sh
task e2e
```

## Per-target

| Target | Command | Notes |
|--------|---------|-------|
| Go | `task e2e-go` | Build tag `e2e`; skips if bridge/model absent |
| Java | `task e2e-java` | Maven profile `e2e`; skips if bridge/model absent |
| Browser JS | `task e2e-js` | Playwright + Chromium; stub mode if WASM not built |

## Model

| Field | Value |
|-------|-------|
| Model | Qwen2.5-0.5B-Instruct |
| Format | GGUF Q4_K_M quantisation (~400 MB) |
| Source | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF |
| Path | `tests/models/qwen2.5-0.5b-instruct-q4_k_m.gguf` |

Override model paths with environment variables:
```sh
LLAMA_CHAT_MODEL=/path/to/chat.gguf  task e2e-go
LLAMA_EMBED_MODEL=/path/to/embed.gguf task e2e-go
```

## JavaScript / Browser (Playwright)

Tests live in `tests/e2e/js-browser/specs/`.
They open real Chromium, serve the project files, navigate to
`tests/e2e/js-browser/app/index.html`, and verify results via
`page.evaluate()`.

Without a built WASM module the stub module is used — all API contracts
and lifecycle rules are exercised with deterministic responses.

```sh
# Run just the browser tests
task e2e-js

# Interactive / headed mode (see the browser)
cd tests/e2e/js-browser && npx playwright test --headed
```

## Go e2e

Tests use build tag `//go:build e2e` so they are never included in normal
`task test-go` runs.

```sh
task e2e-go
```

## Java e2e

Tests use `@Tag("e2e")` and are excluded from the default Surefire run.
The `e2e` Maven profile re-includes them.

```sh
task e2e-java
# or directly:
cd bindings/java && mvn test -Pe2e \
    -Dbridge.lib.dir=../../bridge/build \
    -DLLAMA_CHAT_MODEL=../../tests/models/qwen2.5-0.5b-instruct-q4_k_m.gguf
```
