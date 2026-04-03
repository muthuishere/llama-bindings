# Integration Tests

Cross-language parity tests that exercise Go, Java, and browser JS against the
same shared fixtures in `tests/fixtures/`.

## Goal

Verify that the same conceptual request produces conceptually consistent
responses across all three language targets, and that error behaviour
is consistent.

## Running

```sh
task test          # all targets
task test-go
task test-java
task test-js
```

## Model files

Integration tests that need a real model file look in `tests/models/`:

```
tests/models/
  chat-model.gguf      ← any compatible GGUF chat model
  embed-model.gguf     ← any compatible GGUF embedding model
```

Tests skip gracefully when model files are absent (CI without a model).

## Fixtures

Shared request/response fixtures are in `tests/fixtures/`. All language
targets use the same JSON fixtures to ensure parity.

## Upstream

llama.cpp: https://github.com/ggml-org/llama.cpp
