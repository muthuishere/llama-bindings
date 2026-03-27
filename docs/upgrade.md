# Upgrade Guide

This document explains how to update to a newer version of llama.cpp.

---

## When to upgrade

- You want access to new model architectures or GGUF features
- A security advisory affects llama.cpp
- Performance improvements are available

---

## The upgrade contract

**Only `bridge/src/llama_bridge.c` should need editing after an upstream upgrade.**

Go, Java, and JavaScript bindings depend solely on `bridge/include/llama_bridge.h`,
which defines three stable functions:

```c
llama_engine_t llama_engine_create(const char* model_path);
char*          llama_engine_complete(llama_engine_t engine, const char* prompt);
void           llama_engine_destroy(llama_engine_t engine);
```

As long as these three functions continue to work, no binding code changes.

---

## Upgrade procedure

### Automated (recommended)

```bash
task upgrade
```

This runs `update → build-llama → build-bridge → test` in sequence and
tells you if any binding breaks.

### Manual

```bash
# 1. Pull the latest llama.cpp
task update

# 2. Rebuild llama.cpp
task build-llama

# 3. Try to rebuild the bridge
task build-bridge
```

If step 3 fails, open `bridge/src/llama_bridge.c` and fix the compilation
errors caused by the upstream API change.  The internal header
(`bridge/src/llama_bridge_internal.h`) may also need updating if
`llama_model` or `llama_context` struct names changed.

```bash
# 4. Run all tests
task test

# 5. Optionally run with a real model
LLAMA_TEST_MODEL=/path/to/model.gguf task test
```

---

## Common breaking-change patterns

| Upstream change                            | What to fix in llama_bridge.c              |
|--------------------------------------------|--------------------------------------------|
| Function renamed                           | Update the call site in `llama_bridge.c`   |
| Parameter type / order changed             | Update the call signature                  |
| New required initialisation step           | Add it to `llama_engine_create`            |
| Sampling API restructured                  | Update the generation loop                 |
| `llama_batch` fields changed               | Update batch construction code             |

---

## After upgrading

1. Commit the updated `third_party/llama.cpp` submodule reference (if using submodules)
   or the Taskfile-pinned revision.
2. If `llama_bridge.c` needed changes, commit those too.
3. Tag the release so you can roll back if needed.

---

## Rolling back

```bash
cd third_party/llama.cpp
git checkout <previous-commit-sha>
cd ../..
task build-llama
task build-bridge
task test
```
