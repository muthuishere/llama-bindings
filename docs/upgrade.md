# Upgrading llama.cpp

Upstream repository: **https://github.com/ggml-org/llama.cpp**

## Standard upgrade flow

```sh
task upgrade
```

This is equivalent to:
```sh
task update        # git pull from https://github.com/ggml-org/llama.cpp
task build-bridge  # rebuild the native bridge against the new source
```

Then run the full test suite:
```sh
task test
```

## Where breakage can occur

| Layer | Breakage likelihood | Action |
|-------|---------------------|--------|
| `bridge/src/llama_bridge*.c` | High — implementation files that call llama.cpp | Fix here |
| `bridge/include/llama_bridge.h` | Very low — stable ABI | Rare, deliberate |
| `bindings/go/llama/*.go` | Very low | Only if ABI changes |
| `bindings/java/...` | Very low | Only if ABI changes |
| `bindings/js-browser/src/*.js` | Very low | Only if ABI changes |

## Isolation guarantee

Only the bridge implementation files (`bridge/src/`) call llama.cpp APIs.  
No language binding imports llama.cpp headers or links llama.cpp directly.

This means: **upgrade llama.cpp → fix bridge implementation if needed →
language bindings stay unchanged.**

## What to check after an upgrade

1. `bridge/src/llama_bridge.c` — engine create/destroy
2. `bridge/src/llama_bridge_chat.c` — chat inference
3. `bridge/src/llama_bridge_embed.c` — embedding inference
4. llama.cpp API changelog: https://github.com/ggml-org/llama.cpp/releases

## Pinning a specific version

To pin to a specific llama.cpp commit or tag:

```sh
cd third_party/llama.cpp-src
git fetch origin
git checkout <tag-or-sha>
cd ../..
task build-bridge
```

Then commit the pinned SHA to the repo (e.g. in a `LLAMA_CPP_VERSION` file).
