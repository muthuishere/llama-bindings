# third_party/llama.cpp

This directory is a tracked placeholder so the repository can document the
upstream dependency without trying to commit the full source tree.

`task init` clones the real upstream checkout into `third_party/llama.cpp-src/`
(and `task update` refreshes that checkout).

Upstream repository: **https://github.com/ggml-org/llama.cpp**

```sh
# Clone (first time)
task init

# Pull latest
task update
```

**Do not commit the llama.cpp source tree.**  
Only this placeholder README is tracked in the repository.
