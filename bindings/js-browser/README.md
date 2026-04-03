# @llama-bindings/js-browser

Browser JavaScript / WebAssembly binding for [llama.cpp](https://github.com/ggml-org/llama.cpp).

**Runtime:** Browser JS → WASM (Emscripten) → llama.cpp

> This package targets the **browser**. It does not use Node.js native addons.

## Installation

```sh
npm install @llama-bindings/js-browser
```

## Prerequisites

- Modern browser with WebAssembly support
- WASM build artefacts in `dist/` (built by `task build-wasm`)

## Quick start

### Chat

```js
import { LlamaChat } from '@llama-bindings/js-browser';

const chat = await LlamaChat.load('chat-model.gguf', {
  onEvent(event) { console.log(event); }
});

try {
  const resp = await chat.chat({
    messages: [
      { role: 'system', content: 'You are helpful.' },
      { role: 'user',   content: 'Extract person data.' }
    ],
    responseMode: 'json_schema',
    schema: {
      name: 'person_extract',
      schema: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          age:  { type: 'integer' }
        },
        required: ['name', 'age'],
        additionalProperties: false
      }
    }
  }, {
    temperature: 0.0,
    maxOutputTokens: 128
  });

  console.log(resp.type);  // 'structured_json'
  console.log(resp.json);  // { name: '...', age: ... }
} finally {
  chat.close();
}
```

### Embed

```js
import { LlamaEmbed } from '@llama-bindings/js-browser';

const embed = await LlamaEmbed.load('embed-model.gguf');
try {
  const vec = await embed.embed('hello world');
  // vec is a Float32Array
  console.log('Vector length:', vec.length);
} finally {
  embed.close();
}
```

## API reference

| Symbol | Description |
|--------|-------------|
| `LlamaChat.load(path, opts)` | Load a chat engine |
| `chat.chat(request, opts)` | Run chat inference |
| `chat.close()` | Release engine |
| `LlamaEmbed.load(path, opts)` | Load an embedding engine |
| `embed.embed(text, opts)` | Generate embedding vector (`Float32Array`) |
| `embed.close()` | Release engine |
| `LlamaError` | Error class with `.code` property |

## Tests

```sh
task test-js
# or
cd bindings/js-browser && npm test
```

## Build

```sh
task build-wasm   # compile WASM from https://github.com/ggml-org/llama.cpp
task build-js     # copy artefacts to dist/
```

Upstream llama.cpp: https://github.com/ggml-org/llama.cpp
