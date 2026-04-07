# Agent Export / Import

This document specifies the bundle format and API for exporting an Agent's
knowledge base to a portable ZIP file and restoring it in a fresh Agent.

Inspired by [offline-llm-knowledge-system](https://github.com/muthuishere/offline-llm-knowledge-system),
the goal is a self-contained, portable bundle that captures the entire RAG
state of an Agent so it can be transferred, archived, or deployed to an
air-gapped environment.

---

## Motivation

When you build an Agent and call `addDocument` many times, the embedding
model re-encodes every document and stores the resulting vectors in SQLite
(Go / Java) or in memory (JS-browser). That work is expensive. Export lets
you do it once and re-use the result.

Typical flow:

1. **Build phase** – run the embed model, call `addDocument` for every source
   text, then call `agent.export("bundle.zip")`.
2. **Deploy phase** – ship `bundle.zip` (and the model files) to the target
   machine / browser.
3. **Runtime phase** – call `Agent.importFrom("bundle.zip", ...)` which
   re-creates a fully-loaded Agent without touching the embed model.

---

## Bundle format

The ZIP archive contains exactly two files:

```
agent-bundle.zip
├── manifest.json
└── knowledge.json
```

### `manifest.json`

```json
{
  "version":       "1",
  "created_at":    "2026-04-07T17:00:00Z",
  "doc_count":     42,
  "embedding_dim": 384
}
```

| Field           | Type   | Description                                              |
|-----------------|--------|----------------------------------------------------------|
| `version`       | string | Bundle format version. Currently `"1"`.                  |
| `created_at`    | string | ISO-8601 UTC timestamp of when the bundle was created.   |
| `doc_count`     | int    | Number of documents stored in `knowledge.json`.          |
| `embedding_dim` | int    | Dimension of every embedding vector.                     |

### `knowledge.json`

An array of document objects, one per stored document, in insertion order:

```json
[
  {
    "text":      "The capital of France is Paris.",
    "embedding": [0.012, -0.334, 0.891, ...]
  }
]
```

| Field       | Type              | Description                                   |
|-------------|-------------------|-----------------------------------------------|
| `text`      | string            | Original document text passed to `addDocument`. |
| `embedding` | array of float    | Pre-computed embedding vector (float32 values). |

Embeddings are stored as plain JSON float arrays to maximise portability
across all three runtimes (Go, Java, JS-browser) without requiring binary
encoding.

> **Size note** – For large knowledge bases, consider compressing before
> transfer. ZIP is already applied at the archive level, which typically gives
> 40–60 % size reduction on JSON float arrays.

---

## API

### Go

```go
// Export writes the Agent's knowledge base to a ZIP bundle at zipPath.
// Only the knowledge store is exported; chat sessions are not included.
func (a *Agent) Export(zipPath string) error

// ImportFrom creates a new Agent by loading knowledge from a previously
// exported ZIP bundle. The chat and embed engines are initialised from the
// provided model paths. storagePath is forwarded to the KnowledgeStore
// (use ":memory:" for a temporary in-process store).
//
// The embed model is NOT used during import – embeddings are read directly
// from the bundle, so import is fast even for large knowledge bases.
func ImportFrom(zipPath, chatModelPath, embedModelPath, storagePath string) (*Agent, error)
```

**Example**

```go
// --- build phase ---
agent, _ := agent.New("chat.gguf", "embed.gguf", "knowledge.db")
agent.AddDocument("The capital of France is Paris.")
agent.Export("bundle.zip")
agent.Close()

// --- runtime phase ---
a, _ := agent.ImportFrom("bundle.zip", "chat.gguf", "embed.gguf", ":memory:")
defer a.Close()
reply, _ := a.Chat("s1", "What is the capital of France?")
```

---

### Java

```java
// Export the Agent's knowledge base to a ZIP bundle at zipPath.
// Only the knowledge store is exported; chat sessions are not included.
public void export(String zipPath) throws IOException

// Create an Agent by loading knowledge from a previously exported ZIP bundle.
// The embed model is NOT re-run during import; embeddings come from the bundle.
public static Agent importFrom(String zipPath,
                               String chatModelPath,
                               String embedModelPath,
                               String storagePath) throws IOException, LlamaException, SQLException
```

**Example**

```java
// --- build phase ---
try (Agent agent = Agent.create("chat.gguf", "embed.gguf", "knowledge.db")) {
    agent.addDocument("The capital of France is Paris.");
    agent.export("bundle.zip");
}

// --- runtime phase ---
try (Agent a = Agent.importFrom("bundle.zip", "chat.gguf", "embed.gguf", ":memory:")) {
    String reply = a.chat("s1", "What is the capital of France?");
}
```

---

### JS-browser

The browser has no filesystem, so export returns a `Blob` and importFrom
accepts any `Blob | File | ArrayBuffer | Uint8Array`.

```js
// Export the Agent's knowledge base as a ZIP Blob.
// Callers can save it with a <a download> link or store it in IndexedDB.
async export() → Promise<Blob>

// Create an Agent by loading knowledge from a ZIP bundle.
// zipData can be a Blob, File, ArrayBuffer, or Uint8Array.
// The embed model is NOT re-run; embeddings come from the bundle.
static async importFrom(
  zipData,          // Blob | File | ArrayBuffer | Uint8Array
  chatModelPath,    // string – URL or path to chat GGUF
  embedModelPath,   // string – URL or path to embed GGUF (loaded for future addDocument calls)
  opts = {}         // same opts as Agent.create (e.g. { onEvent })
) → Promise<Agent>
```

**Example**

```js
// --- build phase ---
const agent = await Agent.create('chat.gguf', 'embed.gguf');
await agent.addDocument('The capital of France is Paris.');
const blob = await agent.export();
// trigger browser download:
const url  = URL.createObjectURL(blob);
const link = document.createElement('a');
link.href = url; link.download = 'bundle.zip'; link.click();
agent.close();

// --- runtime phase (File from an <input type="file">) ---
const file = inputElement.files[0];
const a    = await Agent.importFrom(file, 'chat.gguf', 'embed.gguf');
const reply = await a.chat('s1', 'What is the capital of France?');
a.close();
```

---

## Implementation notes

### Export algorithm (all languages)

1. Query `SELECT id, text, embedding FROM documents ORDER BY id` from the
   KnowledgeStore.
2. For each row, decode the stored binary embedding to a `float[]` and
   append `{text, embedding: [...]}` to an array.
3. Serialise the array to `knowledge.json`.
4. Build `manifest.json` with `doc_count` and `embedding_dim` derived from
   the data.
5. Write both files into a ZIP archive at the target path (or as a `Blob`
   in the browser).

### Import algorithm (all languages)

1. Open the ZIP and read `manifest.json` to validate `version`.
2. Read and parse `knowledge.json`.
3. Create a new Agent via the normal factory (`New` / `create`) – this loads
   the chat and embed models.
4. For each document entry, call `store.Add(text, embedding)` **directly**
   (bypassing the embed model) to populate the KnowledgeStore.
5. Return the ready Agent.

> **No re-embedding** – The embed model is still loaded (to support future
> `addDocument` calls), but it is never called during import. All vectors are
> read straight from the bundle.

### ZIP library choices

| Language | Library |
|----------|---------|
| Go       | `archive/zip` (stdlib) |
| Java     | `java.util.zip` (stdlib) |
| JS-browser | [`fflate`](https://github.com/101arrowz/fflate) (small, no WASM, already used by llama.cpp WASM builds) |

### Versioning

`manifest.version` is a simple integer string. If breaking changes are
needed in future, increment the version and reject older importers with a
clear error message.

---

## What is NOT exported

| Item | Reason |
|------|--------|
| Chat session history | Sessions are ephemeral and user-specific. |
| Registered tools | Tools are code; they must be re-registered by the caller. |
| Model files | Too large; caller must supply them at import time. |
| Agent configuration | No persistent config exists today. |
