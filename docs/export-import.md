# Agent Export / Import

This document specifies the bundle format and API for exporting a complete
Agent snapshot to a portable ZIP file and restoring it in a fresh Agent.

Inspired by [offline-llm-knowledge-system](https://github.com/muthuishere/offline-llm-knowledge-system),
the goal is a fully self-contained bundle that captures the entire state of an
Agent — model references, all embeddings, and the full knowledge store — so it
can be transferred, archived, or deployed to an air-gapped environment.

---

## Motivation

When you build an Agent and call `addDocument` many times, the embedding
model re-encodes every document and stores the resulting vectors in SQLite
(Go / Java) or in memory (JS-browser). That work is expensive. Export lets
you do it once and re-use the result.

Typical flow:

1. **Build phase** – run the embed model, call `addDocument` for every source
   text, then call `agent.export("bundle.zip")`.
2. **Deploy phase** – ship `bundle.zip` and the model GGUF files to the target
   machine / browser.
3. **Runtime phase** – call `Agent.importFrom("bundle.zip", ...)` which
   re-creates a fully-loaded Agent without re-running the embed model.

---

## Default SQLite storage path

When no explicit `storagePath` is provided, the Agent defaults to
`agent.db` in the current working directory:

| Language | Default path |
|----------|-------------|
| Go       | `"agent.db"` |
| Java     | `"agent.db"` |
| JS-browser | `":memory:"` (browser has no persistent filesystem by default) |

Callers may still pass an explicit path or `":memory:"` to override the
default.

---

## Bundle format

The ZIP archive contains exactly three files:

```
agent-bundle.zip
├── manifest.json
├── knowledge.db
└── knowledge.json
```

| File             | Description |
|------------------|-------------|
| `manifest.json`  | Bundle metadata including model information. |
| `knowledge.db`   | Full copy of the SQLite knowledge store (Go / Java). |
| `knowledge.json` | JSON snapshot of all documents + embeddings (all runtimes, including JS-browser). |

Both `knowledge.db` and `knowledge.json` are always written during export so
that the same bundle is usable by every runtime. On import, each runtime
prefers the format best suited to it:
- **Go / Java** restore from `knowledge.db` (fast, byte-for-byte copy).
- **JS-browser** restores from `knowledge.json` (no SQLite file support in
  the browser).

### `manifest.json`

```json
{
  "version":          "1",
  "created_at":       "2026-04-07T17:00:00Z",
  "chat_model":       "llama-3.2-3b-instruct-q4_k_m.gguf",
  "embed_model":      "nomic-embed-text-v1.5.q4_k_m.gguf",
  "storage_path":     "agent.db",
  "doc_count":        42,
  "embedding_dim":    384
}
```

| Field            | Type   | Description                                                     |
|------------------|--------|-----------------------------------------------------------------|
| `version`        | string | Bundle format version. Currently `"1"`.                         |
| `created_at`     | string | ISO-8601 UTC timestamp of when the bundle was created.          |
| `chat_model`     | string | Filename (basename) of the chat GGUF model used by this Agent.  |
| `embed_model`    | string | Filename (basename) of the embed GGUF model used by this Agent. |
| `storage_path`   | string | SQLite path the Agent was using at export time.                 |
| `doc_count`      | int    | Number of documents in the knowledge store.                     |
| `embedding_dim`  | int    | Dimension of every embedding vector.                            |

Only the **filename** (not the full path) of each model is recorded. The
caller supplies the actual model file paths at import time so the bundle
remains portable across machines.

### `knowledge.db`

A byte-for-byte copy of the Agent's SQLite database file (the same schema
created by `KnowledgeStore`). This is used by Go and Java importers for a
fast, lossless restore.

For in-memory stores (`":memory:"`), the database is serialised to bytes
before being written into the ZIP.

### `knowledge.json`

A JSON array of all stored documents with their embedding vectors, in
insertion order:

```json
[
  {
    "text":      "The capital of France is Paris.",
    "embedding": [0.012, -0.334, 0.891]
  }
]
```

| Field       | Type           | Description                                     |
|-------------|----------------|-------------------------------------------------|
| `text`      | string         | Original document text passed to `addDocument`. |
| `embedding` | array of float | Pre-computed embedding vector (float32 values). |

This file is used by the JS-browser importer and serves as a
human-readable, portable fallback for any other runtime.

---

## API

### Go

```go
// New creates an Agent with the default storage path ("agent.db").
// Use NewWithStorage to specify a custom path.
func New(chatModelPath, embedModelPath string) (*Agent, error)

// NewWithStorage creates an Agent with an explicit SQLite storage path.
// Use ":memory:" for an ephemeral in-process store.
func NewWithStorage(chatModelPath, embedModelPath, storagePath string) (*Agent, error)

// Export writes a complete Agent bundle (manifest, full SQLite DB, and JSON
// snapshot) to a ZIP archive at zipPath.
func (a *Agent) Export(zipPath string) error

// ImportFrom restores an Agent from a previously exported ZIP bundle.
// chatModelPath and embedModelPath must point to the GGUF model files on
// disk (their basenames should match manifest.chat_model / manifest.embed_model).
// storagePath is where the restored SQLite DB will be written; if empty,
// the default "agent.db" is used.
func ImportFrom(zipPath, chatModelPath, embedModelPath, storagePath string) (*Agent, error)
```

**Example**

```go
// --- build phase ---
a, _ := agent.New("chat.gguf", "embed.gguf")           // uses agent.db by default
a.AddDocument("The capital of France is Paris.")
a.Export("bundle.zip")
a.Close()

// --- runtime phase ---
b, _ := agent.ImportFrom("bundle.zip", "chat.gguf", "embed.gguf", "")
defer b.Close()
reply, _ := b.Chat("s1", "What is the capital of France?")
```

---

### Java

```java
// Create an Agent with the default storage path ("agent.db").
public static Agent create(String chatModelPath, String embedModelPath)
        throws LlamaException, SQLException

// Create an Agent with an explicit SQLite storage path.
public static Agent create(String chatModelPath, String embedModelPath, String storagePath)
        throws LlamaException, SQLException

// Export a complete Agent bundle (manifest, full SQLite DB, and JSON snapshot)
// to a ZIP archive at zipPath.
public void export(String zipPath) throws IOException

// Restore an Agent from a previously exported ZIP bundle.
// If storagePath is null or empty, "agent.db" is used as the default.
public static Agent importFrom(String zipPath,
                               String chatModelPath,
                               String embedModelPath,
                               String storagePath) throws IOException, LlamaException, SQLException
```

**Example**

```java
// --- build phase ---
try (Agent a = Agent.create("chat.gguf", "embed.gguf")) {  // uses agent.db by default
    a.addDocument("The capital of France is Paris.");
    a.export("bundle.zip");
}

// --- runtime phase ---
try (Agent b = Agent.importFrom("bundle.zip", "chat.gguf", "embed.gguf", null)) {
    String reply = b.chat("s1", "What is the capital of France?");
}
```

---

### JS-browser

The browser has no persistent filesystem, so:
- `export()` returns a `Blob` (save with a `<a download>` link or IndexedDB).
- `importFrom()` accepts any `Blob | File | ArrayBuffer | Uint8Array`.
- The default storage is always `":memory:"` (in-browser IndexedDB support
  may be added in a future version).

```js
// Export a complete Agent bundle as a ZIP Blob.
// The Blob contains manifest.json, knowledge.db (serialised), and knowledge.json.
async export() → Promise<Blob>

// Restore an Agent from a ZIP bundle.
// zipData: Blob | File | ArrayBuffer | Uint8Array
// If chatModelPath / embedModelPath are omitted, values from manifest.json are
// used as hints (the caller still needs to supply actual URLs/paths).
static async importFrom(
  zipData,          // Blob | File | ArrayBuffer | Uint8Array
  chatModelPath,    // string – URL or path to chat GGUF
  embedModelPath,   // string – URL or path to embed GGUF
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

// --- runtime phase (File picked from <input type="file">) ---
const file = inputElement.files[0];
const b    = await Agent.importFrom(file, 'chat.gguf', 'embed.gguf');
const reply = await b.chat('s1', 'What is the capital of France?');
b.close();
```

---

## Implementation notes

### Export algorithm (Go / Java)

1. Flush / checkpoint the SQLite WAL so the DB file is consistent.
2. Read the raw SQLite database bytes (copy the file, or use the
   `sqlite3_serialize` / JDBC backup API for in-memory databases).
3. Query `SELECT id, text, embedding FROM documents ORDER BY id` and build
   the `knowledge.json` array (decode binary blobs to float arrays).
4. Build `manifest.json` with model basenames, storage path, doc count, and
   embedding dim.
5. Write all three files into a ZIP archive at `zipPath`.

### Export algorithm (JS-browser)

1. Serialise the in-memory `#docs` array to `knowledge.json`.
2. Write a minimal `knowledge.db` placeholder (empty file or stub) so the
   bundle is structurally complete.
3. Build `manifest.json`.
4. Compress all three files with `fflate` and return a `Blob`.

### Import algorithm (Go / Java)

1. Open the ZIP and parse `manifest.json`; reject bundles with unknown
   `version`.
2. Extract `knowledge.db` to `storagePath` (default `"agent.db"` if empty).
3. Load the chat and embed engines from the provided model paths.
4. Open the KnowledgeStore against the restored SQLite file.
5. Return the ready Agent.

### Import algorithm (JS-browser)

1. Decompress the ZIP with `fflate`.
2. Parse `manifest.json`; validate `version`.
3. Parse `knowledge.json`.
4. Create a new Agent via `Agent.create` (loads both model engines).
5. For each document entry, call `store.add(text, embedding)` directly
   (bypassing the embed model).
6. Return the ready Agent.

### ZIP library choices

| Language   | Library |
|------------|---------|
| Go         | `archive/zip` (stdlib) |
| Java       | `java.util.zip` (stdlib) |
| JS-browser | [`fflate`](https://github.com/101arrowz/fflate) (small, no WASM) |

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
| Model files (GGUF) | Too large; caller must supply them at import time. Only the filename is recorded in `manifest.json` as a hint. |
