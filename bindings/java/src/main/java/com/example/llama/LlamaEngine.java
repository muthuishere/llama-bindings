package com.example.llama;

import com.sun.jna.Pointer;

/**
 * High-level Java wrapper around the llama_bridge C library.
 *
 * <p>Usage:
 * <pre>{@code
 * try (LlamaEngine engine = LlamaEngine.load("model.gguf")) {
 *     String out = engine.complete("Say hello.");
 *     System.out.println(out);
 * }
 * }</pre>
 *
 * <p>Instances are not thread-safe. Use one engine per thread or add
 * external synchronisation.
 */
public final class LlamaEngine implements AutoCloseable {

    private final LlamaLibrary lib;
    private Pointer handle;

    private LlamaEngine(LlamaLibrary lib, Pointer handle) {
        this.lib    = lib;
        this.handle = handle;
    }

    /**
     * Load a GGUF model and create an engine.
     *
     * @param modelPath path to the GGUF model file
     * @return a new {@link LlamaEngine}
     * @throws IllegalArgumentException if {@code modelPath} is null or blank
     * @throws RuntimeException         if the native engine cannot be created
     */
    public static LlamaEngine load(String modelPath) {
        if (modelPath == null || modelPath.isBlank()) {
            throw new IllegalArgumentException("modelPath must not be null or blank");
        }
        LlamaLibrary lib = LlamaLibrary.INSTANCE;
        Pointer handle = lib.llama_engine_create(modelPath);
        if (handle == null) {
            throw new RuntimeException(
                    "Failed to create llama engine for model: " + modelPath);
        }
        return new LlamaEngine(lib, handle);
    }

    /**
     * Run inference on the given prompt.
     *
     * @param prompt the input prompt (may be empty)
     * @return the generated completion text
     * @throws IllegalStateException if the engine has been closed
     * @throws RuntimeException      if inference fails
     */
    public String complete(String prompt) {
        ensureOpen();
        if (prompt == null) {
            prompt = "";
        }
        Pointer result = lib.llama_engine_complete(handle, prompt);
        if (result == null) {
            throw new RuntimeException("llama_engine_complete returned null");
        }
        try {
            return result.getString(0, "UTF-8");
        } finally {
            lib.llama_engine_free_string(result);
        }
    }

    /**
     * Close the engine and free all native resources.
     * Safe to call multiple times.
     */
    @Override
    public void close() {
        if (handle != null) {
            lib.llama_engine_destroy(handle);
            handle = null;
        }
    }

    private void ensureOpen() {
        if (handle == null) {
            throw new IllegalStateException("LlamaEngine has been closed");
        }
    }
}
