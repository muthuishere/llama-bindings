package com.example.llama;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

/**
 * Low-level JNA interface to {@code libllama_bridge}.
 *
 * <p>Language callers must not use this class directly.
 * Use {@link LlamaEngine} instead.
 */
interface LlamaLibrary extends Library {

    /**
     * Shared singleton instance loaded from {@code llama_bridge}.
     */
    LlamaLibrary INSTANCE = Native.load("llama_bridge", LlamaLibrary.class);

    /**
     * Create an engine from a GGUF model file.
     *
     * @param modelPath absolute or relative path to the model file
     * @return opaque engine pointer, or {@code null} on failure
     */
    Pointer llama_engine_create(String modelPath);

    /**
     * Run inference and return a completion string.
     *
     * @param engine engine pointer returned by {@link #llama_engine_create}
     * @param prompt prompt text (may be empty but not null)
     * @return heap-allocated completion string, or {@code null} on failure;
     *         must be freed with {@link #llama_engine_free_string}
     */
    Pointer llama_engine_complete(Pointer engine, String prompt);

    /**
     * Free a string returned by {@link #llama_engine_complete}.
     *
     * @param str pointer to the string to free
     */
    void llama_engine_free_string(Pointer str);

    /**
     * Destroy the engine and release all resources.
     *
     * @param engine engine pointer to destroy
     */
    void llama_engine_destroy(Pointer engine);
}
