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

    /** Shared singleton instance loaded from {@code llama_bridge}. */
    LlamaLibrary INSTANCE = Native.load("llama_bridge", LlamaLibrary.class);

    /* ---- core ---- */
    Pointer llama_engine_create(String modelPath);
    Pointer llama_engine_complete(Pointer engine, String prompt);
    void    llama_engine_free_string(Pointer str);
    void    llama_engine_destroy(Pointer engine);

    /* ---- chat (array-based — primary interface) ---- */

    /**
     * Session-based chat accepting messages as a JSON array.
     * Returns JSON: {@code {"role":"assistant","content":"..."}}
     *
     * @param sessionId    conversation identifier
     * @param messagesJson JSON array of {role, content} objects
     */
    Pointer llama_engine_chat_messages(
            Pointer engine,
            String  sessionId,
            String  messagesJson);

    /**
     * Session-based chat with richer schema response.
     * Returns JSON: {@code {"role":"assistant","content":"...","sessionId":"...","messageCount":N}}
     */
    Pointer llama_engine_chat_with_object_messages(
            Pointer engine,
            String  sessionId,
            String  messagesJson);

    /* ---- legacy chat helpers (kept for direct C callers) ---- */

    Pointer llama_engine_chat(
            Pointer engine,
            String  sessionId,
            String  systemMessage,
            String  userMessage,
            String  assistantMessage,
            String  toolMessage);

    Pointer llama_engine_chat_with_object(
            Pointer engine,
            String  sessionId,
            String  systemMessage,
            String  userMessage,
            String  assistantMessage,
            String  toolMessage);

    /** Clear all history for a session. */
    void llama_engine_chat_session_clear(Pointer engine, String sessionId);
}
