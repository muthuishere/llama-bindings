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

    /* ---- v1: simple completion ---- */

    Pointer llama_engine_create(String modelPath);
    Pointer llama_engine_complete(Pointer engine, String prompt);
    void    llama_engine_free_string(Pointer str);
    void    llama_engine_destroy(Pointer engine);

    /* ---- v2: chat ---- */

    /**
     * One-shot chat with optional system message and user message.
     * systemMsg may be null or empty to omit.
     */
    Pointer llama_engine_chat(Pointer engine, String systemMsg, String userMsg);

    /**
     * Chat with an explicit message array.
     *
     * @param roles     array of role strings ("system"|"user"|"assistant")
     * @param contents  array of message content strings
     * @param nMessages number of messages
     */
    Pointer llama_engine_chat_with_messages(
            Pointer engine, String[] roles, String[] contents, int nMessages);

    /**
     * Session-based multi-turn chat.
     *
     * @param sessionId session identifier
     * @param userMsg   the user turn to append
     */
    Pointer llama_engine_chat_session(
            Pointer engine, String sessionId, String userMsg);

    /**
     * Set (or replace) the system message for a session.
     *
     * @param sessionId session identifier
     * @param systemMsg system message text (null or "" to clear)
     */
    void llama_engine_chat_session_set_system(
            Pointer engine, String sessionId, String systemMsg);

    /**
     * Clear all history for a session.
     *
     * @param sessionId session identifier
     */
    void llama_engine_chat_session_clear(Pointer engine, String sessionId);

    /**
     * Chat with tool definitions.
     *
     * @param roles      array of role strings
     * @param contents   array of message content strings
     * @param nMessages  number of messages
     * @param toolsJson  JSON array of tool definitions (OpenAI-compatible format)
     */
    Pointer llama_engine_chat_with_tools(
            Pointer engine,
            String[] roles,
            String[] contents,
            int      nMessages,
            String   toolsJson);
}

