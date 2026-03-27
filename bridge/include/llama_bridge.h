#ifndef LLAMA_BRIDGE_H
#define LLAMA_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * llama_bridge.h — public stable API for the llama.cpp bridge.
 *
 * Language bindings (Go, Java, JS) depend only on this header.
 * They must never call raw llama.cpp APIs directly.
 */

typedef void* llama_engine_t;

/**
 * Create an engine by loading a GGUF model from the given path.
 * Returns NULL on failure.
 */
llama_engine_t llama_engine_create(const char* model_path);

/**
 * Run a raw completion for the given prompt (no chat template applied).
 * Returns a heap-allocated C string that the caller must free with
 * llama_engine_free_string(). Returns NULL on failure.
 */
char* llama_engine_complete(llama_engine_t engine, const char* prompt);

/**
 * Free a string returned by llama_engine_complete(), llama_engine_chat(),
 * or llama_engine_chat_with_object().
 */
void llama_engine_free_string(char* str);

/**
 * Destroy the engine, freeing all associated resources.
 */
void llama_engine_destroy(llama_engine_t engine);

/* =========================================================================
 * Chat API — two unified session-based methods
 * ====================================================================== */

/**
 * Session-based chat.
 *
 * session_id        — conversation identifier; created automatically on first call.
 * system_message    — optional; sets or replaces the session system prompt.
 *                     Pass NULL or "" to leave the existing system prompt unchanged.
 * user_message      — the user turn (main input for this call).
 * assistant_message — optional; inject a prior assistant turn into the session
 *                     before the user message (for few-shot context / correction).
 * tool_message      — optional; inject a tool-use response (role "tool") before
 *                     the user message.
 *
 * The model's built-in chat template is applied to the full session history.
 * The assistant response is appended to the session history automatically.
 *
 * Returns a heap-allocated JSON string:
 *   {"role":"assistant","content":"<text>"}
 * Free with llama_engine_free_string(). Returns NULL on failure.
 */
char* llama_engine_chat(
    llama_engine_t engine,
    const char*    session_id,
    const char*    system_message,
    const char*    user_message,
    const char*    assistant_message,
    const char*    tool_message
);

/**
 * Same as llama_engine_chat but returns a richer JSON schema response
 * that includes session metadata:
 *   {"role":"assistant","content":"<text>","sessionId":"<id>","messageCount":<n>}
 *
 * Free with llama_engine_free_string(). Returns NULL on failure.
 */
char* llama_engine_chat_with_object(
    llama_engine_t engine,
    const char*    session_id,
    const char*    system_message,
    const char*    user_message,
    const char*    assistant_message,
    const char*    tool_message
);

/**
 * Clear all history for the named session (including the system message).
 * The session slot is released and can be reused.
 */
void llama_engine_chat_session_clear(
    llama_engine_t engine,
    const char*    session_id
);

#ifdef __cplusplus
}
#endif

#endif /* LLAMA_BRIDGE_H */
