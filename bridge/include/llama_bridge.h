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
 * Chat API
 * ====================================================================== */

/**
 * Session-based chat — accepts messages as a JSON array.
 *
 * session_id    — conversation identifier; created automatically on first call.
 * messages_json — JSON array of {role, content} objects to append to the
 *                 session for this turn.  Example:
 *                   [{"role":"system","content":"You are helpful."},
 *                    {"role":"user","content":"Hello!"}]
 *                 Supported roles: "system", "user", "assistant", "tool".
 *                 A "system" entry sets or replaces the session system prompt.
 *
 * Messages are appended to the session in array order.  The model's built-in
 * chat template is applied to the full session history.  The assistant
 * response is appended to the session history automatically.
 *
 * Returns a heap-allocated JSON string:
 *   {"role":"assistant","content":"<text>"}
 * Free with llama_engine_free_string().  Returns NULL on failure.
 */
char* llama_engine_chat_messages(
    llama_engine_t engine,
    const char*    session_id,
    const char*    messages_json
);

/**
 * Same as llama_engine_chat_messages but returns a richer JSON response
 * that includes session metadata:
 *   {"role":"assistant","content":"<text>","sessionId":"<id>","messageCount":<n>}
 *
 * Free with llama_engine_free_string().  Returns NULL on failure.
 */
char* llama_engine_chat_with_object_messages(
    llama_engine_t engine,
    const char*    session_id,
    const char*    messages_json
);

/**
 * Clear all history for the named session (including the system message).
 * The session slot is released and can be reused.
 */
void llama_engine_chat_session_clear(
    llama_engine_t engine,
    const char*    session_id
);

/* =========================================================================
 * Legacy chat helpers (kept for direct C callers; bindings use the
 * _messages variants above).
 * ====================================================================== */

char* llama_engine_chat(
    llama_engine_t engine,
    const char*    session_id,
    const char*    system_message,
    const char*    user_message,
    const char*    assistant_message,
    const char*    tool_message
);

char* llama_engine_chat_with_object(
    llama_engine_t engine,
    const char*    session_id,
    const char*    system_message,
    const char*    user_message,
    const char*    assistant_message,
    const char*    tool_message
);

#ifdef __cplusplus
}
#endif

#endif /* LLAMA_BRIDGE_H */
