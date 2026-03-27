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
 * Run a completion for the given prompt using the engine.
 * Returns a heap-allocated C string that the caller must free with
 * llama_engine_free_string(). Returns NULL on failure.
 */
char* llama_engine_complete(llama_engine_t engine, const char* prompt);

/**
 * Free a string returned by llama_engine_complete().
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
 * One-shot chat with an optional system message and a user message.
 *
 * The model's built-in chat template is applied automatically.
 * system_msg may be NULL or "" to omit.
 * Returns a heap-allocated completion string (free with llama_engine_free_string).
 */
char* llama_engine_chat(
    llama_engine_t engine,
    const char*    system_msg,
    const char*    user_msg
);

/**
 * Chat with an explicit message array (chatWithObject equivalent).
 *
 * roles[i]:    "system" | "user" | "assistant"
 * contents[i]: message text
 * n_messages:  array length
 *
 * Returns a heap-allocated completion string (free with llama_engine_free_string).
 */
char* llama_engine_chat_with_messages(
    llama_engine_t  engine,
    const char**    roles,
    const char**    contents,
    int             n_messages
);

/**
 * Session-based multi-turn chat.
 *
 * Conversation history is maintained inside the engine, keyed by session_id.
 * Appends user_msg to the session and returns the next assistant turn.
 * The assistant response is automatically appended to the session history.
 *
 * Returns a heap-allocated string (free with llama_engine_free_string).
 */
char* llama_engine_chat_session(
    llama_engine_t engine,
    const char*    session_id,
    const char*    user_msg
);

/**
 * Set (or replace) the system message for a named session.
 * Call this before the first llama_engine_chat_session call if a system
 * prompt is required.  Pass NULL or "" to clear the system message.
 */
void llama_engine_chat_session_set_system(
    llama_engine_t engine,
    const char*    session_id,
    const char*    system_msg
);

/**
 * Clear all history for a session (including the system message).
 * The session slot is released and can be reused.
 */
void llama_engine_chat_session_clear(
    llama_engine_t engine,
    const char*    session_id
);

/**
 * Chat with tool definitions.
 *
 * roles / contents / n_messages: message array (same as chat_with_messages).
 * tools_json: JSON string describing available tools (OpenAI-compatible format,
 *   e.g. [{"name":"...", "description":"...", "parameters":{...}}]).
 *
 * The tools block is injected into the system message so that any model can
 * reason about tools.  Raw model output is returned — the model's response
 * may include a tool-call JSON object; the caller is responsible for
 * interpreting and executing it.
 *
 * Returns a heap-allocated string (free with llama_engine_free_string).
 */
char* llama_engine_chat_with_tools(
    llama_engine_t engine,
    const char**   roles,
    const char**   contents,
    int            n_messages,
    const char*    tools_json
);

#ifdef __cplusplus
}
#endif

#endif /* LLAMA_BRIDGE_H */
