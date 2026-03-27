package com.example.llama;

import com.sun.jna.Pointer;

import java.util.List;

/**
 * High-level Java wrapper around the llama_bridge C library.
 *
 * <p>Usage:
 * <pre>{@code
 * try (LlamaEngine engine = LlamaEngine.load("model.gguf")) {
 *
 *     // Simple completion
 *     String out = engine.complete("Say hello.");
 *
 *     // One-shot chat with system + user message
 *     String chat = engine.chat("You are helpful.", "What is 2+2?");
 *
 *     // Chat with structured message list (chatWithObject)
 *     List<LlamaEngine.Message> msgs = List.of(
 *         LlamaEngine.Message.system("You are helpful."),
 *         LlamaEngine.Message.user("What is 2+2?")
 *     );
 *     String chatObj = engine.chatWithMessages(msgs);
 *
 *     // Session-based multi-turn chat
 *     String turn1 = engine.chatSession("sid-1", "Hello!");
 *     String turn2 = engine.chatSession("sid-1", "What did I just say?");
 *
 *     // Chat with tool definitions (raw output returned)
 *     String tools = "[{\"name\":\"add\",\"description\":\"Add numbers\",\"parameters\":{}}]";
 *     String toolResp = engine.chatWithTools(msgs, tools);
 * }
 * }</pre>
 *
 * <p>Instances are not thread-safe. Use one engine per thread or add
 * external synchronisation.
 */
public final class LlamaEngine implements AutoCloseable {

    /**
     * A single chat message carrying a role and content.
     *
     * <p>Role must be one of {@code "system"}, {@code "user"}, or
     * {@code "assistant"}.
     */
    public static final class Message {
        public final String role;
        public final String content;

        public Message(String role, String content) {
            if (role == null || role.isBlank()) {
                throw new IllegalArgumentException("role must not be null or blank");
            }
            this.role    = role;
            this.content = content != null ? content : "";
        }

        /** Convenience factory for a {@code system} message. */
        public static Message system(String content)    { return new Message("system",    content); }
        /** Convenience factory for a {@code user} message. */
        public static Message user(String content)      { return new Message("user",      content); }
        /** Convenience factory for an {@code assistant} message. */
        public static Message assistant(String content) { return new Message("assistant", content); }
    }

    // -----------------------------------------------------------------------

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
        LlamaLibrary lib    = LlamaLibrary.INSTANCE;
        Pointer      handle = lib.llama_engine_create(modelPath);
        if (handle == null) {
            throw new RuntimeException(
                    "Failed to create llama engine for model: " + modelPath);
        }
        return new LlamaEngine(lib, handle);
    }

    /**
     * Run raw inference on the given prompt (no chat template applied).
     *
     * @param prompt the input prompt (may be empty)
     * @return the generated completion text
     * @throws IllegalStateException if the engine has been closed
     * @throws RuntimeException      if inference fails
     */
    public String complete(String prompt) {
        ensureOpen();
        if (prompt == null) prompt = "";
        Pointer result = lib.llama_engine_complete(handle, prompt);
        if (result == null) throw new RuntimeException("llama_engine_complete returned null");
        try {
            return result.getString(0, "UTF-8");
        } finally {
            lib.llama_engine_free_string(result);
        }
    }

    /**
     * One-shot chat with an optional system message and a user message.
     *
     * <p>The model's built-in chat template is applied automatically.
     *
     * @param systemMsg system message (null or "" to omit)
     * @param userMsg   user message
     * @return the assistant's response
     * @throws IllegalStateException if the engine has been closed
     */
    public String chat(String systemMsg, String userMsg) {
        ensureOpen();
        if (systemMsg == null) systemMsg = "";
        if (userMsg   == null) userMsg   = "";
        Pointer result = lib.llama_engine_chat(handle, systemMsg, userMsg);
        if (result == null) throw new RuntimeException("llama_engine_chat returned null");
        try {
            return result.getString(0, "UTF-8");
        } finally {
            lib.llama_engine_free_string(result);
        }
    }

    /**
     * Chat with an explicit ordered list of {@link Message} objects
     * (chatWithObject equivalent).
     *
     * <p>The model's chat template is applied to the full message list.
     *
     * @param messages non-empty list of messages
     * @return the assistant's response
     * @throws IllegalArgumentException if messages is null or empty
     * @throws IllegalStateException    if the engine has been closed
     */
    public String chatWithMessages(List<Message> messages) {
        ensureOpen();
        if (messages == null || messages.isEmpty()) {
            throw new IllegalArgumentException("messages must not be null or empty");
        }
        int      n        = messages.size();
        String[] roles    = new String[n];
        String[] contents = new String[n];
        for (int i = 0; i < n; i++) {
            roles[i]    = messages.get(i).role;
            contents[i] = messages.get(i).content;
        }
        Pointer result = lib.llama_engine_chat_with_messages(handle, roles, contents, n);
        if (result == null) {
            throw new RuntimeException("llama_engine_chat_with_messages returned null");
        }
        try {
            return result.getString(0, "UTF-8");
        } finally {
            lib.llama_engine_free_string(result);
        }
    }

    /**
     * Session-based multi-turn chat.
     *
     * <p>Conversation history is maintained inside the native engine, keyed
     * by {@code sessionId}.  Each call appends {@code userMsg} to the session
     * and returns the next assistant turn.
     *
     * <p>Call {@link #chatSessionSetSystem} before the first turn to set a
     * system prompt.
     *
     * @param sessionId unique session identifier (must not be null or blank)
     * @param userMsg   the user's message
     * @return the assistant's response
     * @throws IllegalArgumentException if sessionId is null or blank
     * @throws IllegalStateException    if the engine has been closed
     */
    public String chatSession(String sessionId, String userMsg) {
        ensureOpen();
        if (sessionId == null || sessionId.isBlank()) {
            throw new IllegalArgumentException("sessionId must not be null or blank");
        }
        if (userMsg == null) userMsg = "";
        Pointer result = lib.llama_engine_chat_session(handle, sessionId, userMsg);
        if (result == null) throw new RuntimeException("llama_engine_chat_session returned null");
        try {
            return result.getString(0, "UTF-8");
        } finally {
            lib.llama_engine_free_string(result);
        }
    }

    /**
     * Set (or replace) the system message for a named session.
     *
     * <p>Call this before the first {@link #chatSession} call when a system
     * prompt is required.  Pass null or "" to clear the existing system
     * message.
     *
     * @param sessionId unique session identifier
     * @param systemMsg system prompt (null or "" to clear)
     * @throws IllegalStateException if the engine has been closed
     */
    public void chatSessionSetSystem(String sessionId, String systemMsg) {
        ensureOpen();
        if (sessionId == null || sessionId.isBlank()) {
            throw new IllegalArgumentException("sessionId must not be null or blank");
        }
        if (systemMsg == null) systemMsg = "";
        lib.llama_engine_chat_session_set_system(handle, sessionId, systemMsg);
    }

    /**
     * Clear all history for the named session (including system message).
     *
     * <p>The session slot is released and can be reused.
     *
     * @param sessionId session identifier to clear
     * @throws IllegalStateException if the engine has been closed
     */
    public void chatSessionClear(String sessionId) {
        ensureOpen();
        if (sessionId == null) return;
        lib.llama_engine_chat_session_clear(handle, sessionId);
    }

    /**
     * Chat with tool definitions.
     *
     * <p>The tool definitions are injected into the system message so that
     * any model can reason about available tools.  The <em>raw</em> model
     * output is returned — the caller is responsible for parsing and
     * executing any tool calls the model emits.
     *
     * @param messages  non-empty list of messages
     * @param toolsJson JSON array of tool definitions (OpenAI-compatible
     *                  format, e.g.
     *                  {@code [{"name":"…","description":"…","parameters":{…}}]})
     * @return the raw assistant response (may contain a tool-call JSON object)
     * @throws IllegalArgumentException if messages is null or empty
     * @throws IllegalStateException    if the engine has been closed
     */
    public String chatWithTools(List<Message> messages, String toolsJson) {
        ensureOpen();
        if (messages == null || messages.isEmpty()) {
            throw new IllegalArgumentException("messages must not be null or empty");
        }
        if (toolsJson == null) toolsJson = "[]";
        int      n        = messages.size();
        String[] roles    = new String[n];
        String[] contents = new String[n];
        for (int i = 0; i < n; i++) {
            roles[i]    = messages.get(i).role;
            contents[i] = messages.get(i).content;
        }
        Pointer result = lib.llama_engine_chat_with_tools(
                handle, roles, contents, n, toolsJson);
        if (result == null) {
            throw new RuntimeException("llama_engine_chat_with_tools returned null");
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

