package com.example.llama;

import com.sun.jna.Pointer;

/**
 * High-level Java wrapper around the llama_bridge C library.
 *
 * <p>Usage:
 * <pre>{@code
 * try (LlamaEngine engine = LlamaEngine.load("model.gguf")) {
 *
 *     // Raw completion (no chat template)
 *     String out = engine.complete("Say hello.");
 *
 *     // Session-based chat — returns ChatMessage{role, content}
 *     LlamaEngine.ChatMessage msg = engine.chat("sid-1",
 *         new LlamaEngine.ChatRequest("You are helpful.", "What is 2+2?", null, null));
 *     System.out.println(msg.content);
 *
 *     // Structured response with session metadata
 *     LlamaEngine.ChatResponse resp = engine.chatWithObject("sid-1",
 *         new LlamaEngine.ChatRequest(null, "Tell me more.", null, null));
 *     System.out.printf("role=%s content=%s session=%s count=%d%n",
 *         resp.role, resp.content, resp.sessionId, resp.messageCount);
 *
 *     // Inject tool result before user message
 *     engine.chat("sid-1", new LlamaEngine.ChatRequest(
 *         null, "What was the result?", null, "{\"result\":42}"));
 *
 *     // Clear session history
 *     engine.chatSessionClear("sid-1");
 * }
 * }</pre>
 *
 * <p>Instances are not thread-safe. Use one engine per thread or add
 * external synchronisation.
 */
public final class LlamaEngine implements AutoCloseable {

    // -----------------------------------------------------------------------
    // Nested types
    // -----------------------------------------------------------------------

    /**
     * Input to {@link #chat} and {@link #chatWithObject}.
     *
     * <p>{@code userMessage} is the main required field.
     * All other fields are optional and may be {@code null} or {@code ""}.
     */
    public static final class ChatRequest {

        /** Sets or replaces the session's system prompt. Null/empty to leave unchanged. */
        public final String systemMessage;

        /** The user turn for this call. */
        public final String userMessage;

        /**
         * Injects a prior assistant turn into the session before {@code userMessage}
         * (for few-shot context / correction). Null/empty to skip.
         */
        public final String assistantMessage;

        /**
         * Injects a tool-use response (role "tool") into the session before
         * {@code userMessage}. Null/empty to skip.
         */
        public final String toolMessage;

        /**
         * Full constructor.
         *
         * @param systemMessage    optional system prompt
         * @param userMessage      user turn (main input)
         * @param assistantMessage optional prior assistant turn to inject
         * @param toolMessage      optional tool response to inject
         */
        public ChatRequest(String systemMessage,
                           String userMessage,
                           String assistantMessage,
                           String toolMessage) {
            this.systemMessage    = systemMessage    != null ? systemMessage    : "";
            this.userMessage      = userMessage      != null ? userMessage      : "";
            this.assistantMessage = assistantMessage != null ? assistantMessage : "";
            this.toolMessage      = toolMessage      != null ? toolMessage      : "";
        }

        /** Convenience constructor for a simple user message. */
        public ChatRequest(String userMessage) {
            this(null, userMessage, null, null);
        }

        /** Convenience constructor for system + user message. */
        public ChatRequest(String systemMessage, String userMessage) {
            this(systemMessage, userMessage, null, null);
        }
    }

    /**
     * The response returned by {@link #chat}.
     * Always has {@code role = "assistant"}.
     */
    public static final class ChatMessage {
        /** Message role — always {@code "assistant"}. */
        public final String role;
        /** The assistant's reply. */
        public final String content;

        ChatMessage(String role, String content) {
            this.role    = role;
            this.content = content;
        }
    }

    /**
     * The richer response returned by {@link #chatWithObject}.
     * Includes session metadata in addition to the assistant reply.
     */
    public static final class ChatResponse {
        /** Message role — always {@code "assistant"}. */
        public final String role;
        /** The assistant's reply. */
        public final String content;
        /** The session ID this response belongs to. */
        public final String sessionId;
        /** Total number of messages in the session after this turn. */
        public final int    messageCount;

        ChatResponse(String role, String content, String sessionId, int messageCount) {
            this.role         = role;
            this.content      = content;
            this.sessionId    = sessionId;
            this.messageCount = messageCount;
        }
    }

    // -----------------------------------------------------------------------
    // JSON parsing helpers — no external dependency required
    // -----------------------------------------------------------------------

    /**
     * Extract a JSON string field from a simple flat JSON object.
     * Handles basic escape sequences (\", \\, \n, \r, \t).
     */
    static String jsonString(String json, String key) {
        String needle = "\"" + key + "\":\"";
        int start = json.indexOf(needle);
        if (start == -1) return "";
        start += needle.length();
        StringBuilder sb = new StringBuilder();
        while (start < json.length()) {
            char c = json.charAt(start);
            if (c == '"') break;
            if (c == '\\' && start + 1 < json.length()) {
                start++;
                switch (json.charAt(start)) {
                    case '"':  sb.append('"');  break;
                    case '\\': sb.append('\\'); break;
                    case 'n':  sb.append('\n'); break;
                    case 'r':  sb.append('\r'); break;
                    case 't':  sb.append('\t'); break;
                    default:   sb.append(json.charAt(start)); break;
                }
            } else {
                sb.append(c);
            }
            start++;
        }
        return sb.toString();
    }

    /**
     * Extract a JSON integer field from a simple flat JSON object.
     */
    static int jsonInt(String json, String key) {
        String needle = "\"" + key + "\":";
        int start = json.indexOf(needle);
        if (start == -1) return 0;
        start += needle.length();
        while (start < json.length() && json.charAt(start) == ' ') start++;
        int end = start;
        while (end < json.length()
                && (Character.isDigit(json.charAt(end)) || json.charAt(end) == '-')) {
            end++;
        }
        if (start == end) return 0;
        try {
            return Integer.parseInt(json.substring(start, end));
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    // -----------------------------------------------------------------------
    // Engine state
    // -----------------------------------------------------------------------

    private final LlamaLibrary lib;
    private Pointer handle;

    private LlamaEngine(LlamaLibrary lib, Pointer handle) {
        this.lib    = lib;
        this.handle = handle;
    }

    // -----------------------------------------------------------------------
    // Factory
    // -----------------------------------------------------------------------

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

    // -----------------------------------------------------------------------
    // Raw completion
    // -----------------------------------------------------------------------

    /**
     * Run raw inference on the given prompt (no chat template applied).
     *
     * @param prompt the input prompt (may be empty)
     * @return the generated completion text
     * @throws IllegalStateException if the engine has been closed
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

    // -----------------------------------------------------------------------
    // Chat API
    // -----------------------------------------------------------------------

    /**
     * Session-based chat turn.
     *
     * <p>The engine maintains conversation history keyed by {@code sessionId}.
     * The {@link ChatRequest} may contain any combination of
     * {@code systemMessage}, {@code userMessage}, {@code assistantMessage}, and
     * {@code toolMessage}; all non-empty fields are appended to the session in
     * that order before running inference.
     *
     * <p>The model's built-in chat template is applied to the full session history.
     * The assistant response is automatically appended to the session history.
     *
     * @param sessionId session identifier (created automatically on first call)
     * @param request   chat request containing the message fields
     * @return a {@link ChatMessage} with the assistant's reply
     * @throws IllegalArgumentException if sessionId is null or blank
     * @throws IllegalStateException    if the engine has been closed
     */
    public ChatMessage chat(String sessionId, ChatRequest request) {
        ensureOpen();
        if (sessionId == null || sessionId.isBlank()) {
            throw new IllegalArgumentException("sessionId must not be null or blank");
        }
        if (request == null) request = new ChatRequest("");

        Pointer result = lib.llama_engine_chat(
                handle,
                sessionId,
                request.systemMessage,
                request.userMessage,
                request.assistantMessage,
                request.toolMessage);

        if (result == null) throw new RuntimeException("llama_engine_chat returned null");

        try {
            String json    = result.getString(0, "UTF-8");
            String role    = jsonString(json, "role");
            String content = jsonString(json, "content");
            return new ChatMessage(role, content);
        } finally {
            lib.llama_engine_free_string(result);
        }
    }

    /**
     * Session-based chat turn that returns a richer schema response.
     *
     * <p>Same as {@link #chat} but the returned {@link ChatResponse} also
     * contains the {@code sessionId} and the total {@code messageCount} in the
     * session after this turn.
     *
     * @param sessionId session identifier (created automatically on first call)
     * @param request   chat request containing the message fields
     * @return a {@link ChatResponse} with the assistant's reply and session metadata
     * @throws IllegalArgumentException if sessionId is null or blank
     * @throws IllegalStateException    if the engine has been closed
     */
    public ChatResponse chatWithObject(String sessionId, ChatRequest request) {
        ensureOpen();
        if (sessionId == null || sessionId.isBlank()) {
            throw new IllegalArgumentException("sessionId must not be null or blank");
        }
        if (request == null) request = new ChatRequest("");

        Pointer result = lib.llama_engine_chat_with_object(
                handle,
                sessionId,
                request.systemMessage,
                request.userMessage,
                request.assistantMessage,
                request.toolMessage);

        if (result == null) {
            throw new RuntimeException("llama_engine_chat_with_object returned null");
        }

        try {
            String json         = result.getString(0, "UTF-8");
            String role         = jsonString(json, "role");
            String content      = jsonString(json, "content");
            String sid          = jsonString(json, "sessionId");
            int    messageCount = jsonInt(json, "messageCount");
            return new ChatResponse(role, content, sid, messageCount);
        } finally {
            lib.llama_engine_free_string(result);
        }
    }

    /**
     * Clear all history for the named session (including the system message).
     * The session slot is released for reuse.
     *
     * @param sessionId session identifier to clear
     * @throws IllegalStateException if the engine has been closed
     */
    public void chatSessionClear(String sessionId) {
        ensureOpen();
        if (sessionId != null && !sessionId.isBlank()) {
            lib.llama_engine_chat_session_clear(handle, sessionId);
        }
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

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
