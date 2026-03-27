package com.example.llama;

import com.sun.jna.Pointer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

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
 *     // Session-based chat — pass an ordered list of ChatMessage values,
 *     // get back a ChatMessage{role="assistant", content="..."}.
 *     LlamaEngine.ChatMessage msg = engine.chat("sid-1", List.of(
 *         new LlamaEngine.ChatMessage("system", "You are helpful."),
 *         new LlamaEngine.ChatMessage("user",   "What is 2+2?")));
 *     System.out.println(msg.content);
 *
 *     // Richer schema response — returns a Map with role, content,
 *     // sessionId, and messageCount.
 *     Map<String, Object> resp = engine.chatWithObject("sid-1", List.of(
 *         new LlamaEngine.ChatMessage("user", "Tell me more.")));
 *     System.out.println(resp.get("role") + " " + resp.get("sessionId"));
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
     * A role+content message — both the element type of the input list
     * passed to {@link #chat} and {@link #chatWithObject}, and the
     * response type returned by {@link #chat}.
     *
     * <p>Supported roles: {@code system}, {@code user}, {@code assistant},
     * {@code tool}.
     */
    public static final class ChatMessage {
        /** Message role. */
        public final String role;
        /** Message content. */
        public final String content;

        /**
         * Create a message with the given role and content.
         * Null values are normalised to empty strings.
         *
         * @param role    message role (e.g. "user", "system", "assistant", "tool")
         * @param content message text
         */
        public ChatMessage(String role, String content) {
            this.role    = role    != null ? role    : "";
            this.content = content != null ? content : "";
        }
    }

    // -----------------------------------------------------------------------
    // JSON helpers — no external dependency required
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

    /** Quote and JSON-escape a string value. */
    private static String jsonQuote(String s) {
        if (s == null) s = "";
        StringBuilder sb = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"':  sb.append("\\\""); break;
                case '\\': sb.append("\\\\"); break;
                case '\n': sb.append("\\n");  break;
                case '\r': sb.append("\\r");  break;
                case '\t': sb.append("\\t");  break;
                default:   sb.append(c);      break;
            }
        }
        sb.append('"');
        return sb.toString();
    }

    /** Serialize a List of ChatMessage to a JSON array string. */
    private static String toMessagesJson(List<ChatMessage> messages) {
        StringBuilder sb = new StringBuilder("[");
        if (messages != null) {
            for (int i = 0; i < messages.size(); i++) {
                ChatMessage m = messages.get(i);
                if (i > 0) sb.append(',');
                sb.append("{\"role\":");
                sb.append(jsonQuote(m.role));
                sb.append(",\"content\":");
                sb.append(jsonQuote(m.content));
                sb.append('}');
            }
        }
        sb.append(']');
        return sb.toString();
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
     * {@code messages} is an ordered list of {@link ChatMessage} values to
     * append to the session history for this turn.  Supported roles:
     * {@code system}, {@code user}, {@code assistant}, {@code tool}.
     * A {@code system} entry sets or replaces the session system prompt.
     *
     * <p>The model's built-in chat template is applied to the full session history.
     * The assistant response is automatically appended to the session history.
     *
     * @param sessionId session identifier (created automatically on first call)
     * @param messages  ordered list of messages to append for this turn
     * @return a {@link ChatMessage} with {@code role="assistant"} and the reply
     * @throws IllegalArgumentException if sessionId is null or blank
     * @throws IllegalStateException    if the engine has been closed
     */
    public ChatMessage chat(String sessionId, List<ChatMessage> messages) {
        ensureOpen();
        if (sessionId == null || sessionId.isBlank()) {
            throw new IllegalArgumentException("sessionId must not be null or blank");
        }
        String messagesJson = toMessagesJson(messages);
        Pointer result = lib.llama_engine_chat_messages(handle, sessionId, messagesJson);
        if (result == null) throw new RuntimeException("llama_engine_chat_messages returned null");
        try {
            String json = result.getString(0, "UTF-8");
            return new ChatMessage(jsonString(json, "role"), jsonString(json, "content"));
        } finally {
            lib.llama_engine_free_string(result);
        }
    }

    /**
     * Session-based chat turn that returns a richer JSON object.
     *
     * <p>Same as {@link #chat} but the returned map also contains
     * {@code sessionId} (String) and {@code messageCount} (Integer).
     *
     * @param sessionId session identifier (created automatically on first call)
     * @param messages  ordered list of messages to append for this turn
     * @return a map with keys {@code role}, {@code content}, {@code sessionId},
     *         and {@code messageCount}
     * @throws IllegalArgumentException if sessionId is null or blank
     * @throws IllegalStateException    if the engine has been closed
     */
    public Map<String, Object> chatWithObject(String sessionId, List<ChatMessage> messages) {
        ensureOpen();
        if (sessionId == null || sessionId.isBlank()) {
            throw new IllegalArgumentException("sessionId must not be null or blank");
        }
        String messagesJson = toMessagesJson(messages);
        Pointer result = lib.llama_engine_chat_with_object_messages(handle, sessionId, messagesJson);
        if (result == null) {
            throw new RuntimeException("llama_engine_chat_with_object_messages returned null");
        }
        try {
            String json = result.getString(0, "UTF-8");
            Map<String, Object> map = new LinkedHashMap<>();
            map.put("role",         jsonString(json, "role"));
            map.put("content",      jsonString(json, "content"));
            map.put("sessionId",    jsonString(json, "sessionId"));
            map.put("messageCount", jsonInt(json, "messageCount"));
            return map;
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
