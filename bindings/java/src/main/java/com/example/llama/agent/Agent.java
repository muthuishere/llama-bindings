package com.example.llama.agent;

import com.example.llama.ChatEngine;
import com.example.llama.ChatOptions;
import com.example.llama.EmbedEngine;
import com.example.llama.LlamaException;
import com.example.llama.LoadOptions;
import com.example.llama.knowledge.KnowledgeStore;
import com.example.llama.model.*;
import com.example.llama.tools.ToolRegistry;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.sql.SQLException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Agentic chat loop that orchestrates:
 * <ul>
 *   <li>ChatEngine – language model inference</li>
 *   <li>EmbedEngine – text embedding for RAG</li>
 *   <li>KnowledgeStore – persistent vector + FTS knowledge base</li>
 *   <li>ToolRegistry – callable tool dispatch</li>
 * </ul>
 *
 * <p>The Agent maintains per-session conversation history and injects relevant
 * knowledge context into the system prompt on every turn.
 *
 * <p>Usage:
 * <pre>{@code
 * try (Agent agent = Agent.create("chat.gguf", "embed.gguf", ":memory:")) {
 *     agent.addDocument("The capital of France is Paris.");
 *     agent.addTool(new ToolDefinition("lookup", "...", params),
 *                   args -> Map.of("result", "..."));
 *     String reply = agent.chat("session-1", "What is the capital of France?");
 * }
 * }</pre>
 */
public final class Agent implements AutoCloseable {

    private static final int MAX_TOOL_ITERATIONS = 10;
    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final ChatEngine      chat;
    private final EmbedEngine     embed;
    private final KnowledgeStore  store;
    private final ToolRegistry    registry;

    /** sessionId → ordered message history. */
    private final Map<String, List<ChatMessage>> sessions = new ConcurrentHashMap<>();

    private volatile boolean closed = false;

    private Agent(ChatEngine chat, EmbedEngine embed,
                  KnowledgeStore store, ToolRegistry registry) {
        this.chat     = chat;
        this.embed    = embed;
        this.store    = store;
        this.registry = registry;
    }

    /**
     * Create an Agent that owns its engines and knowledge store.
     *
     * @param chatModelPath  path to the chat GGUF model
     * @param embedModelPath path to the embedding GGUF model
     * @param storagePath    SQLite path or {@code ":memory:"}
     * @return a ready-to-use Agent
     * @throws LlamaException if an engine cannot be loaded
     * @throws SQLException   if the knowledge store cannot be opened
     */
    public static Agent create(String chatModelPath,
                               String embedModelPath,
                               String storagePath)
            throws LlamaException, SQLException {
        ChatEngine  chat  = ChatEngine.load(chatModelPath, new LoadOptions());
        EmbedEngine embed = EmbedEngine.load(embedModelPath, new LoadOptions());
        KnowledgeStore store;
        try {
            store = new KnowledgeStore(storagePath);
        } catch (SQLException e) {
            chat.close();
            embed.close();
            throw e;
        }
        ToolRegistry registry = new ToolRegistry();
        return new Agent(chat, embed, store, registry);
    }

    /**
     * Embed text and add it to the knowledge store.
     *
     * @param text document text (must not be empty)
     * @throws LlamaException if embedding fails
     * @throws SQLException   if the store cannot be updated
     */
    public void addDocument(String text) throws LlamaException, SQLException {
        float[] vec = embed.embed(text, new com.example.llama.EmbedOptions());
        store.add(text, vec);
    }

    /**
     * Register a callable tool.
     *
     * @param def     tool definition
     * @param handler receives decoded JSON arguments and returns a result
     */
    public void addTool(ToolDefinition def, ToolRegistry.Handler handler) {
        registry.register(def, handler);
    }

    /**
     * Send a user message and return the assistant's final text reply.
     *
     * <p>The method:
     * <ol>
     *   <li>Embeds the message and retrieves relevant knowledge context.</li>
     *   <li>Injects context into the system prompt.</li>
     *   <li>Appends the user message to the session history.</li>
     *   <li>Loops up to {@value #MAX_TOOL_ITERATIONS} times: executes tool
     *       calls and feeds results back until the model returns text.</li>
     * </ol>
     *
     * @param sessionId conversation identifier (arbitrary string)
     * @param message   user message
     * @return assistant text reply
     * @throws LlamaException if chat inference fails
     */
    public String chat(String sessionId, String message) throws LlamaException {
        if (closed) throw new LlamaException("ENGINE_CLOSED", "Agent is closed");

        // 1. Retrieve knowledge context.
        List<KnowledgeStore.Document> context = retrieveContext(message);

        // 2. Build system prompt.
        String systemPrompt = buildSystemPrompt(context);

        // 3. Update session history.
        sessions.computeIfAbsent(sessionId, k -> new ArrayList<>())
                .add(ChatMessage.user(message));

        // 4. Agentic loop.
        List<ToolDefinition> toolDefs = registry.definitions();
        List<ChatMessage> msgs = buildMessages(systemPrompt,
                sessions.get(sessionId));

        for (int i = 0; i < MAX_TOOL_ITERATIONS; i++) {
            boolean isLast = (i == MAX_TOOL_ITERATIONS - 1);
            String responseMode = (!toolDefs.isEmpty() && !isLast)
                    ? "tool_call" : "text";

            ChatRequest req = ChatRequest.builder()
                    .messages(msgs)
                    .responseMode(responseMode)
                    .tools(toolDefs.isEmpty() ? null : toolDefs)
                    .toolChoice(toolDefs.isEmpty() ? null : "auto")
                    .generation(new GenerationOptions(0.7f, 512, 0.95f, 40))
                    .build();

            ChatResponse resp = chat.chat(req, new ChatOptions());

            switch (resp.type == null ? "" : resp.type) {
                case "assistant_text" -> {
                    String text = resp.text != null ? resp.text : "";
                    sessions.get(sessionId).add(ChatMessage.assistant(text));
                    return text;
                }
                case "structured_json" -> {
                    String text = toJson(resp.json);
                    sessions.get(sessionId).add(ChatMessage.assistant(text));
                    return text;
                }
                case "tool_call" -> {
                    if (resp.toolCalls == null || resp.toolCalls.isEmpty()) {
                        throw new LlamaException("INFERENCE_FAILED",
                                "Agent: model returned tool_call with no tool calls");
                    }
                    msgs.add(ChatMessage.assistant(toJson(resp.toolCalls)));
                    for (ToolCall tc : resp.toolCalls) {
                        String resultJson;
                        try {
                            Object result = registry.execute(tc.name, tc.arguments);
                            resultJson = toJson(result);
                        } catch (Exception e) {
                            resultJson = "{\"error\":\"" +
                                    e.getMessage().replace("\"", "'") + "\"}";
                        }
                        msgs.add(ChatMessage.tool(tc.name, resultJson));
                    }
                }
                default -> throw new LlamaException("INFERENCE_FAILED",
                        "Agent: unexpected response type: " + resp.type);
            }
        }

        throw new LlamaException("INFERENCE_FAILED",
                "Agent: exceeded " + MAX_TOOL_ITERATIONS + " tool call iterations");
    }

    /**
     * Clear the conversation history for the given session.
     *
     * @param sessionId session to clear
     */
    public void clearSession(String sessionId) {
        sessions.remove(sessionId);
    }

    /**
     * Close all owned resources (engines + knowledge store).
     * Safe to call multiple times.
     */
    @Override
    public void close() {
        if (closed) return;
        closed = true;
        store.close();
        embed.close();
        chat.close();
    }

    // ──────────────────────────────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────────────────────────────

    private List<KnowledgeStore.Document> retrieveContext(String text) {
        try {
            float[] vec = embed.embed(text, new com.example.llama.EmbedOptions());
            return store.search(vec, text, 5);
        } catch (Exception ignored) {
            return List.of();
        }
    }

    private static String buildSystemPrompt(List<KnowledgeStore.Document> docs) {
        if (docs.isEmpty()) {
            return "You are a helpful AI assistant.";
        }
        StringBuilder sb = new StringBuilder("You are a helpful AI assistant.\n\nRelevant context:\n");
        for (KnowledgeStore.Document d : docs) {
            sb.append("- ").append(d.text).append('\n');
        }
        return sb.toString();
    }

    private static List<ChatMessage> buildMessages(String systemPrompt,
                                                   List<ChatMessage> history) {
        List<ChatMessage> msgs = new ArrayList<>(history.size() + 1);
        msgs.add(ChatMessage.system(systemPrompt));
        msgs.addAll(history);
        return msgs;
    }

    private static String toJson(Object obj) {
        try {
            return MAPPER.writeValueAsString(obj);
        } catch (Exception e) {
            return "{}";
        }
    }
}
