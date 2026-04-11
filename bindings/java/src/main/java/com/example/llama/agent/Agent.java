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

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.SQLException;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

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
    private final String          chatModelPath;
    private final String          embedModelPath;
    private final String          storagePath;

    /** sessionId → ordered message history. */
    private final Map<String, List<ChatMessage>> sessions = new ConcurrentHashMap<>();

    private volatile boolean closed = false;

    private Agent(ChatEngine chat, EmbedEngine embed,
                  KnowledgeStore store, ToolRegistry registry,
                  String chatModelPath, String embedModelPath,
                  String storagePath) {
        this.chat           = chat;
        this.embed          = embed;
        this.store          = store;
        this.registry       = registry;
        this.chatModelPath  = chatModelPath;
        this.embedModelPath = embedModelPath;
        this.storagePath    = storagePath;
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
        return new Agent(chat, embed, store, registry,
                         chatModelPath, embedModelPath, storagePath);
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
        boolean justCalledTool = false;

        for (int i = 0; i < MAX_TOOL_ITERATIONS; i++) {
            // After a tool call + result, ask the model for a text answer.
            List<ToolDefinition> useTools = toolDefs;
            String responseMode = toolDefs.isEmpty() ? "text" : "tool_call";
            if (justCalledTool) {
                useTools = List.of();
                responseMode = "text";
                justCalledTool = false;
            }

            ChatRequest req = ChatRequest.builder()
                    .messages(msgs)
                    .responseMode(responseMode)
                    .tools(useTools.isEmpty() ? null : useTools)
                    .toolChoice(useTools.isEmpty() ? null : "auto")
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
                    // Continue the loop — next iteration uses text mode.
                    justCalledTool = true;
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
     * Export a complete Agent bundle to a ZIP archive at {@code zipPath}.
     *
     * <p>The archive contains:
     * <ul>
     *   <li>{@code manifest.json} – metadata (version, model names, doc count, etc.)</li>
     *   <li>{@code knowledge.json} – all documents with text and embeddings</li>
     *   <li>{@code knowledge.db} – raw SQLite file (empty placeholder for in-memory stores)</li>
     * </ul>
     *
     * @param zipPath path where the ZIP archive will be written
     * @throws IOException  if writing fails
     * @throws SQLException if the knowledge store cannot be queried
     */
    public void export(String zipPath) throws IOException, SQLException {
        List<KnowledgeStore.Document> docs = store.all();

        // Build manifest.
        Map<String, Object> manifest = new LinkedHashMap<>();
        manifest.put("version", "1");
        manifest.put("created_at", Instant.now().toString());
        manifest.put("chat_model", Path.of(chatModelPath).getFileName().toString());
        manifest.put("embed_model", Path.of(embedModelPath).getFileName().toString());
        manifest.put("storage_path", storagePath);
        manifest.put("doc_count", docs.size());
        manifest.put("embedding_dim",
                docs.isEmpty() ? 0
                        : (docs.get(0).embedding != null ? docs.get(0).embedding.length : 0));

        // Build knowledge array.
        List<Map<String, Object>> knowledgeList = new ArrayList<>(docs.size());
        for (KnowledgeStore.Document doc : docs) {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("text", doc.text);
            if (doc.embedding != null) {
                List<Float> embList = new ArrayList<>(doc.embedding.length);
                for (float f : doc.embedding) embList.add(f);
                entry.put("embedding", embList);
            } else {
                entry.put("embedding", List.of());
            }
            knowledgeList.add(entry);
        }

        byte[] manifestBytes  = MAPPER.writerWithDefaultPrettyPrinter()
                .writeValueAsBytes(manifest);
        byte[] knowledgeBytes = MAPPER.writerWithDefaultPrettyPrinter()
                .writeValueAsBytes(knowledgeList);

        // Read SQLite file bytes (or empty placeholder for :memory:).
        byte[] dbBytes;
        if (":memory:".equals(storagePath)) {
            dbBytes = new byte[0];
        } else {
            Path dbPath = Path.of(storagePath);
            dbBytes = Files.exists(dbPath) ? Files.readAllBytes(dbPath) : new byte[0];
        }

        // Write ZIP archive.
        try (ZipOutputStream zos = new ZipOutputStream(
                new BufferedOutputStream(Files.newOutputStream(Path.of(zipPath))))) {
            writeZipEntry(zos, "manifest.json", manifestBytes);
            writeZipEntry(zos, "knowledge.json", knowledgeBytes);
            writeZipEntry(zos, "knowledge.db", dbBytes);
        }
    }

    /**
     * Import an Agent from a previously exported ZIP archive.
     *
     * <p>The method extracts the SQLite database from the archive, creates
     * chat and embed engines from the supplied model paths, and returns a
     * fully initialised Agent.
     *
     * @param zipPath        path to the ZIP archive
     * @param chatModelPath  path to the chat GGUF model
     * @param embedModelPath path to the embedding GGUF model
     * @param storagePath    SQLite path for the restored database;
     *                       defaults to {@code "agent.db"} if {@code null} or empty
     * @return a restored Agent
     * @throws IOException    if reading the archive fails
     * @throws LlamaException if an engine cannot be loaded
     * @throws SQLException   if the knowledge store cannot be opened
     */
    @SuppressWarnings("unchecked")
    public static Agent importFrom(String zipPath, String chatModelPath,
                                   String embedModelPath, String storagePath)
            throws IOException, LlamaException, SQLException {
        if (storagePath == null || storagePath.isEmpty()) {
            storagePath = "agent.db";
        }

        // Read ZIP entries.
        byte[] manifestBytes  = null;
        byte[] dbBytes        = null;

        try (ZipInputStream zis = new ZipInputStream(
                new BufferedInputStream(Files.newInputStream(Path.of(zipPath))))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                switch (entry.getName()) {
                    case "manifest.json"  -> manifestBytes = zis.readAllBytes();
                    case "knowledge.db"   -> dbBytes = zis.readAllBytes();
                    default -> { /* skip unknown entries */ }
                }
                zis.closeEntry();
            }
        }

        // Validate manifest.
        if (manifestBytes == null) {
            throw new IOException("Invalid agent archive: missing manifest.json");
        }
        Map<String, Object> manifest = MAPPER.readValue(manifestBytes, Map.class);
        String version = String.valueOf(manifest.get("version"));
        if (!"1".equals(version)) {
            throw new IOException("Unsupported agent archive version: " + version);
        }

        // Restore SQLite database.
        if (dbBytes != null && dbBytes.length > 0) {
            Files.write(Path.of(storagePath), dbBytes);
        }

        // Create engines and knowledge store.
        ChatEngine chat = ChatEngine.load(chatModelPath, new LoadOptions());
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
        return new Agent(chat, embed, store, registry,
                         chatModelPath, embedModelPath, storagePath);
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

    private static void writeZipEntry(ZipOutputStream zos, String name, byte[] data)
            throws IOException {
        zos.putNextEntry(new ZipEntry(name));
        zos.write(data);
        zos.closeEntry();
    }
}
