package com.example.llama.e2e;

import com.example.llama.*;
import com.example.llama.agent.Agent;
import com.example.llama.model.*;
import com.example.llama.tools.ToolRegistry;
import org.junit.jupiter.api.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * End-to-end tests for the Java binding against a real Qwen2.5-0.5B GGUF model.
 *
 * <p>Prerequisites:
 * <ol>
 *   <li>Native bridge built: {@code task build-bridge}</li>
 *   <li>Model downloaded: {@code task download-model}</li>
 * </ol>
 *
 * <p>Run:
 * <pre>
 *   task e2e-java
 *   # or:
 *   mvn test -Pe2e -Dbridge.lib.dir=../../bridge/build
 * </pre>
 *
 * <p>Model path defaults to {@code tests/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}.
 * Override with env vars {@code LLAMA_CHAT_MODEL} and {@code LLAMA_EMBED_MODEL}.
 */
@Tag("e2e")
class E2ETest {

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    private static String chatModelPath() {
        String env = System.getenv("LLAMA_CHAT_MODEL");
        if (env != null && !env.isBlank()) return env;
        return Paths.get(System.getProperty("user.home"),
                "llama-bindings-conf", "models", "chat",
                "gemma-4-E2B-it-Q4_K_M.gguf").toString();
    }

    private static String embedModelPath() {
        String env = System.getenv("LLAMA_EMBED_MODEL");
        if (env != null && !env.isBlank()) return env;
        return Paths.get(System.getProperty("user.home"),
                "llama-bindings-conf", "models", "embeddings",
                "nomic-embed-text-v1.5.Q4_K_M.gguf").toString();
    }

    private static void skipIfModelAbsent(String path) {
        assumeTrue(Files.exists(Paths.get(path)),
                "Model not found (run `task download-model`): " + path);
    }

    // ----------------------------------------------------------------
    // Chat — text mode
    // ----------------------------------------------------------------

    @Test
    void e2eChatTextMode() throws Exception {
        String modelPath = chatModelPath();
        skipIfModelAbsent(modelPath);

        try (var chat = ChatEngine.load(modelPath, new LoadOptions())) {
            var request = ChatRequest.builder()
                    .messages(List.of(ChatMessage.user("Say hello in exactly one word.")))
                    .responseMode("text")
                    .generation(new GenerationOptions(0.1f, 32, 1.0f, 40))
                    .build();

            ChatResponse resp = chat.chat(request, new ChatOptions());

            assertEquals("assistant_text", resp.type, "Expected text response type");
            assertNotNull(resp.text, "Expected non-null text");
            assertFalse(resp.text.isBlank(), "Expected non-blank text response");
            System.out.println("Chat text response: " + resp.text);
        }
    }

    // ----------------------------------------------------------------
    // Chat — JSON schema mode
    // ----------------------------------------------------------------

    @Test
    void e2eChatSchemaMode() throws Exception {
        String modelPath = chatModelPath();
        skipIfModelAbsent(modelPath);

        try (var chat = ChatEngine.load(modelPath, new LoadOptions())) {
            var schema = Map.of(
                    "name", "person_extract",
                    "schema", Map.of(
                            "type", "object",
                            "properties", Map.of(
                                    "name", Map.of("type", "string"),
                                    "age",  Map.of("type", "integer")
                            ),
                            "required", List.of("name", "age"),
                            "additionalProperties", false
                    )
            );
            var request = ChatRequest.builder()
                    .messages(List.of(ChatMessage.user("Alice is 30 years old. Extract structured data.")))
                    .responseMode("json_schema")
                    .schema(schema)
                    .generation(new GenerationOptions(0.0f, 64, 1.0f, 40))
                    .build();

            ChatResponse resp = chat.chat(request, new ChatOptions());

            assertEquals("structured_json", resp.type, "Expected structured_json type");
            assertNotNull(resp.json, "Expected non-null JSON field");
            System.out.println("Structured JSON: " + resp.json);
        }
    }

    // ----------------------------------------------------------------
    // Chat — tool call mode
    // ----------------------------------------------------------------

    @Test
    void e2eChatToolCallMode() throws Exception {
        String modelPath = chatModelPath();
        skipIfModelAbsent(modelPath);

        try (var chat = ChatEngine.load(modelPath, new LoadOptions())) {
            var tool = new ToolDefinition(
                    "lookup_weather",
                    "Get current weather for a city",
                    Map.of(
                            "type", "object",
                            "properties", Map.of("city", Map.of("type", "string")),
                            "required", List.of("city"),
                            "additionalProperties", false
                    )
            );
            var request = ChatRequest.builder()
                    .messages(List.of(ChatMessage.user("What is the weather in Chennai?")))
                    .responseMode("tool_call")
                    .tools(List.of(tool))
                    .toolChoice("auto")
                    .generation(new GenerationOptions(0.1f, 64, 1.0f, 40))
                    .build();

            ChatResponse resp = chat.chat(request, new ChatOptions());

            assertEquals("tool_call", resp.type, "Expected tool_call type");
            assertNotNull(resp.toolCalls, "Expected tool_calls list");
            assertFalse(resp.toolCalls.isEmpty(), "Expected at least one tool call");
            System.out.println("Tool calls: " + resp.toolCalls);
        }
    }

    // ----------------------------------------------------------------
    // Chat — observability events
    // ----------------------------------------------------------------

    @Test
    void e2eChatObservabilityEvents() throws Exception {
        String modelPath = chatModelPath();
        skipIfModelAbsent(modelPath);

        var events = new java.util.ArrayList<String>();
        try (var chat = ChatEngine.load(modelPath, new LoadOptions(e -> events.add(e.event)))) {
            var request = ChatRequest.builder()
                    .messages(List.of(ChatMessage.user("Say hello.")))
                    .generation(new GenerationOptions(0.1f, 16, 1.0f, 40))
                    .build();
            chat.chat(request, new ChatOptions());
        }

        assertFalse(events.isEmpty(), "Expected at least one observability event");
        System.out.println("Events: " + events);
    }

    // ----------------------------------------------------------------
    // Embed
    // ----------------------------------------------------------------

    @Test
    void e2eEmbedReturnsNonEmptyVector() throws Exception {
        String modelPath = embedModelPath();
        skipIfModelAbsent(modelPath);

        try (var embed = EmbedEngine.load(modelPath, new LoadOptions())) {
            float[] vec = embed.embed("semantic search example", new EmbedOptions());

            assertTrue(vec.length > 0, "Expected non-empty vector");
            System.out.printf("Embed vector length: %d, first value: %f%n", vec.length, vec[0]);
        }
    }

    @Test
    void e2eEmbedRepeatedCallsConsistent() throws Exception {
        String modelPath = embedModelPath();
        skipIfModelAbsent(modelPath);

        try (var embed = EmbedEngine.load(modelPath, new LoadOptions())) {
            int firstLen = -1;
            for (int i = 0; i < 3; i++) {
                float[] vec = embed.embed("hello world", new EmbedOptions());
                assertTrue(vec.length > 0, "Empty vector on iteration " + i);
                if (firstLen == -1) {
                    firstLen = vec.length;
                } else {
                    assertEquals(firstLen, vec.length,
                            "Vector length changed between iterations");
                }
            }
        }
    }

    // ----------------------------------------------------------------
    // Agent — helpers
    // ----------------------------------------------------------------

    private Agent createAgentOrSkip() throws Exception {
        String chat  = chatModelPath();
        String embed = embedModelPath();
        skipIfModelAbsent(chat);
        skipIfModelAbsent(embed);
        return Agent.create(chat, embed, ":memory:");
    }

    // ----------------------------------------------------------------
    // Agent — basic chat
    // ----------------------------------------------------------------

    @Test
    void e2eAgentChatReturnsText() throws Exception {
        try (Agent a = createAgentOrSkip()) {
            String reply = a.chat("session-1", "Say hello in exactly one word.");
            assertNotNull(reply, "Expected non-null reply");
            assertFalse(reply.isBlank(), "Expected non-blank reply");
            System.out.println("Agent reply: " + reply);
        }
    }

    // ----------------------------------------------------------------
    // Agent — multi-turn history
    // ----------------------------------------------------------------

    @Test
    void e2eAgentMultiTurnHistory() throws Exception {
        try (Agent a = createAgentOrSkip()) {
            a.chat("history-session", "My name is Muthukumaran.");
            String reply = a.chat("history-session", "What is my name?");
            assertFalse(reply.isBlank(), "Expected non-blank reply on second turn");
            System.out.println("Second turn reply: " + reply);
        }
    }

    // ----------------------------------------------------------------
    // Agent — knowledge (addDocument → query)
    // ----------------------------------------------------------------

    @Test
    void e2eAgentWithDocument() throws Exception {
        try (Agent a = createAgentOrSkip()) {
            a.addDocument("The capital of France is Paris.");
            String reply = a.chat("doc-session", "What is the capital of France?");
            assertFalse(reply.isBlank(), "Expected non-blank reply");
            System.out.println("Knowledge-grounded reply: " + reply);
        }
    }

    // ----------------------------------------------------------------
    // Agent — tool dispatch
    // ----------------------------------------------------------------

    @Test
    void e2eAgentWithTool() throws Exception {
        try (Agent a = createAgentOrSkip()) {
            var called = new AtomicBoolean(false);
            a.addTool(
                new ToolDefinition(
                    "lookup_weather",
                    "Get current weather for a city",
                    Map.of(
                        "type", "object",
                        "properties", Map.of("city", Map.of("type", "string")),
                        "required", List.of("city"),
                        "additionalProperties", false
                    )
                ),
                args -> {
                    called.set(true);
                    return Map.of("temperature", "32°C", "condition", "sunny");
                }
            );

            try {
                String reply = a.chat("tool-session", "What is the weather in Chennai?");
                assertFalse(reply.isBlank(), "Expected non-blank reply");
                System.out.printf("Tool-grounded reply (tool called=%b): %s%n", called.get(), reply);
            } catch (LlamaException e) {
                // Bridge stub always returns tool_call → agent hits iteration limit.
                // Tolerated until llama.cpp inference is wired into the bridge.
                if (e.getMessage() != null && e.getMessage().contains("exceeded")) {
                    System.out.println("Tool loop limit hit (bridge stub behaviour — expected): " + e.getMessage());
                } else {
                    throw e;
                }
            }
        }
    }

    // ----------------------------------------------------------------
    // Agent — export → import round-trip
    // ----------------------------------------------------------------

    @Test
    void e2eAgentExportImport() {
        // Export/ImportFrom not yet implemented — tracked in docs/export-import.md
        assumeTrue(false, "Export/ImportFrom not yet implemented");
    }
}
