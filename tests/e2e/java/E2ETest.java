// This file is a reference copy.
// The canonical source compiled by Maven is:
//   bindings/java/src/test/java/com/example/llama/e2e/E2ETest.java
//
// Run via Taskfile:
//   task e2e-java
// Or directly:
//   cd bindings/java && mvn test -Pe2e -Dbridge.lib.dir=../../bridge/build \
//     -DLLAMA_CHAT_MODEL=../../tests/models/qwen2.5-0.5b-instruct-q4_k_m.gguf


import com.example.llama.*;
import com.example.llama.model.*;
import org.junit.jupiter.api.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

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
        // Resolve relative to project root (3 levels up from bindings/java/src/test)
        Path root = Paths.get(System.getProperty("user.dir")).getParent().getParent();
        return root.resolve("tests/models/qwen2.5-0.5b-instruct-q4_k_m.gguf").toString();
    }

    private static String embedModelPath() {
        String env = System.getenv("LLAMA_EMBED_MODEL");
        if (env != null && !env.isBlank()) return env;
        return chatModelPath(); // reuse Qwen model for embedding
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
}
