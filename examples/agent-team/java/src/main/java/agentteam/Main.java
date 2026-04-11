package agentteam;

// examples/agent-team/java/src/main/java/agentteam/Main.java
//
// Knowledge Assistant HTTP server — Java example for llama-bindings.
// Uses Spring Boot + NDJSON streaming.
//
// Exposes:
//   POST /chat         — NDJSON streaming: each line is {"token":"..."} then {"done":true}
//   POST /embed        — JSON: {"embedding":[...],"dim":768,"duration_ms":123}
//   POST /chat-schema  — NDJSON streaming with json_schema response mode
//   GET  /             — Chat UI (src/main/resources/static/index.html, auto-served by Spring)
//
// Prerequisites:
//   1. task build-bridge
//   2. task download-model
//   3. cd bindings/java && mvn install -DskipTests -Dbridge.lib.dir=../../bridge/build
//
// Run:
//   cd examples/agent-team/java
//   mvn package -Dbridge.lib.dir=../../../bridge/build
//   java -Djava.library.path=../../../bridge/build \
//        -jar target/agent-team-java-1.0-SNAPSHOT.jar
//   open http://localhost:8081
//
// Or via Taskfile:
//   task example-java

import com.example.llama.ChatEngine;
import com.example.llama.ChatOptions;
import com.example.llama.EmbedEngine;
import com.example.llama.EmbedOptions;
import com.example.llama.LoadOptions;
import com.example.llama.agent.Agent;
import com.example.llama.model.ChatMessage;
import com.example.llama.model.ChatRequest;
import com.example.llama.model.ChatResponse;
import com.example.llama.model.ToolDefinition;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.StreamingResponseBody;

import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@SpringBootApplication
@RestController
public class Main {

    // ----------------------------------------------------------------
    // Request / response records
    // ----------------------------------------------------------------

    record ChatReq(String session, String message) {}
    record EmbedRequest(String text) {}
    record SchemaRequest(String session, String message, Map<String, Object> schema) {}

    // ----------------------------------------------------------------
    // Dependencies
    // ----------------------------------------------------------------

    @Autowired
    private ObjectMapper objectMapper;

    private Agent agent;
    private EmbedEngine embedEngine;
    private ChatEngine chatEngine;

    // ----------------------------------------------------------------
    // Lifecycle
    // ----------------------------------------------------------------

    @PostConstruct
    public void initAgent() throws Exception {
        System.out.println("=== llama-bindings · Agent Team Example (Java / Spring Boot) ===");
        System.out.println("Loading agent...");

        agent = Agent.create(chatModelPath(), embedModelPath(), ":memory:");

        var docs = List.of(
            "llama-bindings is a cross-language library that wraps llama.cpp with a unified API for Go, Java, and Browser JavaScript.",
            "The Go binding uses purego for dynamic loading — no CGO required. Import path: github.com/muthuishere/llama-bindings/go/llama.",
            "The Java binding uses Project Panama FFM (JDK 22) instead of JNI. Artifact: com.example.llama:llama-java.",
            "The Browser JS binding compiles llama.cpp to WebAssembly via Emscripten. Package: @llama-bindings/js-browser.",
            "The Agent layer combines ChatEngine, EmbedEngine, KnowledgeStore, and ToolRegistry into a single orchestrated loop.",
            "Build all targets with: task build. Run all tests with: task test. Download models with: task download-model.",
            "The chat model is Gemma 4 E2B (2.3B effective parameters, native tool calling tokens, Apache 2.0 license).",
            "The embed model is nomic-embed-text-v1.5 (137M parameters, 80 MB Q4_K_M, purpose-built for semantic search)."
        );
        for (var doc : docs) agent.addDocument(doc);

        agent.addTool(
            new ToolDefinition(
                "calculate",
                "Perform basic arithmetic. Supports add, subtract, multiply, divide, sqrt.",
                Map.of(
                    "type", "object",
                    "properties", Map.of(
                        "operation", Map.of(
                            "type", "string",
                            "enum", List.of("add", "subtract", "multiply", "divide", "sqrt"),
                            "description", "The arithmetic operation to perform"
                        ),
                        "a", Map.of("type", "number", "description", "First operand"),
                        "b", Map.of("type", "number", "description", "Second operand (not used for sqrt)")
                    ),
                    "required", List.of("operation", "a"),
                    "additionalProperties", false
                )
            ),
            toolArgs -> {
                String op = (String) toolArgs.get("operation");
                double a = toDouble(toolArgs.get("a"));
                double b = toDouble(toolArgs.getOrDefault("b", 0.0));
                double result = switch (op) {
                    case "add"      -> a + b;
                    case "subtract" -> a - b;
                    case "multiply" -> a * b;
                    case "divide"   -> {
                        if (b == 0) throw new ArithmeticException("division by zero");
                        yield a / b;
                    }
                    case "sqrt" -> Math.sqrt(a);
                    default -> throw new IllegalArgumentException("unknown operation: " + op);
                };
                return Map.of("result", result);
            }
        );

        System.out.println("Agent ready. 8 documents loaded. Calculator tool registered.");

        System.out.println("Loading EmbedEngine...");
        embedEngine = EmbedEngine.load(embedModelPath(), new LoadOptions());
        System.out.println("EmbedEngine ready.");

        System.out.println("Loading ChatEngine...");
        chatEngine = ChatEngine.load(chatModelPath(), new LoadOptions());
        System.out.println("ChatEngine ready.");

        System.out.println("Listening on http://localhost:8081");
    }

    @PreDestroy
    public void closeAgent() {
        if (embedEngine != null) {
            embedEngine.close();
            System.out.println("EmbedEngine closed.");
        }
        if (chatEngine != null) {
            chatEngine.close();
            System.out.println("ChatEngine closed.");
        }
        if (agent != null) {
            agent.close();
            System.out.println("Agent closed.");
        }
    }

    // ----------------------------------------------------------------
    // Endpoints
    // ----------------------------------------------------------------

    @PostMapping(value = "/chat", produces = "application/x-ndjson")
    public StreamingResponseBody chat(@RequestBody ChatReq req) {
        return (OutputStream out) -> {
            try {
                String reply = agent.chat(req.session(), req.message());
                String[] parts = reply.split(" ", -1);

                for (int i = 0; i < parts.length; i++) {
                    String token = parts[i] + (i < parts.length - 1 ? " " : "");
                    writeNDJSON(out, Map.of("token", token));
                    Thread.sleep(25);
                }
                writeNDJSON(out, Map.of("done", true, "reply", reply));

            } catch (Exception e) {
                try {
                    writeNDJSON(out, Map.of("error", e.getMessage() != null ? e.getMessage() : "unknown error", "done", true));
                } catch (Exception ignored) { }
            }
        };
    }

    @PostMapping(value = "/embed", produces = MediaType.APPLICATION_JSON_VALUE)
    public Map<String, Object> embed(@RequestBody EmbedRequest req) {
        try {
            long start = System.currentTimeMillis();
            float[] vec = embedEngine.embed(req.text(), new EmbedOptions());
            long duration = System.currentTimeMillis() - start;

            // Convert float[] to List<Float> for JSON serialisation
            List<Float> embedding = new ArrayList<>(vec.length);
            for (float v : vec) embedding.add(v);

            return Map.of(
                "embedding", embedding,
                "dim",       vec.length,
                "duration_ms", duration
            );
        } catch (Exception e) {
            return Map.of("error", e.getMessage() != null ? e.getMessage() : "unknown error");
        }
    }

    @PostMapping(value = "/chat-schema", produces = "application/x-ndjson")
    public StreamingResponseBody chatSchema(@RequestBody SchemaRequest req) {
        return (OutputStream out) -> {
            try {
                ChatRequest chatReq = ChatRequest.builder()
                        .messages(List.of(ChatMessage.user(req.message())))
                        .responseMode("json_schema")
                        .schema(req.schema())
                        .build();

                ChatResponse resp = chatEngine.chat(chatReq, new ChatOptions());

                // The response text (or json) is streamed word-by-word as NDJSON
                String reply = resp.text != null ? resp.text
                        : (resp.json != null ? objectMapper.writeValueAsString(resp.json) : "");
                String[] parts = reply.split(" ", -1);

                for (int i = 0; i < parts.length; i++) {
                    String token = parts[i] + (i < parts.length - 1 ? " " : "");
                    writeNDJSON(out, Map.of("token", token));
                    Thread.sleep(25);
                }
                writeNDJSON(out, Map.of("done", true, "reply", reply));

            } catch (Exception e) {
                try {
                    writeNDJSON(out, Map.of("error", e.getMessage() != null ? e.getMessage() : "unknown error", "done", true));
                } catch (Exception ignored) { }
            }
        };
    }

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    private void writeNDJSON(OutputStream out, Object value) throws Exception {
        String line = objectMapper.writeValueAsString(value) + "\n";
        out.write(line.getBytes(StandardCharsets.UTF_8));
        out.flush();
    }

    private static double toDouble(Object v) {
        if (v instanceof Double d)  return d;
        if (v instanceof Float f)   return f.doubleValue();
        if (v instanceof Integer i) return i.doubleValue();
        if (v instanceof Long l)    return l.doubleValue();
        if (v instanceof String s)  return Double.parseDouble(s);
        return 0.0;
    }

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

    // ----------------------------------------------------------------
    // Entry point
    // ----------------------------------------------------------------

    public static void main(String[] args) {
        SpringApplication.run(Main.class, args);
    }
}
