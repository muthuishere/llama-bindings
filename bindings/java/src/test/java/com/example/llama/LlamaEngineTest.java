package com.example.llama;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link LlamaEngine}.
 *
 * <p>Integration tests that require a real model are gated on the
 * {@code llama.test.model} system property.  Run them with:
 * <pre>
 *   mvn test -Dllama.test.model=/path/to/model.gguf
 * </pre>
 */
class LlamaEngineTest {

    // ---------------------------------------------------------------------------
    // Unit-style tests — no native library required
    // ---------------------------------------------------------------------------

    @Test
    void load_throwsOnNullPath() {
        assertThrows(IllegalArgumentException.class,
                () -> LlamaEngine.load(null));
    }

    @Test
    void load_throwsOnEmptyPath() {
        assertThrows(IllegalArgumentException.class,
                () -> LlamaEngine.load(""));
    }

    @Test
    void load_throwsOnBlankPath() {
        assertThrows(IllegalArgumentException.class,
                () -> LlamaEngine.load("   "));
    }

    @Test
    void message_throwsOnNullRole() {
        assertThrows(IllegalArgumentException.class,
                () -> new LlamaEngine.Message(null, "hi"));
    }

    @Test
    void message_factoryMethods() {
        LlamaEngine.Message sys  = LlamaEngine.Message.system("sys");
        LlamaEngine.Message usr  = LlamaEngine.Message.user("usr");
        LlamaEngine.Message asst = LlamaEngine.Message.assistant("asst");
        assertEquals("system",    sys.role);
        assertEquals("user",      usr.role);
        assertEquals("assistant", asst.role);
    }

    // ---------------------------------------------------------------------------
    // Integration tests — require llama.test.model system property
    // ---------------------------------------------------------------------------

    private static final String MODEL_PROP = "llama.test.model";

    private static String requireModelPath() {
        String path = System.getProperty(MODEL_PROP);
        org.junit.jupiter.api.Assumptions.assumeTrue(
                path != null && !path.isBlank(),
                MODEL_PROP + " not set — skipping integration test");
        return path;
    }

    @Test
    void smoke_loadCompleteClose() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            String result = engine.complete("Say hello in one short sentence.");
            assertNotNull(result, "completion must not be null");
            assertFalse(result.isBlank(), "completion must not be blank");
        }
    }

    @Test
    void factual_completionIsNonEmpty() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            String result = engine.complete("Complete this: The capital of France is");
            assertFalse(result.isBlank(), "factual completion must not be blank");
        }
    }

    @Test
    void emptyPrompt_doesNotCrash() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            engine.complete("");
        }
    }

    @Test
    void repeatedCompletions_noCorruption() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            for (int i = 0; i < 5; i++) {
                String result = engine.complete("Say hello.");
                assertFalse(result.isBlank(),
                        "completion " + i + " must not be blank");
            }
        }
    }

    @Test
    void invalidModelPath_throwsRuntimeException() {
        assertThrows(RuntimeException.class,
                () -> LlamaEngine.load("/nonexistent/model.gguf"));
    }

    @Test
    void close_isIdempotent() {
        String model = requireModelPath();
        LlamaEngine engine = LlamaEngine.load(model);
        engine.close();
        assertDoesNotThrow(engine::close);
    }

    @Test
    void complete_afterClose_throwsIllegalStateException() {
        String model = requireModelPath();
        LlamaEngine engine = LlamaEngine.load(model);
        engine.close();
        assertThrows(IllegalStateException.class,
                () -> engine.complete("hello"));
    }

    @Test
    void createDestroyCycle_doesNotCrash() {
        String model = requireModelPath();
        for (int i = 0; i < 3; i++) {
            try (LlamaEngine engine = LlamaEngine.load(model)) {
                assertNotNull(engine);
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Chat integration tests
    // ---------------------------------------------------------------------------

    @Test
    void chat_withSystem_returnsNonEmpty() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            String result = engine.chat("You are a helpful assistant.", "Say hello.");
            assertFalse(result.isBlank(), "chat response must not be blank");
        }
    }

    @Test
    void chat_noSystem_returnsNonEmpty() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            String result = engine.chat("", "Say hello.");
            assertFalse(result.isBlank(), "chat (no system) response must not be blank");
        }
    }

    @Test
    void chat_afterClose_throwsIllegalStateException() {
        String model = requireModelPath();
        LlamaEngine engine = LlamaEngine.load(model);
        engine.close();
        assertThrows(IllegalStateException.class,
                () -> engine.chat("sys", "user"));
    }

    @Test
    void chatWithMessages_returnsNonEmpty() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            List<LlamaEngine.Message> msgs = List.of(
                    LlamaEngine.Message.system("You are a helpful assistant."),
                    LlamaEngine.Message.user("Say hello in one sentence.")
            );
            String result = engine.chatWithMessages(msgs);
            assertFalse(result.isBlank(), "chatWithMessages response must not be blank");
        }
    }

    @Test
    void chatWithMessages_throwsOnEmptyList() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            assertThrows(IllegalArgumentException.class,
                    () -> engine.chatWithMessages(List.of()));
        }
    }

    @Test
    void chatSession_multiTurn_doesNotCrash() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            String turn1 = engine.chatSession("sid-java-1", "Say hello.");
            assertFalse(turn1.isBlank(), "turn 1 must not be blank");

            String turn2 = engine.chatSession("sid-java-1", "What did you just say?");
            assertFalse(turn2.isBlank(), "turn 2 must not be blank");
        }
    }

    @Test
    void chatSession_setSystem_doesNotCrash() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            engine.chatSessionSetSystem("sid-java-sys", "You are a helpful assistant.");
            String result = engine.chatSession("sid-java-sys", "Say hello.");
            assertFalse(result.isBlank(), "session with system response must not be blank");
        }
    }

    @Test
    void chatSession_clearThenContinue_doesNotCrash() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            engine.chatSession("sid-java-clear", "Say hello.");
            engine.chatSessionClear("sid-java-clear");
            String result = engine.chatSession("sid-java-clear", "Say hello again.");
            assertFalse(result.isBlank(), "response after clear must not be blank");
        }
    }

    @Test
    void chatSession_throwsOnBlankSessionId() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            assertThrows(IllegalArgumentException.class,
                    () -> engine.chatSession("", "hello"));
        }
    }

    @Test
    void chatWithTools_returnsRawOutput() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            String tools = "[{\"name\":\"get_weather\",\"description\":\"Get the weather\","
                    + "\"parameters\":{\"location\":{\"type\":\"string\"}}}]";
            List<LlamaEngine.Message> msgs = List.of(
                    LlamaEngine.Message.user("What is the weather in Paris?")
            );
            String result = engine.chatWithTools(msgs, tools);
            assertFalse(result.isBlank(), "chatWithTools response must not be blank");
        }
    }

    @Test
    void chatWithTools_throwsOnEmptyMessages() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            assertThrows(IllegalArgumentException.class,
                    () -> engine.chatWithTools(List.of(), "[]"));
        }
    }
}
