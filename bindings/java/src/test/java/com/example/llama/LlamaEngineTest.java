package com.example.llama;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

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
        assertThrows(IllegalArgumentException.class, () -> LlamaEngine.load(null));
    }

    @Test
    void load_throwsOnEmptyPath() {
        assertThrows(IllegalArgumentException.class, () -> LlamaEngine.load(""));
    }

    @Test
    void load_throwsOnBlankPath() {
        assertThrows(IllegalArgumentException.class, () -> LlamaEngine.load("   "));
    }

    @Test
    void chatMessage_constructor_setsFields() {
        LlamaEngine.ChatMessage msg = new LlamaEngine.ChatMessage("user", "hello");
        assertEquals("user",  msg.role);
        assertEquals("hello", msg.content);
    }

    @Test
    void chatMessage_nullsBecomeEmptyStrings() {
        LlamaEngine.ChatMessage msg = new LlamaEngine.ChatMessage(null, null);
        assertEquals("", msg.role);
        assertEquals("", msg.content);
    }

    @Test
    void jsonString_parsesSimpleString() {
        String json = "{\"role\":\"assistant\",\"content\":\"Hello!\"}";
        assertEquals("assistant", LlamaEngine.jsonString(json, "role"));
        assertEquals("Hello!",    LlamaEngine.jsonString(json, "content"));
    }

    @Test
    void jsonString_handlesEscapeSequences() {
        String json = "{\"content\":\"line1\\nline2\"}";
        assertEquals("line1\nline2", LlamaEngine.jsonString(json, "content"));
    }

    @Test
    void jsonInt_parsesInteger() {
        String json = "{\"messageCount\":7}";
        assertEquals(7, LlamaEngine.jsonInt(json, "messageCount"));
    }

    @Test
    void jsonInt_returnZeroForMissingField() {
        String json = "{\"other\":\"value\"}";
        assertEquals(0, LlamaEngine.jsonInt(json, "messageCount"));
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
            assertFalse(result.isBlank(), "completion must not be blank");
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
                assertFalse(result.isBlank(), "completion " + i + " must not be blank");
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
        assertThrows(IllegalStateException.class, () -> engine.complete("hello"));
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
    void chat_withSystem_returnsChatMessage() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            LlamaEngine.ChatMessage msg = engine.chat("sid-java-1", List.of(
                    new LlamaEngine.ChatMessage("system", "You are a helpful assistant."),
                    new LlamaEngine.ChatMessage("user",   "Say hello.")));
            assertEquals("assistant", msg.role);
            assertFalse(msg.content.isBlank(), "content must not be blank");
        }
    }

    @Test
    void chat_noSystem_returnsChatMessage() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            LlamaEngine.ChatMessage msg = engine.chat("sid-java-2", List.of(
                    new LlamaEngine.ChatMessage("user", "Say hello.")));
            assertEquals("assistant", msg.role);
            assertFalse(msg.content.isBlank(), "content must not be blank");
        }
    }

    @Test
    void chat_multiTurn_maintainsHistory() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            String sid = "sid-java-mt";
            LlamaEngine.ChatMessage t1 = engine.chat(sid, List.of(
                    new LlamaEngine.ChatMessage("system", "You are helpful."),
                    new LlamaEngine.ChatMessage("user",   "Say hello.")));
            assertFalse(t1.content.isBlank(), "turn 1 must not be blank");

            LlamaEngine.ChatMessage t2 = engine.chat(sid, List.of(
                    new LlamaEngine.ChatMessage("user", "What did you just say?")));
            assertFalse(t2.content.isBlank(), "turn 2 must not be blank");
        }
    }

    @Test
    void chat_withAssistantMessage_doesNotCrash() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            LlamaEngine.ChatMessage msg = engine.chat("sid-java-asst", List.of(
                    new LlamaEngine.ChatMessage("system",    "You are helpful."),
                    new LlamaEngine.ChatMessage("assistant", "I said hello earlier."),
                    new LlamaEngine.ChatMessage("user",      "What did you say before?")));
            assertFalse(msg.content.isBlank(), "content must not be blank");
        }
    }

    @Test
    void chat_withToolMessage_doesNotCrash() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            LlamaEngine.ChatMessage msg = engine.chat("sid-java-tool", List.of(
                    new LlamaEngine.ChatMessage("system", "You are a helpful assistant."),
                    new LlamaEngine.ChatMessage("tool",   "{\"weather\":\"sunny\",\"temp\":\"22C\"}"),
                    new LlamaEngine.ChatMessage("user",   "What is the weather?")));
            assertFalse(msg.content.isBlank(), "content must not be blank");
        }
    }

    @Test
    void chat_afterClose_throwsIllegalStateException() {
        String model = requireModelPath();
        LlamaEngine engine = LlamaEngine.load(model);
        engine.close();
        assertThrows(IllegalStateException.class,
                () -> engine.chat("sid", List.of(new LlamaEngine.ChatMessage("user", "hello"))));
    }

    @Test
    void chat_blankSessionId_throwsIllegalArgumentException() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            assertThrows(IllegalArgumentException.class,
                    () -> engine.chat("", List.of(new LlamaEngine.ChatMessage("user", "hello"))));
        }
    }

    @Test
    void chatWithObject_returnsJsonObject() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            String sid = "sid-java-obj";
            Map<String, Object> resp = engine.chatWithObject(sid, List.of(
                    new LlamaEngine.ChatMessage("system", "You are helpful."),
                    new LlamaEngine.ChatMessage("user",   "Say hello.")));
            assertEquals("assistant", resp.get("role"));
            assertFalse(((String) resp.get("content")).isBlank(), "content must not be blank");
            assertEquals(sid, resp.get("sessionId"));
            assertTrue((Integer) resp.get("messageCount") > 0, "messageCount must be > 0");
        }
    }

    @Test
    void chatWithObject_multiTurn_messageCountGrows() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            String sid = "sid-java-obj-mt";
            Map<String, Object> r1 = engine.chatWithObject(sid, List.of(
                    new LlamaEngine.ChatMessage("user", "Hello.")));
            Map<String, Object> r2 = engine.chatWithObject(sid, List.of(
                    new LlamaEngine.ChatMessage("user", "How are you?")));
            assertTrue((Integer) r2.get("messageCount") > (Integer) r1.get("messageCount"),
                    "messageCount must grow: r1=" + r1.get("messageCount")
                            + " r2=" + r2.get("messageCount"));
        }
    }

    @Test
    void chatSessionClear_resetsHistory() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            String sid = "sid-java-clear";
            engine.chat(sid, List.of(new LlamaEngine.ChatMessage("user", "Say hello.")));
            engine.chatSessionClear(sid);

            LlamaEngine.ChatMessage msg = engine.chat(sid, List.of(
                    new LlamaEngine.ChatMessage("user", "Say hello again.")));
            assertFalse(msg.content.isBlank(), "content after clear must not be blank");
        }
    }

    @Test
    void chatWithObject_afterClose_throwsIllegalStateException() {
        String model = requireModelPath();
        LlamaEngine engine = LlamaEngine.load(model);
        engine.close();
        assertThrows(IllegalStateException.class,
                () -> engine.chatWithObject("sid",
                        List.of(new LlamaEngine.ChatMessage("user", "hello"))));
    }
}
