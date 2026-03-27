package com.example.llama;

import org.junit.jupiter.api.Test;

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
    void chatRequest_nullFieldsBecomeEmpty() {
        LlamaEngine.ChatRequest req = new LlamaEngine.ChatRequest(null, null, null, null);
        assertEquals("", req.systemMessage);
        assertEquals("", req.userMessage);
        assertEquals("", req.assistantMessage);
        assertEquals("", req.toolMessage);
    }

    @Test
    void chatRequest_convenienceConstructors() {
        LlamaEngine.ChatRequest r1 = new LlamaEngine.ChatRequest("hello");
        assertEquals("hello", r1.userMessage);
        assertEquals("", r1.systemMessage);

        LlamaEngine.ChatRequest r2 = new LlamaEngine.ChatRequest("sys", "usr");
        assertEquals("sys", r2.systemMessage);
        assertEquals("usr", r2.userMessage);
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
            LlamaEngine.ChatMessage msg = engine.chat("sid-java-1",
                    new LlamaEngine.ChatRequest("You are a helpful assistant.", "Say hello."));
            assertEquals("assistant", msg.role);
            assertFalse(msg.content.isBlank(), "content must not be blank");
        }
    }

    @Test
    void chat_noSystem_returnsChatMessage() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            LlamaEngine.ChatMessage msg = engine.chat("sid-java-2",
                    new LlamaEngine.ChatRequest("Say hello."));
            assertEquals("assistant", msg.role);
            assertFalse(msg.content.isBlank(), "content must not be blank");
        }
    }

    @Test
    void chat_multiTurn_maintainsHistory() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            String sid = "sid-java-mt";
            LlamaEngine.ChatMessage t1 = engine.chat(sid,
                    new LlamaEngine.ChatRequest("You are helpful.", "Say hello."));
            assertFalse(t1.content.isBlank(), "turn 1 must not be blank");

            LlamaEngine.ChatMessage t2 = engine.chat(sid,
                    new LlamaEngine.ChatRequest("What did you just say?"));
            assertFalse(t2.content.isBlank(), "turn 2 must not be blank");
        }
    }

    @Test
    void chat_withAssistantMessage_doesNotCrash() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            LlamaEngine.ChatMessage msg = engine.chat("sid-java-asst",
                    new LlamaEngine.ChatRequest(
                            "You are helpful.",
                            "What did you say before?",
                            "I said hello earlier.",
                            null));
            assertFalse(msg.content.isBlank(), "content must not be blank");
        }
    }

    @Test
    void chat_withToolMessage_doesNotCrash() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            LlamaEngine.ChatMessage msg = engine.chat("sid-java-tool",
                    new LlamaEngine.ChatRequest(
                            "You are a helpful assistant.",
                            "What is the weather?",
                            null,
                            "{\"weather\":\"sunny\",\"temp\":\"22C\"}"));
            assertFalse(msg.content.isBlank(), "content must not be blank");
        }
    }

    @Test
    void chat_afterClose_throwsIllegalStateException() {
        String model = requireModelPath();
        LlamaEngine engine = LlamaEngine.load(model);
        engine.close();
        assertThrows(IllegalStateException.class,
                () -> engine.chat("sid", new LlamaEngine.ChatRequest("hello")));
    }

    @Test
    void chat_blankSessionId_throwsIllegalArgumentException() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            assertThrows(IllegalArgumentException.class,
                    () -> engine.chat("", new LlamaEngine.ChatRequest("hello")));
        }
    }

    @Test
    void chatWithObject_returnsSchemaResponse() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            String sid = "sid-java-obj";
            LlamaEngine.ChatResponse resp = engine.chatWithObject(sid,
                    new LlamaEngine.ChatRequest("You are helpful.", "Say hello."));
            assertEquals("assistant", resp.role);
            assertFalse(resp.content.isBlank(), "content must not be blank");
            assertEquals(sid, resp.sessionId);
            assertTrue(resp.messageCount > 0, "messageCount must be > 0");
        }
    }

    @Test
    void chatWithObject_multiTurn_messageCountGrows() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            String sid = "sid-java-obj-mt";
            LlamaEngine.ChatResponse r1 = engine.chatWithObject(sid,
                    new LlamaEngine.ChatRequest("Hello."));
            LlamaEngine.ChatResponse r2 = engine.chatWithObject(sid,
                    new LlamaEngine.ChatRequest("How are you?"));
            assertTrue(r2.messageCount > r1.messageCount,
                    "messageCount must grow: r1=" + r1.messageCount + " r2=" + r2.messageCount);
        }
    }

    @Test
    void chatSessionClear_resetsHistory() {
        String model = requireModelPath();
        try (LlamaEngine engine = LlamaEngine.load(model)) {
            String sid = "sid-java-clear";
            engine.chat(sid, new LlamaEngine.ChatRequest("Say hello."));
            engine.chatSessionClear(sid);

            LlamaEngine.ChatMessage msg = engine.chat(sid,
                    new LlamaEngine.ChatRequest("Say hello again."));
            assertFalse(msg.content.isBlank(), "content after clear must not be blank");
        }
    }

    @Test
    void chatWithObject_afterClose_throwsIllegalStateException() {
        String model = requireModelPath();
        LlamaEngine engine = LlamaEngine.load(model);
        engine.close();
        assertThrows(IllegalStateException.class,
                () -> engine.chatWithObject("sid", new LlamaEngine.ChatRequest("hello")));
    }
}
