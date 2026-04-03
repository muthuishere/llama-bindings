package com.example.llama;

import com.example.llama.model.*;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class ChatEngineTest {

    private static final String DUMMY_MODEL = "testdata/dummy.gguf";

    @Test
    void loadInvalidModelPath() {
        assertThrows(LlamaException.class, () ->
                ChatEngine.load("/nonexistent/model.gguf", new LoadOptions()));
    }

    @Test
    void closedEngineThrows() throws Exception {
        ChatEngine chat = loadOrSkip();
        if (chat == null) return;
        chat.close();

        LlamaException ex = assertThrows(LlamaException.class, () ->
                chat.chat(buildTextRequest(), new ChatOptions()));
        assertEquals("ENGINE_CLOSED", ex.getCode());
    }

    @Test
    void closeIsIdempotent() throws Exception {
        ChatEngine chat = loadOrSkip();
        if (chat == null) return;
        chat.close();
        assertDoesNotThrow(chat::close);
    }

    @Test
    void chatTextMode() throws Exception {
        try (ChatEngine chat = loadOrSkip()) {
            if (chat == null) return;
            ChatResponse resp = chat.chat(buildTextRequest(), new ChatOptions());
            assertEquals("assistant_text", resp.type);
            assertNotNull(resp.text);
        }
    }

    @Test
    void chatSchemaModeReturnsStructuredJson() throws Exception {
        try (ChatEngine chat = loadOrSkip()) {
            if (chat == null) return;
            ChatRequest req = ChatRequest.builder()
                    .messages(List.of(
                            ChatMessage.user("John is 32 years old. Extract data.")))
                    .responseMode("json_schema")
                    .schema(java.util.Map.of(
                            "name", "person_extract",
                            "schema", java.util.Map.of("type", "object")))
                    .generation(new GenerationOptions(0.0f, 64, 1.0f, 40))
                    .build();
            ChatResponse resp = chat.chat(req, new ChatOptions());
            assertEquals("structured_json", resp.type);
        }
    }

    @Test
    void chatToolCallModeReturnsToolCall() throws Exception {
        try (ChatEngine chat = loadOrSkip()) {
            if (chat == null) return;
            ChatRequest req = ChatRequest.builder()
                    .messages(List.of(
                            ChatMessage.user("What is the weather in Chennai?")))
                    .responseMode("tool_call")
                    .tools(List.of(new ToolDefinition(
                            "lookup_weather",
                            "Get weather by city",
                            java.util.Map.of("type", "object"))))
                    .toolChoice("auto")
                    .generation(new GenerationOptions(0.2f, 64, 0.95f, 40))
                    .build();
            ChatResponse resp = chat.chat(req, new ChatOptions());
            assertEquals("tool_call", resp.type);
        }
    }

    @Test
    void eventsEmittedDuringLoad() throws Exception {
        java.util.List<String> events = new java.util.ArrayList<>();
        ChatEngine chat = null;
        try {
            chat = ChatEngine.load(DUMMY_MODEL, new LoadOptions(e -> events.add(e.event)));
        } catch (LlamaException ex) {
            // model not available in CI – skip
            return;
        }
        try {
            assertFalse(events.isEmpty(), "Expected at least one load event");
        } finally {
            chat.close();
        }
    }

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    private static ChatEngine loadOrSkip() {
        try {
            return ChatEngine.load(DUMMY_MODEL, new LoadOptions());
        } catch (LlamaException ex) {
            return null; // model not available – skip test
        }
    }

    private static ChatRequest buildTextRequest() {
        return ChatRequest.builder()
                .messages(List.of(ChatMessage.user("Say hello.")))
                .responseMode("text")
                .generation(new GenerationOptions(0.2f, 64, 0.95f, 40))
                .build();
    }
}
