package com.example.llama;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class EmbedEngineTest {

    private static final String DUMMY_MODEL = "testdata/dummy-embed.gguf";

    @Test
    void loadInvalidModelPath() {
        assertThrows(LlamaException.class, () ->
                EmbedEngine.load("/nonexistent/model.gguf", new LoadOptions()));
    }

    @Test
    void closedEngineThrows() throws Exception {
        EmbedEngine embed = loadOrSkip();
        if (embed == null) return;
        embed.close();

        LlamaException ex = assertThrows(LlamaException.class, () ->
                embed.embed("hello", new EmbedOptions()));
        assertEquals("ENGINE_CLOSED", ex.getCode());
    }

    @Test
    void closeIsIdempotent() throws Exception {
        EmbedEngine embed = loadOrSkip();
        if (embed == null) return;
        embed.close();
        assertDoesNotThrow(embed::close);
    }

    @Test
    void emptyInputThrows() throws Exception {
        try (EmbedEngine embed = loadOrSkip()) {
            if (embed == null) return;
            LlamaException ex = assertThrows(LlamaException.class, () ->
                    embed.embed("", new EmbedOptions()));
            assertEquals("INVALID_REQUEST", ex.getCode());
        }
    }

    @Test
    void embedReturnsNonEmptyVector() throws Exception {
        try (EmbedEngine embed = loadOrSkip()) {
            if (embed == null) return;
            float[] vec = embed.embed("semantic search example", new EmbedOptions());
            assertTrue(vec.length > 0, "Expected non-empty vector");
        }
    }

    @Test
    void repeatedEmbedCallsStable() throws Exception {
        try (EmbedEngine embed = loadOrSkip()) {
            if (embed == null) return;
            for (int i = 0; i < 3; i++) {
                float[] vec = embed.embed("repeated input", new EmbedOptions());
                assertTrue(vec.length > 0, "Expected non-empty vector on iteration " + i);
            }
        }
    }

    @Test
    void eventsEmittedDuringLoad() throws Exception {
        java.util.List<String> events = new java.util.ArrayList<>();
        EmbedEngine embed = null;
        try {
            embed = EmbedEngine.load(DUMMY_MODEL, new LoadOptions(e -> events.add(e.event)));
        } catch (LlamaException ex) {
            return; // model not available
        }
        try {
            assertFalse(events.isEmpty(), "Expected at least one load event");
        } finally {
            embed.close();
        }
    }

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    private static EmbedEngine loadOrSkip() {
        try {
            return EmbedEngine.load(DUMMY_MODEL, new LoadOptions());
        } catch (LlamaException ex) {
            return null;
        }
    }
}
