package com.example.llama;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

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

    // ---------------------------------------------------------------------------
    // Integration tests — require llama.test.model system property
    // ---------------------------------------------------------------------------

    private static final String MODEL_PROP = "llama.test.model";

    /** Returns model path or skips the test if not configured. */
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
            // Must not throw or crash.
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
        // Second close should not throw.
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
}
