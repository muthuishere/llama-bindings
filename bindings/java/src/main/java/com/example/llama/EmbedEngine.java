package com.example.llama;

import java.lang.foreign.*;
import java.nio.charset.StandardCharsets;

/**
 * Embedding engine: loads a GGUF embedding model and generates float vectors.
 *
 * <p>Usage:
 * <pre>{@code
 * try (var embed = EmbedEngine.load("embed-model.gguf", new LoadOptions(listener))) {
 *     float[] vec = embed.embed("hello world", new EmbedOptions(null));
 * }
 * }</pre>
 */
public final class EmbedEngine implements AutoCloseable {

    private final Arena         arena;
    private final MemorySegment handle;
    private volatile boolean    closed;

    private EmbedEngine(Arena arena, MemorySegment handle) {
        this.arena  = arena;
        this.handle = handle;
    }

    /**
     * Load an embedding engine from the given GGUF model file.
     *
     * @param modelPath absolute or relative path to the GGUF file
     * @param opts      load options (listener may be null)
     * @return a ready-to-use {@code EmbedEngine}
     * @throws LlamaException if the model cannot be loaded
     */
    public static EmbedEngine load(String modelPath, LoadOptions opts)
            throws LlamaException {
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment cPath = arena.allocateFrom(modelPath, StandardCharsets.UTF_8);

            MemorySegment cbStub   = MemorySegment.NULL;
            MemorySegment userData = MemorySegment.NULL;
            if (opts.listener != null) {
                cbStub = NativeLibrary.LINKER.upcallStub(
                        buildEventHandle(opts.listener),
                        NativeLibrary.EVENT_CB_DESCRIPTOR,
                        arena);
            }

            MemorySegment handle = (MemorySegment) NativeLibrary.EMBED_CREATE
                    .invoke(cPath, cbStub, userData);

            if (handle == null || handle.equals(MemorySegment.NULL)) {
                arena.close();
                throw new LlamaException("MODEL_LOAD_FAILED",
                        "llama_embed_create returned NULL for: " + modelPath);
            }

            return new EmbedEngine(arena, handle);
        } catch (LlamaException e) {
            throw e;
        } catch (Throwable t) {
            arena.close();
            throw new LlamaException("INTERNAL_BRIDGE_ERROR", t.getMessage());
        }
    }

    /**
     * Generate an embedding vector for the given text.
     *
     * @param text input text (must not be empty)
     * @param opts per-call options
     * @return float array of the embedding vector
     * @throws LlamaException if the engine is closed or inference fails
     */
    public float[] embed(String text, EmbedOptions opts) throws LlamaException {
        if (closed) {
            throw new LlamaException("ENGINE_CLOSED", "EmbedEngine is closed");
        }
        if (text == null || text.isEmpty()) {
            throw new LlamaException("INVALID_REQUEST", "input text must not be empty");
        }

        try (Arena callArena = Arena.ofConfined()) {
            MemorySegment cText   = callArena.allocateFrom(text, StandardCharsets.UTF_8);
            MemorySegment outLen  = callArena.allocate(ValueLayout.JAVA_INT);
            MemorySegment cVec    = (MemorySegment) NativeLibrary.EMBED_INFER
                    .invoke(handle, cText, outLen);

            if (cVec == null || cVec.equals(MemorySegment.NULL)) {
                throw new LlamaException("INFERENCE_FAILED", "bridge returned NULL vector");
            }

            int len = outLen.get(ValueLayout.JAVA_INT, 0);
            if (len <= 0) {
                NativeLibrary.FLOAT_FREE.invoke(cVec);
                throw new LlamaException("INFERENCE_FAILED", "bridge returned empty vector");
            }

            MemorySegment floatArr = cVec.reinterpret((long) len * Float.BYTES);
            float[] result = new float[len];
            for (int i = 0; i < len; i++) {
                result[i] = floatArr.get(ValueLayout.JAVA_FLOAT, (long) i * Float.BYTES);
            }
            NativeLibrary.FLOAT_FREE.invoke(cVec);
            return result;
        } catch (LlamaException e) {
            throw e;
        } catch (Throwable t) {
            throw new LlamaException("INTERNAL_BRIDGE_ERROR", t.getMessage());
        }
    }

    /** Release the native engine. Safe to call multiple times. */
    @Override
    public void close() {
        if (closed) return;
        closed = true;
        try {
            NativeLibrary.EMBED_DESTROY.invoke(handle);
        } catch (Throwable ignored) {
        } finally {
            arena.close();
        }
    }

    // ----------------------------------------------------------------
    // Internal
    // ----------------------------------------------------------------

    private static java.lang.invoke.MethodHandle buildEventHandle(
            EventListener listener) throws Exception {
        return java.lang.invoke.MethodHandles.lookup()
                .findVirtual(EventListener.class, "onEventJson",
                        java.lang.invoke.MethodType.methodType(
                                void.class, String.class, MemorySegment.class))
                .bindTo(listener);
    }
}
