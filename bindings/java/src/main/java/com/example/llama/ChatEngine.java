package com.example.llama;

import com.example.llama.model.ChatRequest;
import com.example.llama.model.ChatResponse;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.lang.foreign.*;
import java.nio.charset.StandardCharsets;

/**
 * Chat engine: loads a GGUF model and provides chat inference.
 *
 * <p>Usage:
 * <pre>{@code
 * try (var chat = ChatEngine.load("chat-model.gguf", new LoadOptions(listener))) {
 *     var resp = chat.chat(request, new ChatOptions(0.2f, 256, 0.95f, 40, null));
 * }
 * }</pre>
 *
 * <p>Internally this class uses Project Panama FFM (JDK 21+) to call the
 * native bridge. No Panama types are exposed in the public API.
 */
public final class ChatEngine implements AutoCloseable {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final Arena         arena;
    private final MemorySegment handle;
    private volatile boolean    closed;

    private ChatEngine(Arena arena, MemorySegment handle) {
        this.arena  = arena;
        this.handle = handle;
    }

    /**
     * Load a chat engine from the given GGUF model file.
     *
     * @param modelPath absolute or relative path to the GGUF file
     * @param opts      load options (listener may be null)
     * @return a ready-to-use {@code ChatEngine}
     * @throws LlamaException if the model cannot be loaded
     */
    public static ChatEngine load(String modelPath, LoadOptions opts)
            throws LlamaException {
        NativeLibrary.checkAvailable();
        Arena arena = Arena.ofConfined();
        try {
            MemorySegment cPath = arena.allocateFrom(modelPath, StandardCharsets.UTF_8);

            MemorySegment cbStub    = MemorySegment.NULL;
            MemorySegment userData  = MemorySegment.NULL;
            if (opts.listener != null) {
                cbStub = NativeLibrary.LINKER.upcallStub(
                        buildEventHandle(opts.listener),
                        NativeLibrary.EVENT_CB_DESCRIPTOR,
                        arena);
            }

            MemorySegment handle = (MemorySegment) NativeLibrary.CHAT_CREATE
                    .invoke(cPath, cbStub, userData);

            if (handle == null || handle.equals(MemorySegment.NULL)) {
                arena.close();
                throw new LlamaException("MODEL_LOAD_FAILED",
                        "llama_chat_create returned NULL for: " + modelPath);
            }

            return new ChatEngine(arena, handle);
        } catch (LlamaException e) {
            throw e;
        } catch (Throwable t) {
            arena.close();
            throw new LlamaException("INTERNAL_BRIDGE_ERROR", t.getMessage());
        }
    }

    /**
     * Run a chat inference request.
     *
     * @param request the chat request
     * @param opts    per-call options
     * @return the normalized response
     * @throws LlamaException if the engine is closed or inference fails
     */
    public ChatResponse chat(ChatRequest request, ChatOptions opts)
            throws LlamaException {
        if (closed) {
            throw new LlamaException("ENGINE_CLOSED", "ChatEngine is closed");
        }

        try {
            // Merge generation options from ChatOptions into the request.
            if (request.generation == null) {
                request.generation = new com.example.llama.model.GenerationOptions(
                        opts.temperature, opts.maxOutputTokens, opts.topP, opts.topK);
            }
            if (request.responseMode == null) {
                request.responseMode = "text";
            }

            String reqJson = MAPPER.writeValueAsString(request);

            try (Arena callArena = Arena.ofConfined()) {
                MemorySegment cReq  = callArena.allocateFrom(reqJson, StandardCharsets.UTF_8);
                MemorySegment cResp = (MemorySegment) NativeLibrary.CHAT_INFER
                        .invoke(handle, cReq);

                if (cResp == null || cResp.equals(MemorySegment.NULL)) {
                    throw new LlamaException("INFERENCE_FAILED", "bridge returned NULL");
                }

                String respJson = NativeLibrary.readCString(cResp);
                NativeLibrary.STRING_FREE.invoke(cResp);

                ChatResponse resp = MAPPER.readValue(respJson, ChatResponse.class);
                if (resp.isError()) {
                    throw new LlamaException(
                            resp.error != null ? resp.error.code : "INFERENCE_FAILED",
                            resp.error != null ? resp.error.message : "unknown error");
                }
                return resp;
            }
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
            NativeLibrary.CHAT_DESTROY.invoke(handle);
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
                                void.class, MemorySegment.class, MemorySegment.class))
                .bindTo(listener);
    }
}
