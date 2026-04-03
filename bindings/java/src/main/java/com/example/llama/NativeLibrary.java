package com.example.llama;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Path;

/**
 * Low-level JVM ↔ native bridge via Project Panama FFM (JDK 21+).
 *
 * <p>This class is internal to the library. Application code should use
 * {@link ChatEngine} and {@link EmbedEngine} exclusively.
 */
final class NativeLibrary {

    private NativeLibrary() {}

    // ----------------------------------------------------------------
    // Library loading
    // ----------------------------------------------------------------

    static {
        // Allow the bridge library path to be supplied via a system property,
        // e.g. -Djava.library.path=/path/to/bridge/build
        System.loadLibrary("llama_bridge");
    }

    static final Linker LINKER = Linker.nativeLinker();
    static final SymbolLookup SYMBOLS = SymbolLookup.loaderLookup()
            .or(LINKER.defaultLookup());

    // ----------------------------------------------------------------
    // Method handles
    // ----------------------------------------------------------------

    /** void* llama_chat_create(const char* model_path, void* cb, void* user_data) */
    static final MethodHandle CHAT_CREATE = lookupOrNull("llama_chat_create",
            FunctionDescriptor.of(
                    ValueLayout.ADDRESS,        // return: engine handle
                    ValueLayout.ADDRESS,        // model_path
                    ValueLayout.ADDRESS,        // on_event callback
                    ValueLayout.ADDRESS         // user_data
            ));

    /** char* llama_chat_infer_json(void* engine, const char* request_json) */
    static final MethodHandle CHAT_INFER = lookupOrNull("llama_chat_infer_json",
            FunctionDescriptor.of(
                    ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS
            ));

    /** void llama_chat_destroy(void* engine) */
    static final MethodHandle CHAT_DESTROY = lookupOrNull("llama_chat_destroy",
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

    /** void* llama_embed_create(const char* model_path, void* cb, void* user_data) */
    static final MethodHandle EMBED_CREATE = lookupOrNull("llama_embed_create",
            FunctionDescriptor.of(
                    ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS
            ));

    /** float* llama_embed_infer(void* engine, const char* input, int* out_len) */
    static final MethodHandle EMBED_INFER = lookupOrNull("llama_embed_infer",
            FunctionDescriptor.of(
                    ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS
            ));

    /** void llama_embed_destroy(void* engine) */
    static final MethodHandle EMBED_DESTROY = lookupOrNull("llama_embed_destroy",
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

    /** void llama_bridge_string_free(char* s) */
    static final MethodHandle STRING_FREE = lookupOrNull("llama_bridge_string_free",
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

    /** void llama_bridge_float_free(float* p) */
    static final MethodHandle FLOAT_FREE = lookupOrNull("llama_bridge_float_free",
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

    // ----------------------------------------------------------------
    // Event callback descriptor (for upcall stubs)
    // ----------------------------------------------------------------

    /** Function descriptor for: void callback(const char* event_json, void* user_data) */
    static final FunctionDescriptor EVENT_CB_DESCRIPTOR = FunctionDescriptor.ofVoid(
            ValueLayout.ADDRESS,   // event_json
            ValueLayout.ADDRESS    // user_data
    );

    // ----------------------------------------------------------------
    // Helpers
    // ----------------------------------------------------------------

    static MethodHandle lookupOrNull(String name, FunctionDescriptor desc) {
        return SYMBOLS.find(name)
                .map(addr -> LINKER.downcallHandle(addr, desc))
                .orElse(null);
    }

    /**
     * Read a NUL-terminated C string from a native address into a Java String.
     * Returns an empty string if the address is NULL.
     */
    static String readCString(MemorySegment addr) {
        if (addr == null || addr.equals(MemorySegment.NULL)) return "";
        return addr.reinterpret(Long.MAX_VALUE).getString(0);
    }

    /**
     * Create an upcall stub for the given EventListener so the bridge can
     * invoke it from C code.
     *
     * <p>The returned MemorySegment must be kept alive as long as the engine
     * that uses it is open. Managed by the engine classes.
     */
    static MemorySegment makeEventStub(EventListener listener, Arena arena)
            throws Exception {
        MethodHandle target = MethodHandles.lookup().findVirtual(
                EventListener.class,
                "onEventJson",
                MethodType.methodType(void.class, String.class, MemorySegment.class)
        ).bindTo(listener);

        return LINKER.upcallStub(target, EVENT_CB_DESCRIPTOR, arena);
    }
}
