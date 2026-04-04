package com.example.llama;

import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Low-level JVM ↔ native bridge via Project Panama FFM (JDK 21+).
 *
 * <p>Library loading strategy (in order):
 * <ol>
 *   <li>Extract the platform-specific native library bundled inside the JAR
 *       (path: {@code native/<os>-<arch>/<libname>}) to a temporary file, then
 *       load it with {@link System#load}.</li>
 *   <li>Fall back to {@link System#loadLibrary}("llama_bridge"), which searches
 *       {@code java.library.path} — useful for developer builds.</li>
 * </ol>
 *
 * <p>This class is internal to the library. Application code should use
 * {@link ChatEngine} and {@link EmbedEngine} exclusively.
 */
final class NativeLibrary {

    private NativeLibrary() {}

    // ----------------------------------------------------------------
    // Bridge availability flag
    // ----------------------------------------------------------------

    /** Non-null when the native bridge could not be loaded. */
    static final Throwable LOAD_ERROR;

    static {
        Throwable loadError = null;
        try {
            loadNativeBridge();
        } catch (Throwable t) {
            loadError = t;
        }
        LOAD_ERROR = loadError;
    }

    /**
     * Throw a {@link LlamaException} if the native bridge failed to load.
     * Called at the top of {@link ChatEngine#load} and {@link EmbedEngine#load}.
     */
    static void checkAvailable() throws LlamaException {
        if (LOAD_ERROR != null) {
            throw new LlamaException("BRIDGE_NOT_AVAILABLE",
                    "Native bridge could not be loaded: " + LOAD_ERROR.getMessage());
        }
    }

    // ----------------------------------------------------------------
    // Library loading
    // ----------------------------------------------------------------

    private static void loadNativeBridge() throws Exception {
        // 1. Try bundled prebuilt native inside the JAR.
        String resourcePath = bundledResourcePath();
        if (resourcePath != null) {
            try (InputStream in = NativeLibrary.class.getResourceAsStream(resourcePath)) {
                if (in != null) {
                    Path tmp = extractToTemp(in, nativeLibFileName());
                    System.load(tmp.toAbsolutePath().toString());
                    return;
                }
            }
        }

        // 2. Fall back to java.library.path (developer build via task build-bridge).
        System.loadLibrary("llama_bridge");
    }

    /**
     * Returns the JAR-resource path for the native library matching the current
     * OS and CPU architecture, or {@code null} for unsupported platforms.
     */
    private static String bundledResourcePath() {
        String os   = System.getProperty("os.name", "").toLowerCase();
        String arch = System.getProperty("os.arch", "").toLowerCase();

        String osKey;
        if (os.contains("linux")) {
            osKey = "linux";
        } else if (os.contains("mac") || os.contains("darwin")) {
            osKey = "darwin";
        } else if (os.contains("win")) {
            osKey = "windows";
        } else {
            return null;
        }

        String archKey;
        if (arch.equals("amd64") || arch.equals("x86_64")) {
            archKey = "x86_64";
        } else if (arch.equals("aarch64") || arch.equals("arm64")) {
            archKey = "aarch64";
        } else {
            return null;
        }

        return "/native/" + osKey + "-" + archKey + "/" + nativeLibFileName();
    }

    private static String nativeLibFileName() {
        String os = System.getProperty("os.name", "").toLowerCase();
        if (os.contains("win")) return "llama_bridge.dll";
        if (os.contains("mac") || os.contains("darwin")) return "libllama_bridge.dylib";
        return "libllama_bridge.so";
    }

    /** Copy an input stream to a temp file and return its path. */
    private static Path extractToTemp(InputStream in, String libName) throws IOException {
        Path tmp = Files.createTempFile("llama_bridge_", "_" + libName);
        tmp.toFile().deleteOnExit();
        Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
        tmp.toFile().setExecutable(true);
        return tmp;
    }

    // ----------------------------------------------------------------
    // Panama FFM linker
    // ----------------------------------------------------------------

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
        if (LOAD_ERROR != null) return null;
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
