package com.example.llama;

/**
 * Thrown when the llama bridge or model reports an error.
 */
public final class LlamaException extends Exception {

    private final String code;

    public LlamaException(String code, String message) {
        super("[" + code + "] " + message);
        this.code = code;
    }

    /** The bridge error code, e.g. {@code "MODEL_LOAD_FAILED"}. */
    public String getCode() {
        return code;
    }
}
