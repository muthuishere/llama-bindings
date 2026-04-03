package com.example.llama;

/**
 * Per-call options for {@link EmbedEngine#embed}.
 */
public final class EmbedOptions {

    /** Optional per-call event listener. May be {@code null}. */
    public final EventListener listener;

    public EmbedOptions() {
        this.listener = null;
    }

    public EmbedOptions(EventListener listener) {
        this.listener = listener;
    }
}
