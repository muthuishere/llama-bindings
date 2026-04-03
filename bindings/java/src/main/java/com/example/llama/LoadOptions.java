package com.example.llama;

/**
 * Options passed when loading a {@link ChatEngine} or {@link EmbedEngine}.
 */
public final class LoadOptions {

    /** Optional listener for load and lifecycle events. May be {@code null}. */
    public final EventListener listener;

    public LoadOptions() {
        this.listener = null;
    }

    public LoadOptions(EventListener listener) {
        this.listener = listener;
    }
}
