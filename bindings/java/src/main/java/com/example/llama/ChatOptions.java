package com.example.llama;

/**
 * Per-call options for {@link ChatEngine#chat}.
 */
public final class ChatOptions {

    public final float         temperature;
    public final int           maxOutputTokens;
    public final float         topP;
    public final int           topK;
    /** Optional per-call event listener. May be {@code null}. */
    public final EventListener listener;

    public ChatOptions() {
        this(0.7f, 256, 0.95f, 40, null);
    }

    public ChatOptions(float temperature, int maxOutputTokens, float topP,
                       int topK, EventListener listener) {
        this.temperature     = temperature;
        this.maxOutputTokens = maxOutputTokens;
        this.topP            = topP;
        this.topK            = topK;
        this.listener        = listener;
    }
}
