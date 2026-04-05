package com.example.llama;

/**
 * Receives observability events from the bridge.
 *
 * <p>Implementations must be thread-safe; callbacks may arrive from a native
 * thread. Implementations must not block.
 */
public interface EventListener {

    /**
     * Called for each event.
     *
     * @param event the parsed event
     */
    void onEvent(Event event);

    /**
     * Internal bridge entry-point. Parses the raw JSON and delegates to
     * {@link #onEvent(Event)}.
     *
     * @param eventJsonPtr native pointer to the raw JSON event string from the bridge
     * @param userData   opaque pointer (unused; routing handled in NativeLibrary)
     */
    default void onEventJson(java.lang.foreign.MemorySegment eventJsonPtr,
                             java.lang.foreign.MemorySegment userData) {
        try {
            String eventJson = NativeLibrary.readCString(eventJsonPtr);
            Event evt = Event.fromJson(eventJson);
            onEvent(evt);
        } catch (Exception ignored) {
            // Never propagate exceptions across the FFM boundary.
        }
    }
}
