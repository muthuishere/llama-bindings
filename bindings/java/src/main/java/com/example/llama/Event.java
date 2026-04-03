package com.example.llama;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * An observability event emitted by the bridge (see spec §10.2).
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public final class Event {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @JsonProperty("event")        public String event;
    @JsonProperty("engine_type") public String engineType;
    @JsonProperty("stage")       public String stage;
    @JsonProperty("progress")    public int    progress;
    @JsonProperty("message")     public String message;
    @JsonProperty("partial_text") public String partialText;
    @JsonProperty("timestamp_ms") public long  timestampMs;

    public Event() {}

    /**
     * Parse a bridge event from its JSON representation.
     */
    public static Event fromJson(String json) throws Exception {
        return MAPPER.readValue(json, Event.class);
    }

    @Override
    public String toString() {
        return "Event{event='" + event + "', engine='" + engineType
                + "', stage='" + stage + "', progress=" + progress
                + ", message='" + message + "'}";
    }
}
