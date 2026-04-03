package com.example.llama.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

/**
 * Normalized response from the chat engine (see spec §8.4 – 8.6).
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public final class ChatResponse {

    /** "assistant_text" | "structured_json" | "tool_call" | "error" */
    @JsonProperty("type")          public String           type;
    @JsonProperty("text")          public String           text;
    @JsonProperty("json")          public Map<String, Object> json;
    @JsonProperty("tool_calls")    public List<ToolCall>   toolCalls;
    @JsonProperty("finish_reason") public String           finishReason;
    @JsonProperty("usage")         public UsageInfo        usage;
    @JsonProperty("error")         public ErrorInfo        error;

    public ChatResponse() {}

    /** Convenience: true if the bridge returned an error response. */
    public boolean isError() {
        return "error".equals(type);
    }
}
