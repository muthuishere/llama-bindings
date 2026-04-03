package com.example.llama.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Map;

/**
 * A tool invocation returned by the model (see spec §8.6).
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public final class ToolCall {

    @JsonProperty("id")        public String             id;
    @JsonProperty("name")      public String             name;
    @JsonProperty("arguments") public Map<String, Object> arguments;

    public ToolCall() {}
}
