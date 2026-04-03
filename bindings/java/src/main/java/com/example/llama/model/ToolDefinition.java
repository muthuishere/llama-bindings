package com.example.llama.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Map;

/**
 * A tool the model may call (see spec §8.6).
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public final class ToolDefinition {

    @JsonProperty("name")        public String             name;
    @JsonProperty("description") public String             description;
    @JsonProperty("parameters")  public Map<String, Object> parameters;

    public ToolDefinition() {}

    public ToolDefinition(String name, String description,
                          Map<String, Object> parameters) {
        this.name        = name;
        this.description = description;
        this.parameters  = parameters;
    }
}
