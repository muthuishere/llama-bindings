package com.example.llama.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * Sampling/generation parameters merged into a {@link ChatRequest}.
 */
@JsonInclude(JsonInclude.Include.NON_DEFAULT)
public final class GenerationOptions {

    @JsonProperty("temperature")      public float        temperature;
    @JsonProperty("max_output_tokens") public int         maxOutputTokens;
    @JsonProperty("top_p")            public float        topP;
    @JsonProperty("top_k")            public int          topK;
    @JsonProperty("stop")             public List<String> stop;

    public GenerationOptions() {}

    public GenerationOptions(float temperature, int maxOutputTokens,
                             float topP, int topK) {
        this.temperature     = temperature;
        this.maxOutputTokens = maxOutputTokens;
        this.topP            = topP;
        this.topK            = topK;
    }
}
