package com.example.llama.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Token-usage accounting returned by the bridge (see spec §8.4).
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public final class UsageInfo {

    @JsonProperty("prompt_tokens")     public int promptTokens;
    @JsonProperty("completion_tokens") public int completionTokens;
    @JsonProperty("total_tokens")      public int totalTokens;

    public UsageInfo() {}
}
