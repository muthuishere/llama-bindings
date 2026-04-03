package com.example.llama.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Error detail returned inside an error-type {@link ChatResponse}.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public final class ErrorInfo {

    @JsonProperty("code")    public String code;
    @JsonProperty("message") public String message;

    public ErrorInfo() {}
}
