package com.example.llama.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * A single chat message (see spec §8.2).
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public final class ChatMessage {

    @JsonProperty("role")      public String role;    // system | user | assistant | tool
    @JsonProperty("content")   public String content;
    @JsonProperty("tool_name") public String toolName; // set when role == "tool"

    public ChatMessage() {}

    public ChatMessage(String role, String content) {
        this.role    = role;
        this.content = content;
    }

    public static ChatMessage system(String content)    { return new ChatMessage("system", content); }
    public static ChatMessage user(String content)      { return new ChatMessage("user", content); }
    public static ChatMessage assistant(String content) { return new ChatMessage("assistant", content); }
}
