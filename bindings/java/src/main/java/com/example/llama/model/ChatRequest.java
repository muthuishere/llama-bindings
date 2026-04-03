package com.example.llama.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;

/**
 * Top-level request sent to the chat engine (see spec §8.1).
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public final class ChatRequest {

    @JsonProperty("messages")      public List<ChatMessage>    messages;
    @JsonProperty("response_mode") public String               responseMode;
    @JsonProperty("schema")        public Map<String, Object>  schema;
    @JsonProperty("tools")         public List<ToolDefinition> tools;
    @JsonProperty("tool_choice")   public String               toolChoice;
    @JsonProperty("generation")    public GenerationOptions    generation;

    public ChatRequest() {}

    public static Builder builder() { return new Builder(); }

    public static final class Builder {
        private final ChatRequest r = new ChatRequest();

        public Builder messages(List<ChatMessage> m)       { r.messages = m;      return this; }
        public Builder responseMode(String mode)           { r.responseMode = mode; return this; }
        public Builder schema(Map<String, Object> schema)  { r.schema = schema;   return this; }
        public Builder tools(List<ToolDefinition> tools)   { r.tools = tools;     return this; }
        public Builder toolChoice(String tc)               { r.toolChoice = tc;   return this; }
        public Builder generation(GenerationOptions g)     { r.generation = g;    return this; }

        public ChatRequest build() { return r; }
    }
}
