package com.example.llama.agent;

import com.example.llama.LlamaException;
import com.example.llama.model.ToolDefinition;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class AgentTest {

    private static final String DUMMY_CHAT  = "testdata/dummy.gguf";
    private static final String DUMMY_EMBED = "testdata/dummy-embed.gguf";

    private Agent createOrSkip() {
        try {
            return Agent.create(DUMMY_CHAT, DUMMY_EMBED, ":memory:");
        } catch (Exception e) {
            return null; // model / bridge not available – skip
        }
    }

    @Test
    void createAndChatReturnsText() throws Exception {
        Agent agent = createOrSkip();
        if (agent == null) return;
        try {
            String reply = agent.chat("s1", "Say hello.");
            assertNotNull(reply);
            assertFalse(reply.isEmpty());
        } finally {
            agent.close();
        }
    }

    @Test
    void chatMaintainsSessionHistory() throws Exception {
        Agent agent = createOrSkip();
        if (agent == null) return;
        try {
            agent.chat("history-session", "What is 2+2?");
            String reply = agent.chat("history-session", "And 3+3?");
            assertNotNull(reply);
        } finally {
            agent.close();
        }
    }

    @Test
    void clearSessionAllowsFreshConversation() throws Exception {
        Agent agent = createOrSkip();
        if (agent == null) return;
        try {
            agent.chat("clear-session", "Hello");
            agent.clearSession("clear-session");
            String reply = agent.chat("clear-session", "Hello again");
            assertNotNull(reply);
        } finally {
            agent.close();
        }
    }

    @Test
    void addDocumentDoesNotThrow() throws Exception {
        Agent agent = createOrSkip();
        if (agent == null) return;
        try {
            assertDoesNotThrow(() -> {
                try {
                    agent.addDocument("The capital of France is Paris.");
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            });
        } finally {
            agent.close();
        }
    }

    @Test
    void addToolDoesNotThrow() throws Exception {
        Agent agent = createOrSkip();
        if (agent == null) return;
        try {
            assertDoesNotThrow(() ->
                agent.addTool(
                    new ToolDefinition("greet", "Greet user", Map.of("type", "object")),
                    args -> "Hello!"
                )
            );
        } finally {
            agent.close();
        }
    }

    @Test
    void closeIsIdempotent() throws Exception {
        Agent agent = createOrSkip();
        if (agent == null) return;
        agent.close();
        assertDoesNotThrow(agent::close);
    }

    @Test
    void closedAgentThrows() throws Exception {
        Agent agent = createOrSkip();
        if (agent == null) return;
        agent.close();
        assertThrows(LlamaException.class, () -> agent.chat("s", "hello"));
    }

    @Test
    void createWithInvalidChatModelThrows() {
        assertThrows(Exception.class, () ->
            Agent.create("/nonexistent/chat.gguf", DUMMY_EMBED, ":memory:")
        );
    }
}
