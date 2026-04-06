package com.example.llama.tools;

import com.example.llama.model.ToolDefinition;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ToolRegistryTest {

    @Test
    void registerAndExecute() {
        ToolRegistry reg = new ToolRegistry();
        reg.register(new ToolDefinition("add", "Add two numbers", Map.of("type", "object")),
                args -> (double) args.get("a") + (double) args.get("b"));

        Object result = reg.execute("add", Map.of("a", 2.0, "b", 3.0));
        assertEquals(5.0, result);
    }

    @Test
    void executeUnknownToolThrows() {
        ToolRegistry reg = new ToolRegistry();
        assertThrows(IllegalArgumentException.class,
                () -> reg.execute("nonexistent", Map.of()));
    }

    @Test
    void registerEmptyNameThrows() {
        ToolRegistry reg = new ToolRegistry();
        assertThrows(IllegalArgumentException.class,
                () -> reg.register(new ToolDefinition("", "desc", null), args -> null));
    }

    @Test
    void registerNullHandlerThrows() {
        ToolRegistry reg = new ToolRegistry();
        assertThrows(IllegalArgumentException.class,
                () -> reg.register(new ToolDefinition("foo", "desc", null), null));
    }

    @Test
    void definitionsInRegistrationOrder() {
        ToolRegistry reg = new ToolRegistry();
        for (String name : new String[]{"a", "b", "c"}) {
            reg.register(new ToolDefinition(name, "", null), args -> null);
        }
        List<ToolDefinition> defs = reg.definitions();
        assertEquals(3, defs.size());
        assertEquals("a", defs.get(0).name);
        assertEquals("b", defs.get(1).name);
        assertEquals("c", defs.get(2).name);
    }

    @Test
    void hasReturnsTrueAfterRegister() {
        ToolRegistry reg = new ToolRegistry();
        assertFalse(reg.has("foo"));
        reg.register(new ToolDefinition("foo", "", null), args -> null);
        assertTrue(reg.has("foo"));
    }

    @Test
    void reregisterUpdatesDefinition() {
        ToolRegistry reg = new ToolRegistry();
        reg.register(new ToolDefinition("x", "v1", null), args -> "v1");
        reg.register(new ToolDefinition("x", "v2", null), args -> "v2");
        List<ToolDefinition> defs = reg.definitions();
        assertEquals(1, defs.size());
        assertEquals("v2", defs.get(0).description);
    }
}
