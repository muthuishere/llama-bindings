package com.example.llama.tools;

import com.example.llama.model.ToolDefinition;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;

/**
 * ToolRegistry maps tool names to their definitions and Java handler functions.
 *
 * <p>Thread-safe for concurrent reads. {@link #register} should be called
 * before the Agent starts handling requests.
 *
 * <p>Usage:
 * <pre>{@code
 * ToolRegistry reg = new ToolRegistry();
 * reg.register(new ToolDefinition("add", "Add two numbers", params),
 *              args -> (double) args.get("a") + (double) args.get("b"));
 * Object result = reg.execute("add", Map.of("a", 2.0, "b", 3.0));
 * }</pre>
 */
public final class ToolRegistry {

    /** Handler function: receives decoded JSON arguments, returns a result. */
    @FunctionalInterface
    public interface Handler extends Function<Map<String, Object>, Object> {
        Object apply(Map<String, Object> args);
    }

    private record Entry(ToolDefinition def, Handler handler) {}

    private final Map<String, Entry> tools   = new ConcurrentHashMap<>();
    private final List<ToolDefinition> order = new ArrayList<>(); // registration order

    /**
     * Register a tool.
     *
     * @param def     tool definition (name must be non-empty)
     * @param handler handler function (must not be null)
     * @throws IllegalArgumentException if name is empty or handler is null
     */
    public synchronized void register(ToolDefinition def, Handler handler) {
        if (def == null || def.name == null || def.name.isEmpty()) {
            throw new IllegalArgumentException("ToolRegistry: tool name must not be empty");
        }
        if (handler == null) {
            throw new IllegalArgumentException(
                    "ToolRegistry: handler for \"" + def.name + "\" must not be null");
        }
        if (!tools.containsKey(def.name)) {
            order.add(def);
        } else {
            // Update ordered definition in place.
            for (int i = 0; i < order.size(); i++) {
                if (order.get(i).name.equals(def.name)) {
                    order.set(i, def);
                    break;
                }
            }
        }
        tools.put(def.name, new Entry(def, handler));
    }

    /**
     * Return all registered tool definitions in registration order.
     *
     * @return unmodifiable snapshot
     */
    public synchronized List<ToolDefinition> definitions() {
        return List.copyOf(order);
    }

    /**
     * Execute the named tool.
     *
     * @param name tool name
     * @param args decoded argument map (may be null → treated as empty)
     * @return handler return value (JSON-serialisable)
     * @throws IllegalArgumentException if the tool is not registered
     */
    public Object execute(String name, Map<String, Object> args) {
        Entry entry = tools.get(name);
        if (entry == null) {
            throw new IllegalArgumentException("ToolRegistry: unknown tool \"" + name + "\"");
        }
        return entry.handler().apply(args != null ? args : Map.of());
    }

    /**
     * Returns {@code true} if a tool with the given name is registered.
     *
     * @param name tool name
     * @return true if registered
     */
    public boolean has(String name) {
        return tools.containsKey(name);
    }
}
