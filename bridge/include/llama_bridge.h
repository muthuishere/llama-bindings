#ifndef LLAMA_BRIDGE_H
#define LLAMA_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * llama_bridge.h — public stable API for the llama.cpp bridge.
 *
 * Language bindings (Go, Java, JS) depend only on this header.
 * They must never call raw llama.cpp APIs directly.
 */

typedef void* llama_engine_t;

/**
 * Create an engine by loading a GGUF model from the given path.
 * Returns NULL on failure.
 */
llama_engine_t llama_engine_create(const char* model_path);

/**
 * Run a completion for the given prompt using the engine.
 * Returns a heap-allocated C string that the caller must free with
 * llama_engine_free_string(). Returns NULL on failure.
 */
char* llama_engine_complete(llama_engine_t engine, const char* prompt);

/**
 * Free a string returned by llama_engine_complete().
 */
void llama_engine_free_string(char* str);

/**
 * Destroy the engine, freeing all associated resources.
 */
void llama_engine_destroy(llama_engine_t engine);

#ifdef __cplusplus
}
#endif

#endif /* LLAMA_BRIDGE_H */
