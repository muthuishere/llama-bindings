#ifndef LLAMA_BRIDGE_H
#define LLAMA_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/* ------------------------------------------------------------------ */
/* Opaque engine handles                                               */
/* ------------------------------------------------------------------ */

typedef void* llama_chat_engine_t;
typedef void* llama_embed_engine_t;

/* ------------------------------------------------------------------ */
/* Event callback                                                      */
/*                                                                     */
/* callback receives a JSON-encoded event payload (see spec §10.2).   */
/* The string is valid only for the duration of the callback.         */
/* user_data is passed through from the create call.                  */
/* ------------------------------------------------------------------ */

typedef void (*llama_event_cb)(const char* event_json, void* user_data);

/* ------------------------------------------------------------------ */
/* Chat engine                                                         */
/* ------------------------------------------------------------------ */

/**
 * Create a chat engine loaded from model_path.
 *
 * @param model_path  Path to the GGUF chat model file.
 * @param on_event    Optional event callback (may be NULL).
 * @param user_data   Opaque pointer forwarded to on_event.
 * @return            Engine handle, or NULL on failure.
 */
llama_chat_engine_t llama_chat_create(const char* model_path,
                                      llama_event_cb on_event,
                                      void* user_data);

/**
 * Run inference on a JSON-encoded chat request (see spec §8.1).
 *
 * @param engine       Chat engine handle.
 * @param request_json NUL-terminated JSON request string.
 * @return             Heap-allocated JSON response string, or NULL on OOM.
 *                     Caller must free with llama_bridge_string_free().
 */
char* llama_chat_infer_json(llama_chat_engine_t engine,
                            const char* request_json);

/**
 * Destroy a chat engine and release all associated resources.
 */
void llama_chat_destroy(llama_chat_engine_t engine);

/* ------------------------------------------------------------------ */
/* Embedding engine                                                    */
/* ------------------------------------------------------------------ */

/**
 * Create an embedding engine loaded from model_path.
 *
 * @param model_path  Path to the GGUF embedding model file.
 * @param on_event    Optional event callback (may be NULL).
 * @param user_data   Opaque pointer forwarded to on_event.
 * @return            Engine handle, or NULL on failure.
 */
llama_embed_engine_t llama_embed_create(const char* model_path,
                                        llama_event_cb on_event,
                                        void* user_data);

/**
 * Generate an embedding vector for input_text.
 *
 * @param engine      Embedding engine handle.
 * @param input_text  NUL-terminated input string.
 * @param out_len     Receives the number of floats in the returned array.
 * @return            Heap-allocated float array, or NULL on failure.
 *                    Caller must free with llama_bridge_float_free().
 */
float* llama_embed_infer(llama_embed_engine_t engine,
                         const char* input_text,
                         int* out_len);

/**
 * Destroy an embedding engine and release all associated resources.
 */
void llama_embed_destroy(llama_embed_engine_t engine);

/* ------------------------------------------------------------------ */
/* Memory helpers                                                      */
/* ------------------------------------------------------------------ */

/**
 * Free a string returned by the bridge.
 * Must be called instead of free() for any char* the bridge allocates.
 */
void llama_bridge_string_free(char* s);

/**
 * Free a float array returned by llama_embed_infer.
 * Must be called instead of free() for any float* the bridge allocates.
 */
void llama_bridge_float_free(float* p);

#ifdef __cplusplus
}
#endif

#endif /* LLAMA_BRIDGE_H */
