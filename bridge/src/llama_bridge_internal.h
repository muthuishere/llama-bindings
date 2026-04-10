#ifndef LLAMA_BRIDGE_INTERNAL_H
#define LLAMA_BRIDGE_INTERNAL_H

#include "../include/llama_bridge.h"
#include "llama.h"
#include <stdint.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Internal engine structs                                             */
/* ------------------------------------------------------------------ */

typedef struct llama_chat_engine_impl {
    char*                      model_path;
    llama_event_cb             on_event;
    void*                      user_data;
    int                        closed;
    struct llama_model *       llama_model;
    struct llama_context *     llama_ctx;
} llama_chat_engine_impl_t;

typedef struct llama_embed_engine_impl {
    char*                      model_path;
    llama_event_cb             on_event;
    void*                      user_data;
    int                        closed;
    int                        embed_dim;
    struct llama_model *       llama_model;
    struct llama_context *     llama_ctx;
} llama_embed_engine_impl_t;

/* ------------------------------------------------------------------ */
/* JSON helpers (llama_bridge_json.c)                                  */
/* ------------------------------------------------------------------ */

#ifdef __cplusplus
extern "C" {
#endif

/* Parse a string field from a flat JSON object. Caller must free result. */
char* bridge_json_get_string(const char* json, const char* key);

/* Parse an integer field from a flat JSON object. Returns default_val on error. */
int bridge_json_get_int(const char* json, const char* key, int default_val);

/* Parse a double field from a flat JSON object. Returns default_val on error. */
double bridge_json_get_double(const char* json, const char* key, double default_val);

/* Build an error response JSON. Caller must free result. */
char* bridge_json_error(const char* code, const char* message);

/* ------------------------------------------------------------------ */
/* Observability helpers (llama_bridge_observe.c)                      */
/* ------------------------------------------------------------------ */

/* Emit an event to the engine's callback if set. */
void bridge_emit_event(llama_event_cb cb, void* user_data, const char* event_json);

/* Emit a structured event. stage and message may be NULL. */
void bridge_emit(llama_event_cb cb, void* user_data,
                 const char* event_name,
                 const char* engine_type,
                 const char* stage,
                 int progress,
                 const char* message);

/* Return current time in milliseconds since epoch. */
int64_t bridge_now_ms(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

/* ------------------------------------------------------------------ */
/* Chat inference (llama_bridge_chat.cpp)                              */
/* ------------------------------------------------------------------ */

char* bridge_chat_infer(llama_chat_engine_impl_t* impl, const char* request_json);

/* ------------------------------------------------------------------ */
/* Embed inference (llama_bridge_embed.cpp)                            */
/* ------------------------------------------------------------------ */

float* bridge_embed_infer(llama_embed_engine_impl_t* impl,
                          const char* input_text,
                          int* out_len);

#endif /* LLAMA_BRIDGE_INTERNAL_H */
