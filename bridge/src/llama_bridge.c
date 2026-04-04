/*
 * llama_bridge.c
 *
 * Central bridge entry points: engine create/destroy and memory helpers.
 * Inference is delegated to llama_bridge_chat.c and llama_bridge_embed.c.
 */

#include "../include/llama_bridge.h"
#include "llama_bridge_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Chat engine                                                         */
/* ------------------------------------------------------------------ */

llama_chat_engine_t llama_chat_create(const char* model_path,
                                      llama_event_cb on_event,
                                      void* user_data)
{
    if (!model_path) {
        return NULL;
    }

    /* Verify the model file is accessible before allocating any resources. */
    {
        FILE* f = fopen(model_path, "rb");
        if (!f) {
            bridge_emit(on_event, user_data,
                        "chat_engine_create_failure", "chat", NULL, 0,
                        "Model file not found or not accessible");
            return NULL;
        }
        fclose(f);
    }

    llama_chat_engine_impl_t* impl =
        (llama_chat_engine_impl_t*)calloc(1, sizeof(llama_chat_engine_impl_t));
    if (!impl) {
        return NULL;
    }

    impl->model_path = strdup(model_path);
    if (!impl->model_path) {
        free(impl);
        return NULL;
    }
    impl->on_event  = on_event;
    impl->user_data = user_data;
    impl->closed    = 0;

    bridge_emit(on_event, user_data,
                "chat_engine_create_start", "chat", "open_model_file", 0,
                "Loading chat model");

    /*
     * TODO(integration): call llama.cpp model-load here.
     * On failure: emit chat_engine_create_failure, free impl, return NULL.
     */

    bridge_emit(on_event, user_data,
                "chat_engine_create_success", "chat", "ready", 100,
                "Chat model loaded");

    return (llama_chat_engine_t)impl;
}

char* llama_chat_infer_json(llama_chat_engine_t engine,
                            const char* request_json)
{
    llama_chat_engine_impl_t* impl = (llama_chat_engine_impl_t*)engine;
    if (!impl || impl->closed) {
        return bridge_json_error("ENGINE_CLOSED", "Chat engine is closed");
    }
    if (!request_json) {
        return bridge_json_error("INVALID_REQUEST", "request_json is NULL");
    }
    return bridge_chat_infer(impl, request_json);
}

void llama_chat_destroy(llama_chat_engine_t engine)
{
    llama_chat_engine_impl_t* impl = (llama_chat_engine_impl_t*)engine;
    if (!impl) {
        return;
    }
    impl->closed = 1;
    bridge_emit(impl->on_event, impl->user_data,
                "engine_destroy", "chat", NULL, 0, "Chat engine destroyed");

    /* TODO(integration): release llama.cpp context here. */

    free(impl->model_path);
    free(impl);
}

/* ------------------------------------------------------------------ */
/* Embedding engine                                                    */
/* ------------------------------------------------------------------ */

llama_embed_engine_t llama_embed_create(const char* model_path,
                                        llama_event_cb on_event,
                                        void* user_data)
{
    if (!model_path) {
        return NULL;
    }

    /* Verify the model file is accessible before allocating any resources. */
    {
        FILE* f = fopen(model_path, "rb");
        if (!f) {
            bridge_emit(on_event, user_data,
                        "embed_engine_create_failure", "embed", NULL, 0,
                        "Model file not found or not accessible");
            return NULL;
        }
        fclose(f);
    }

    llama_embed_engine_impl_t* impl =
        (llama_embed_engine_impl_t*)calloc(1, sizeof(llama_embed_engine_impl_t));
    if (!impl) {
        return NULL;
    }

    impl->model_path = strdup(model_path);
    if (!impl->model_path) {
        free(impl);
        return NULL;
    }
    impl->on_event  = on_event;
    impl->user_data = user_data;
    impl->closed    = 0;
    impl->embed_dim = 0;

    bridge_emit(on_event, user_data,
                "embed_engine_create_start", "embed", "open_model_file", 0,
                "Loading embedding model");

    /*
     * TODO(integration): call llama.cpp embedding model load here.
     * Set impl->embed_dim from model metadata.
     * On failure: emit embed_engine_create_failure, free impl, return NULL.
     */

    bridge_emit(on_event, user_data,
                "embed_engine_create_success", "embed", "ready", 100,
                "Embedding model loaded");

    return (llama_embed_engine_t)impl;
}

float* llama_embed_infer(llama_embed_engine_t engine,
                         const char* input_text,
                         int* out_len)
{
    llama_embed_engine_impl_t* impl = (llama_embed_engine_impl_t*)engine;
    if (!impl || impl->closed || !input_text || !out_len) {
        if (out_len) *out_len = 0;
        return NULL;
    }
    return bridge_embed_infer(impl, input_text, out_len);
}

void llama_embed_destroy(llama_embed_engine_t engine)
{
    llama_embed_engine_impl_t* impl = (llama_embed_engine_impl_t*)engine;
    if (!impl) {
        return;
    }
    impl->closed = 1;
    bridge_emit(impl->on_event, impl->user_data,
                "engine_destroy", "embed", NULL, 0, "Embed engine destroyed");

    /* TODO(integration): release llama.cpp context here. */

    free(impl->model_path);
    free(impl);
}

/* ------------------------------------------------------------------ */
/* Memory helpers                                                      */
/* ------------------------------------------------------------------ */

void llama_bridge_string_free(char* s)
{
    free(s);
}

void llama_bridge_float_free(float* p)
{
    free(p);
}
