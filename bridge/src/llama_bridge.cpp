/*
 * llama_bridge.cpp
 *
 * Central bridge entry points: engine create/destroy and memory helpers.
 * Inference is delegated to llama_bridge_chat.cpp and llama_bridge_embed.cpp.
 */

#include "../include/llama_bridge.h"
#include "llama_bridge_internal.h"
#include "llama.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* CPU-only device list helper                                         */
/* ------------------------------------------------------------------ */

static ggml_backend_dev_t s_cpu_dev_list[2] = {nullptr, nullptr};
static bool s_cpu_dev_list_inited = false;

/* Returns a NULL-terminated list containing only the CPU device, or
   NULL if the CPU device could not be found. */
static ggml_backend_dev_t* cpu_only_devices() {
    if (!s_cpu_dev_list_inited) {
        s_cpu_dev_list_inited = true;
        size_t n = ggml_backend_dev_count();
        for (size_t i = 0; i < n; i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                s_cpu_dev_list[0] = dev;
                s_cpu_dev_list[1] = nullptr;
                break;
            }
        }
    }
    return s_cpu_dev_list[0] ? s_cpu_dev_list : nullptr;
}

/* ------------------------------------------------------------------ */
/* Chat engine                                                         */
/* ------------------------------------------------------------------ */

llama_chat_engine_t llama_chat_create(const char* model_path,
                                      llama_event_cb on_event,
                                      void* user_data)
{
    if (!model_path) return NULL;

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
    if (!impl) return NULL;

    impl->model_path = strdup(model_path);
    if (!impl->model_path) { free(impl); return NULL; }
    impl->on_event  = on_event;
    impl->user_data = user_data;
    impl->closed    = 0;

    bridge_emit(on_event, user_data,
                "chat_engine_create_start", "chat", "open_model_file", 0,
                "Loading chat model");

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    /* Restrict to CPU device to prevent Metal from being used on older GPUs */
    ggml_backend_dev_t* cpu_devs = cpu_only_devices();
    if (cpu_devs) {
        mparams.devices = cpu_devs;
    }

    impl->llama_model = llama_model_load_from_file(model_path, mparams);
    if (!impl->llama_model) {
        bridge_emit(on_event, user_data,
                    "chat_engine_create_failure", "chat", NULL, 0,
                    "Failed to load model");
        free(impl->model_path);
        free(impl);
        return NULL;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx   = 4096;
    cparams.n_batch = 512;

    impl->llama_ctx = llama_init_from_model(impl->llama_model, cparams);
    if (!impl->llama_ctx) {
        bridge_emit(on_event, user_data,
                    "chat_engine_create_failure", "chat", NULL, 0,
                    "Failed to create context");
        llama_model_free(impl->llama_model);
        free(impl->model_path);
        free(impl);
        return NULL;
    }

    bridge_emit(on_event, user_data,
                "chat_engine_create_success", "chat", "ready", 100,
                "Chat model loaded");

    return (llama_chat_engine_t)impl;
}

char* llama_chat_infer_json(llama_chat_engine_t engine,
                            const char* request_json)
{
    llama_chat_engine_impl_t* impl = (llama_chat_engine_impl_t*)engine;
    if (!impl || impl->closed)
        return bridge_json_error("ENGINE_CLOSED", "Chat engine is closed");
    if (!request_json)
        return bridge_json_error("INVALID_REQUEST", "request_json is NULL");
    return bridge_chat_infer(impl, request_json);
}

void llama_chat_destroy(llama_chat_engine_t engine)
{
    llama_chat_engine_impl_t* impl = (llama_chat_engine_impl_t*)engine;
    if (!impl) return;
    impl->closed = 1;
    bridge_emit(impl->on_event, impl->user_data,
                "engine_destroy", "chat", NULL, 0, "Chat engine destroyed");
    if (impl->llama_ctx)   { llama_free(impl->llama_ctx);          impl->llama_ctx   = NULL; }
    if (impl->llama_model) { llama_model_free(impl->llama_model);  impl->llama_model = NULL; }
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
    if (!model_path) return NULL;

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
    if (!impl) return NULL;

    impl->model_path = strdup(model_path);
    if (!impl->model_path) { free(impl); return NULL; }
    impl->on_event  = on_event;
    impl->user_data = user_data;
    impl->closed    = 0;
    impl->embed_dim = 0;

    bridge_emit(on_event, user_data,
                "embed_engine_create_start", "embed", "open_model_file", 0,
                "Loading embedding model");

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    ggml_backend_dev_t* cpu_devs = cpu_only_devices();
    if (cpu_devs) {
        mparams.devices = cpu_devs;
    }

    impl->llama_model = llama_model_load_from_file(model_path, mparams);
    if (!impl->llama_model) {
        bridge_emit(on_event, user_data,
                    "embed_engine_create_failure", "embed", NULL, 0,
                    "Failed to load embedding model");
        free(impl->model_path);
        free(impl);
        return NULL;
    }

    impl->embed_dim = llama_model_n_embd(impl->llama_model);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx        = 512;
    cparams.n_batch      = 512;
    cparams.embeddings   = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;

    impl->llama_ctx = llama_init_from_model(impl->llama_model, cparams);
    if (!impl->llama_ctx) {
        bridge_emit(on_event, user_data,
                    "embed_engine_create_failure", "embed", NULL, 0,
                    "Failed to create embedding context");
        llama_model_free(impl->llama_model);
        free(impl->model_path);
        free(impl);
        return NULL;
    }

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
    if (!impl) return;
    impl->closed = 1;
    bridge_emit(impl->on_event, impl->user_data,
                "engine_destroy", "embed", NULL, 0, "Embed engine destroyed");
    if (impl->llama_ctx)   { llama_free(impl->llama_ctx);          impl->llama_ctx   = NULL; }
    if (impl->llama_model) { llama_model_free(impl->llama_model);  impl->llama_model = NULL; }
    free(impl->model_path);
    free(impl);
}

/* ------------------------------------------------------------------ */
/* Memory helpers                                                      */
/* ------------------------------------------------------------------ */

void llama_bridge_string_free(char* s) { free(s); }
void llama_bridge_float_free(float* p) { free(p); }
