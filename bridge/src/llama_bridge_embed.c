/*
 * llama_bridge_embed.c
 *
 * Embedding inference implementation.
 *
 * Responsibility:
 *   - accept a text string
 *   - produce a float vector
 *   - emit observability events
 *
 * TODO(integration): replace stub vector with real llama.cpp embedding call.
 */

#include "llama_bridge_internal.h"

#include <stdlib.h>
#include <string.h>

float* bridge_embed_infer(llama_embed_engine_impl_t* impl,
                          const char* input_text,
                          int* out_len)
{
    bridge_emit(impl->on_event, impl->user_data,
                "embed_infer_start", "embed", "processing", 0,
                "Embedding started");

    /*
     * TODO(integration): replace this stub with a real llama.cpp
     * embedding call using impl->llama_model / impl->llama_ctx.
     *
     * The stub returns a minimal vector so callers can exercise the
     * full API surface without a model file present.
     */
    int dim = (impl->embed_dim > 0) ? impl->embed_dim : 4;
    float* vec = (float*)calloc((size_t)dim, sizeof(float));
    if (!vec) {
        *out_len = 0;
        bridge_emit(impl->on_event, impl->user_data,
                    "embed_infer_failure", "embed", NULL, 0,
                    "Out of memory allocating vector");
        return NULL;
    }

    /* Populate stub values based on input length so tests can assert > 0. */
    size_t ilen = strlen(input_text);
    for (int i = 0; i < dim; i++) {
        vec[i] = (float)((ilen + i + 1) % 100) / 100.0f;
    }
    *out_len = dim;

    bridge_emit(impl->on_event, impl->user_data,
                "embed_infer_success", "embed", "completed", 100,
                "Embedding complete");
    bridge_emit(impl->on_event, impl->user_data,
                "embed_complete", "embed", NULL, 100, NULL);

    return vec;
}
