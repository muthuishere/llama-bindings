/*
 * llama_bridge_embed.cpp
 *
 * Embedding inference implementation using llama.cpp.
 */

#include "llama_bridge_internal.h"
#include "llama.h"

#include <stdlib.h>
#include <string.h>
#include <vector>

float* bridge_embed_infer(llama_embed_engine_impl_t* impl,
                          const char* input_text,
                          int* out_len)
{
    bridge_emit(impl->on_event, impl->user_data,
                "embed_infer_start", "embed", "processing", 0,
                "Embedding started");

    const struct llama_vocab* vocab = llama_model_get_vocab(impl->llama_model);
    int n_embd = llama_model_n_embd(impl->llama_model);

    /* Tokenize input */
    int n_ctx = (int)llama_n_ctx(impl->llama_ctx);
    std::vector<llama_token> tokens(n_ctx);
    int n_tokens = llama_tokenize(vocab,
                                  input_text, (int32_t)strlen(input_text),
                                  tokens.data(), (int32_t)tokens.size(),
                                  true, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab,
                                  input_text, (int32_t)strlen(input_text),
                                  tokens.data(), (int32_t)tokens.size(),
                                  true, true);
    }
    if (n_tokens <= 0 || n_tokens > n_ctx) {
        *out_len = 0;
        bridge_emit(impl->on_event, impl->user_data,
                    "embed_infer_failure", "embed", NULL, 0,
                    "Tokenization failed");
        return NULL;
    }
    tokens.resize(n_tokens);

    /* Clear KV cache */
    llama_memory_t mem = llama_get_memory(impl->llama_ctx);
    if (mem) llama_memory_clear(mem, true);

    /* Decode */
    struct llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    if (llama_decode(impl->llama_ctx, batch) != 0) {
        *out_len = 0;
        bridge_emit(impl->on_event, impl->user_data,
                    "embed_infer_failure", "embed", NULL, 0,
                    "Decode failed");
        return NULL;
    }

    /* Get pooled embeddings for sequence 0 */
    float* embd = llama_get_embeddings_seq(impl->llama_ctx, 0);
    if (!embd) {
        /* Fall back to last-token embeddings */
        embd = llama_get_embeddings(impl->llama_ctx);
    }
    if (!embd) {
        *out_len = 0;
        bridge_emit(impl->on_event, impl->user_data,
                    "embed_infer_failure", "embed", NULL, 0,
                    "No embeddings available");
        return NULL;
    }

    /* Copy to heap-allocated array (caller owns it via llama_bridge_float_free) */
    float* out = (float*)malloc((size_t)n_embd * sizeof(float));
    if (!out) {
        *out_len = 0;
        bridge_emit(impl->on_event, impl->user_data,
                    "embed_infer_failure", "embed", NULL, 0,
                    "Out of memory");
        return NULL;
    }
    memcpy(out, embd, (size_t)n_embd * sizeof(float));
    *out_len = n_embd;

    bridge_emit(impl->on_event, impl->user_data,
                "embed_infer_success", "embed", "completed", 100,
                "Embedding complete");
    bridge_emit(impl->on_event, impl->user_data,
                "embed_complete", "embed", NULL, 100, NULL);

    return out;
}
