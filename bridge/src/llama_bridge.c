/*
 * llama_bridge.c — implementation of the stable C bridge over llama.cpp.
 *
 * This is the ONLY file that knows the internals of llama.cpp.
 * When upstream llama.cpp changes its API, only this file needs updating.
 *
 * Public API: see bridge/include/llama_bridge.h
 */

#include "../include/llama_bridge.h"
#include "llama_bridge_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Engine creation
 * ---------------------------------------------------------------------- */

llama_engine_t llama_engine_create(const char* model_path) {
    if (!model_path || model_path[0] == '\0') {
        fprintf(stderr, "[llama_bridge] llama_engine_create: empty model path\n");
        return NULL;
    }

    /* Initialise the backend (idempotent after first call). */
    llama_backend_init();

    /* Load model. */
    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "[llama_bridge] failed to load model: %s\n", model_path);
        return NULL;
    }

    /* Create context. */
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = BRIDGE_N_CTX_DEFAULT;

    struct llama_context* ctx = llama_new_context_with_params(model, cparams);
    if (!ctx) {
        fprintf(stderr, "[llama_bridge] failed to create context\n");
        llama_model_free(model);
        return NULL;
    }

    struct llama_engine* engine =
        (struct llama_engine*)malloc(sizeof(struct llama_engine));
    if (!engine) {
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }

    engine->model = model;
    engine->ctx   = ctx;
    return (llama_engine_t)engine;
}

/* -------------------------------------------------------------------------
 * Completion
 * ---------------------------------------------------------------------- */

char* llama_engine_complete(llama_engine_t handle, const char* prompt) {
    if (!handle) {
        fprintf(stderr, "[llama_bridge] llama_engine_complete: null handle\n");
        return NULL;
    }

    struct llama_engine* engine = (struct llama_engine*)handle;

    /* Handle NULL / empty prompt gracefully. */
    if (!prompt) {
        prompt = "";
    }

    /* ---- tokenise prompt ---- */
    int n_prompt_tokens_max = (int)strlen(prompt) + 32;
    llama_token* prompt_tokens =
        (llama_token*)malloc((size_t)n_prompt_tokens_max * sizeof(llama_token));
    if (!prompt_tokens) {
        return NULL;
    }

    int n_prompt = llama_tokenize(
        engine->model,
        prompt,
        (int32_t)strlen(prompt),
        prompt_tokens,
        n_prompt_tokens_max,
        /* add_special */ true,
        /* parse_special */ false
    );

    if (n_prompt < 0) {
        /* Buffer was too small — try again with the required size. */
        int needed = -n_prompt;
        llama_token* tmp = (llama_token*)realloc(
            prompt_tokens, (size_t)needed * sizeof(llama_token));
        if (!tmp) {
            free(prompt_tokens);
            return NULL;
        }
        prompt_tokens = tmp;
        n_prompt = llama_tokenize(
            engine->model,
            prompt,
            (int32_t)strlen(prompt),
            prompt_tokens,
            needed,
            true,
            false
        );
        if (n_prompt < 0) {
            free(prompt_tokens);
            return NULL;
        }
    }

    /* ---- clear KV cache so this call is stateless ---- */
    llama_kv_cache_clear(engine->ctx);

    /* ---- feed prompt tokens ---- */
    struct llama_batch batch = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        batch.token[i]     = prompt_tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 0;
    }
    batch.logits[n_prompt - 1] = 1; /* only need logits for last token */
    batch.n_tokens = n_prompt;
    free(prompt_tokens);

    if (llama_decode(engine->ctx, batch) != 0) {
        fprintf(stderr, "[llama_bridge] llama_decode failed on prompt\n");
        llama_batch_free(batch);
        return NULL;
    }
    llama_batch_free(batch);

    /* ---- set up greedy sampler chain ---- */
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler* smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    /* ---- generation loop ---- */
    const llama_token eos_token = llama_token_eos(engine->model);

    /* Dynamic output buffer */
    size_t out_cap  = 256;
    size_t out_len  = 0;
    char*  out_buf  = (char*)malloc(out_cap);
    if (!out_buf) {
        llama_sampler_free(smpl);
        return NULL;
    }
    out_buf[0] = '\0';

    int n_cur = n_prompt;

    for (int step = 0; step < BRIDGE_N_PREDICT; step++) {
        llama_token new_token = llama_sampler_sample(smpl, engine->ctx, -1);
        llama_sampler_accept(smpl, new_token);

        if (new_token == eos_token) {
            break;
        }

        /* Convert token to piece text */
        char piece[128];
        int piece_len = llama_token_to_piece(
            engine->model, new_token, piece, sizeof(piece) - 1, 0, true);
        if (piece_len < 0) {
            piece_len = 0;
        }
        piece[piece_len] = '\0';

        /* Grow output buffer if needed */
        if (out_len + (size_t)piece_len + 1 > out_cap) {
            out_cap = (out_cap + (size_t)piece_len + 1) * 2;
            char* tmp = (char*)realloc(out_buf, out_cap);
            if (!tmp) {
                free(out_buf);
                llama_sampler_free(smpl);
                return NULL;
            }
            out_buf = tmp;
        }

        memcpy(out_buf + out_len, piece, (size_t)piece_len);
        out_len += (size_t)piece_len;
        out_buf[out_len] = '\0';

        /* Feed the new token back for the next step */
        struct llama_batch next_batch = llama_batch_init(1, 0, 1);
        next_batch.token[0]     = new_token;
        next_batch.pos[0]       = n_cur;
        next_batch.n_seq_id[0]  = 1;
        next_batch.seq_id[0][0] = 0;
        next_batch.logits[0]    = 1;
        next_batch.n_tokens     = 1;
        n_cur++;

        if (llama_decode(engine->ctx, next_batch) != 0) {
            fprintf(stderr, "[llama_bridge] llama_decode failed at step %d\n", step);
            llama_batch_free(next_batch);
            break;
        }
        llama_batch_free(next_batch);
    }

    llama_sampler_free(smpl);
    return out_buf;
}

/* -------------------------------------------------------------------------
 * String free helper
 * ---------------------------------------------------------------------- */

void llama_engine_free_string(char* str) {
    free(str);
}

/* -------------------------------------------------------------------------
 * Engine destroy
 * ---------------------------------------------------------------------- */

void llama_engine_destroy(llama_engine_t handle) {
    if (!handle) {
        return;
    }
    struct llama_engine* engine = (struct llama_engine*)handle;
    if (engine->ctx) {
        llama_free(engine->ctx);
        engine->ctx = NULL;
    }
    if (engine->model) {
        llama_model_free(engine->model);
        engine->model = NULL;
    }
    free(engine);
}
