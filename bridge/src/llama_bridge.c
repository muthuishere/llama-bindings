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

/* Tool-injection header/footer used by llama_engine_chat_with_tools */
static const char* BRIDGE_TOOL_HEADER =
    "You have access to the following tools. "
    "Use them when appropriate by responding with JSON:\n"
    "{\"tool\": \"<name>\", \"arguments\": {<key>: <value>}}\n\n"
    "<tools>\n";
static const char* BRIDGE_TOOL_FOOTER = "\n</tools>\n";

/* =========================================================================
 * Internal helpers
 * ====================================================================== */

static char* _strdup_safe(const char* s) {
    if (!s) return NULL;
    size_t n = strlen(s) + 1;
    char* copy = (char*)malloc(n);
    if (copy) memcpy(copy, s, n);
    return copy;
}

/*
 * Core inference: tokenise `prompt`, run the decode loop with a greedy
 * sampler, and return a heap-allocated completion string.
 * The caller must free the result with llama_engine_free_string().
 */
static char* _run_inference(struct llama_engine* engine, const char* prompt) {
    if (!prompt) prompt = "";

    /* ---- tokenise ---- */
    int n_max = (int)strlen(prompt) + 32;
    llama_token* tokens = (llama_token*)malloc((size_t)n_max * sizeof(llama_token));
    if (!tokens) return NULL;

    int n = llama_tokenize(
        engine->model, prompt, (int32_t)strlen(prompt),
        tokens, n_max, true, false);

    if (n < 0) {
        int needed = -n;
        llama_token* tmp = (llama_token*)realloc(
            tokens, (size_t)needed * sizeof(llama_token));
        if (!tmp) { free(tokens); return NULL; }
        tokens = tmp;
        n = llama_tokenize(engine->model, prompt, (int32_t)strlen(prompt),
                           tokens, needed, true, false);
        if (n < 0) { free(tokens); return NULL; }
    }

    /* ---- clear KV cache so this call is stateless ---- */
    llama_kv_cache_clear(engine->ctx);

    /* ---- feed prompt tokens ---- */
    struct llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 0;
    }
    batch.logits[n - 1] = 1;
    batch.n_tokens = n;
    free(tokens);

    if (llama_decode(engine->ctx, batch) != 0) {
        fprintf(stderr, "[llama_bridge] _run_inference: decode failed on prompt\n");
        llama_batch_free(batch);
        return NULL;
    }
    llama_batch_free(batch);

    /* ---- greedy sampler ---- */
    struct llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    struct llama_sampler* smpl = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    const llama_token eos = llama_token_eos(engine->model);

    /* ---- generation loop ---- */
    size_t out_cap = 256, out_len = 0;
    char*  out = (char*)malloc(out_cap);
    if (!out) { llama_sampler_free(smpl); return NULL; }
    out[0] = '\0';

    int n_cur = n;
    for (int step = 0; step < BRIDGE_N_PREDICT; step++) {
        llama_token tok = llama_sampler_sample(smpl, engine->ctx, -1);
        llama_sampler_accept(smpl, tok);
        if (tok == eos) break;

        char piece[128];
        int plen = llama_token_to_piece(
            engine->model, tok, piece, sizeof(piece) - 1, 0, true);
        if (plen < 0) plen = 0;
        piece[plen] = '\0';

        if (out_len + (size_t)plen + 1 > out_cap) {
            out_cap = (out_cap + (size_t)plen + 1) * 2;
            char* tmp = (char*)realloc(out, out_cap);
            if (!tmp) { free(out); llama_sampler_free(smpl); return NULL; }
            out = tmp;
        }
        memcpy(out + out_len, piece, (size_t)plen);
        out_len += (size_t)plen;
        out[out_len] = '\0';

        struct llama_batch nb = llama_batch_init(1, 0, 1);
        nb.token[0]     = tok;
        nb.pos[0]       = n_cur++;
        nb.n_seq_id[0]  = 1;
        nb.seq_id[0][0] = 0;
        nb.logits[0]    = 1;
        nb.n_tokens     = 1;
        if (llama_decode(engine->ctx, nb) != 0) {
            fprintf(stderr, "[llama_bridge] _run_inference: decode failed at step %d\n", step);
            llama_batch_free(nb);
            break;
        }
        llama_batch_free(nb);
    }

    llama_sampler_free(smpl);
    return out;
}

/*
 * Apply the model's chat template to the given message array and then
 * run inference.  Returns heap-allocated string or NULL on failure.
 */
static char* _chat_complete(struct llama_engine*             engine,
                             const struct llama_chat_message* msgs,
                             size_t                           n_msgs) {
    int32_t buf_sz = 4096;
    char*   buf    = (char*)malloc((size_t)buf_sz);
    if (!buf) return NULL;

    int32_t needed = llama_chat_apply_template(
        engine->model, NULL, msgs, n_msgs, true, buf, buf_sz);

    if (needed < 0) {
        fprintf(stderr, "[llama_bridge] llama_chat_apply_template failed\n");
        free(buf);
        return NULL;
    }
    if (needed > buf_sz) {
        buf_sz = needed + 1;
        char* tmp = (char*)realloc(buf, (size_t)buf_sz);
        if (!tmp) { free(buf); return NULL; }
        buf = tmp;
        needed = llama_chat_apply_template(
            engine->model, NULL, msgs, n_msgs, true, buf, buf_sz);
        if (needed < 0) {
            fprintf(stderr, "[llama_bridge] llama_chat_apply_template retry failed\n");
            free(buf);
            return NULL;
        }
    }
    buf[needed] = '\0';

    char* result = _run_inference(engine, buf);
    free(buf);
    return result;
}

/* ---- Session helpers ---- */

static struct bridge_session* _session_find(struct llama_engine* e,
                                             const char*          id) {
    for (int i = 0; i < BRIDGE_MAX_SESSIONS; i++) {
        if (e->sessions[i].active &&
            strcmp(e->sessions[i].id, id) == 0) {
            return &e->sessions[i];
        }
    }
    return NULL;
}

static struct bridge_session* _session_get_or_create(struct llama_engine* e,
                                                       const char*          id) {
    struct bridge_session* s = _session_find(e, id);
    if (s) return s;
    for (int i = 0; i < BRIDGE_MAX_SESSIONS; i++) {
        if (!e->sessions[i].active) {
            memset(&e->sessions[i], 0, sizeof(e->sessions[i]));
            e->sessions[i].active = 1;
            strncpy(e->sessions[i].id, id, BRIDGE_SESSION_ID_LEN - 1);
            return &e->sessions[i];
        }
    }
    fprintf(stderr, "[llama_bridge] max sessions (%d) reached\n", BRIDGE_MAX_SESSIONS);
    return NULL;
}

static void _session_add(struct bridge_session* s,
                          const char* role, const char* content) {
    if (s->n_msgs >= BRIDGE_MAX_SESSION_MSGS) {
        fprintf(stderr, "[llama_bridge] session message limit (%d) reached\n",
                BRIDGE_MAX_SESSION_MSGS);
        return;
    }
    s->msgs[s->n_msgs].role    = _strdup_safe(role);
    s->msgs[s->n_msgs].content = _strdup_safe(content);
    s->n_msgs++;
}

static void _session_free_msgs(struct bridge_session* s) {
    for (int i = 0; i < s->n_msgs; i++) {
        free(s->msgs[i].role);
        free(s->msgs[i].content);
        s->msgs[i].role    = NULL;
        s->msgs[i].content = NULL;
    }
    s->n_msgs = 0;
}

/* =========================================================================
 * Public API — completion
 * ====================================================================== */

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

    /* Use calloc so that sessions[] is zero-initialised. */
    struct llama_engine* engine =
        (struct llama_engine*)calloc(1, sizeof(struct llama_engine));
    if (!engine) {
        llama_free(ctx);
        llama_model_free(model);
        return NULL;
    }

    engine->model = model;
    engine->ctx   = ctx;
    return (llama_engine_t)engine;
}

char* llama_engine_complete(llama_engine_t handle, const char* prompt) {
    if (!handle) {
        fprintf(stderr, "[llama_bridge] llama_engine_complete: null handle\n");
        return NULL;
    }
    return _run_inference((struct llama_engine*)handle, prompt);
}

void llama_engine_free_string(char* str) {
    free(str);
}

void llama_engine_destroy(llama_engine_t handle) {
    if (!handle) return;
    struct llama_engine* engine = (struct llama_engine*)handle;

    /* Free all active session messages. */
    for (int i = 0; i < BRIDGE_MAX_SESSIONS; i++) {
        if (engine->sessions[i].active) {
            _session_free_msgs(&engine->sessions[i]);
        }
    }

    if (engine->ctx)   { llama_free(engine->ctx);         engine->ctx   = NULL; }
    if (engine->model) { llama_model_free(engine->model); engine->model = NULL; }
    free(engine);
}

/* =========================================================================
 * Public API — chat
 * ====================================================================== */

char* llama_engine_chat(llama_engine_t handle,
                        const char*    system_msg,
                        const char*    user_msg) {
    if (!handle) return NULL;
    if (!user_msg) user_msg = "";

    struct llama_chat_message msgs[2];
    size_t n = 0;

    if (system_msg && system_msg[0] != '\0') {
        msgs[n].role    = "system";
        msgs[n].content = system_msg;
        n++;
    }
    msgs[n].role    = "user";
    msgs[n].content = user_msg;
    n++;

    return _chat_complete((struct llama_engine*)handle, msgs, n);
}

char* llama_engine_chat_with_messages(llama_engine_t handle,
                                       const char**   roles,
                                       const char**   contents,
                                       int            n_messages) {
    if (!handle || n_messages <= 0 || !roles || !contents) return NULL;

    struct llama_chat_message* msgs =
        (struct llama_chat_message*)malloc(
            (size_t)n_messages * sizeof(struct llama_chat_message));
    if (!msgs) return NULL;

    for (int i = 0; i < n_messages; i++) {
        msgs[i].role    = roles[i]    ? roles[i]    : "user";
        msgs[i].content = contents[i] ? contents[i] : "";
    }

    char* result = _chat_complete(
        (struct llama_engine*)handle, msgs, (size_t)n_messages);
    free(msgs);
    return result;
}

char* llama_engine_chat_session(llama_engine_t handle,
                                 const char*    session_id,
                                 const char*    user_msg) {
    if (!handle || !session_id || session_id[0] == '\0') return NULL;
    if (!user_msg) user_msg = "";

    struct llama_engine*   engine = (struct llama_engine*)handle;
    struct bridge_session* s      = _session_get_or_create(engine, session_id);
    if (!s) return NULL;

    /* Append user turn. */
    _session_add(s, "user", user_msg);

    /* Build llama_chat_message array from session history. */
    struct llama_chat_message* msgs =
        (struct llama_chat_message*)malloc(
            (size_t)s->n_msgs * sizeof(struct llama_chat_message));
    if (!msgs) return NULL;

    for (int i = 0; i < s->n_msgs; i++) {
        msgs[i].role    = s->msgs[i].role;
        msgs[i].content = s->msgs[i].content;
    }

    char* response = _chat_complete(engine, msgs, (size_t)s->n_msgs);
    free(msgs);

    /* Append assistant turn to history so the next call has context. */
    if (response) {
        _session_add(s, "assistant", response);
    }

    return response;
}

void llama_engine_chat_session_set_system(llama_engine_t handle,
                                           const char*    session_id,
                                           const char*    system_msg) {
    if (!handle || !session_id || session_id[0] == '\0') return;

    struct llama_engine*   engine = (struct llama_engine*)handle;
    struct bridge_session* s      = _session_get_or_create(engine, session_id);
    if (!s) return;

    /* Replace existing system message at index 0 if present. */
    if (s->n_msgs > 0 && s->msgs[0].role &&
        strcmp(s->msgs[0].role, "system") == 0) {
        free(s->msgs[0].content);
        s->msgs[0].content = _strdup_safe(
            (system_msg && system_msg[0]) ? system_msg : "");
        return;
    }

    /* Prepend new system message — nothing to prepend if empty. */
    if (!system_msg || system_msg[0] == '\0') return;
    if (s->n_msgs >= BRIDGE_MAX_SESSION_MSGS) return;

    /* Shift all existing messages up by one slot. */
    for (int i = s->n_msgs; i > 0; i--) {
        s->msgs[i] = s->msgs[i - 1];
    }
    s->msgs[0].role    = _strdup_safe("system");
    s->msgs[0].content = _strdup_safe(system_msg);
    s->n_msgs++;
}

void llama_engine_chat_session_clear(llama_engine_t handle,
                                      const char*    session_id) {
    if (!handle || !session_id) return;
    struct llama_engine*   engine = (struct llama_engine*)handle;
    struct bridge_session* s      = _session_find(engine, session_id);
    if (s) {
        _session_free_msgs(s);
        s->active = 0;
        s->id[0]  = '\0';
    }
}

char* llama_engine_chat_with_tools(llama_engine_t handle,
                                    const char**   roles,
                                    const char**   contents,
                                    int            n_messages,
                                    const char*    tools_json) {
    if (!handle || n_messages <= 0 || !roles || !contents) return NULL;
    if (!tools_json) tools_json = "[]";

    /*
     * Build a tools block and inject it into the system message.
     * Any model can reason about tools this way; models with native
     * tool-call templates will handle the JSON block correctly.
     */
    size_t block_len = strlen(BRIDGE_TOOL_HEADER) + strlen(tools_json) +
                       strlen(BRIDGE_TOOL_FOOTER) + 1;
    char* tools_block = (char*)malloc(block_len);
    if (!tools_block) return NULL;
    snprintf(tools_block, block_len, "%s%s%s",
             BRIDGE_TOOL_HEADER, tools_json, BRIDGE_TOOL_FOOTER);

    /* Determine whether the caller already has a system message. */
    int has_system = (n_messages > 0 && roles[0] &&
                      strcmp(roles[0], "system") == 0);

    int          total        = has_system ? n_messages : n_messages + 1;
    const char** new_roles    = (const char**)malloc((size_t)total * sizeof(char*));
    const char** new_contents = (const char**)malloc((size_t)total * sizeof(char*));
    char*        merged_sys   = NULL;

    if (!new_roles || !new_contents) {
        free(tools_block);
        free(new_roles);
        free(new_contents);
        return NULL;
    }

    if (has_system) {
        const char* orig = contents[0] ? contents[0] : "";
        size_t mlen = strlen(tools_block) + strlen(orig) + 2;
        merged_sys = (char*)malloc(mlen);
        if (!merged_sys) {
            free(tools_block);
            free(new_roles);
            free(new_contents);
            return NULL;
        }
        snprintf(merged_sys, mlen, "%s\n%s", tools_block, orig);
        new_roles[0]    = "system";
        new_contents[0] = merged_sys;
        for (int i = 1; i < n_messages; i++) {
            new_roles[i]    = roles[i];
            new_contents[i] = contents[i];
        }
    } else {
        new_roles[0]    = "system";
        new_contents[0] = tools_block;
        for (int i = 0; i < n_messages; i++) {
            new_roles[i + 1]    = roles[i];
            new_contents[i + 1] = contents[i];
        }
    }

    char* result = llama_engine_chat_with_messages(
        handle, new_roles, new_contents, total);

    free(tools_block);
    free(merged_sys);
    free(new_roles);
    free(new_contents);
    return result;
}

