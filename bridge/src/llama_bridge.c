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

/*
 * Set or replace the system message at the head of the session.
 * If a system message already exists at index 0, its content is replaced.
 * Otherwise a new system message is prepended (existing messages shift up).
 */
static void _session_set_system(struct bridge_session* s,
                                 const char*            system_msg) {
    if (!system_msg || system_msg[0] == '\0') return;

    /* Replace existing system message if present at index 0. */
    if (s->n_msgs > 0 && s->msgs[0].role &&
        strcmp(s->msgs[0].role, "system") == 0) {
        free(s->msgs[0].content);
        s->msgs[0].content = _strdup_safe(system_msg);
        return;
    }

    /* Prepend a new system message. */
    if (s->n_msgs >= BRIDGE_MAX_SESSION_MSGS) return;
    for (int i = s->n_msgs; i > 0; i--) {
        s->msgs[i] = s->msgs[i - 1];
    }
    s->msgs[0].role    = _strdup_safe("system");
    s->msgs[0].content = _strdup_safe(system_msg);
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

/*
 * JSON string encoder — returns a heap-allocated, double-quoted,
 * escape-safe JSON string representation of `s`.
 * Returns NULL on allocation failure.
 */
static char* _json_encode_string(const char* s) {
    if (!s) return _strdup_safe("\"\"");
    size_t out_cap = strlen(s) * 2 + 4;
    char*  out     = (char*)malloc(out_cap);
    if (!out) return NULL;
    size_t i = 0, j = 0;
    out[j++] = '"';
    while (s[i]) {
        if (j + 6 >= out_cap) {
            out_cap *= 2;
            char* tmp = (char*)realloc(out, out_cap);
            if (!tmp) { free(out); return NULL; }
            out = tmp;
        }
        switch (s[i]) {
            case '"':  out[j++] = '\\'; out[j++] = '"';  break;
            case '\\': out[j++] = '\\'; out[j++] = '\\'; break;
            case '\n': out[j++] = '\\'; out[j++] = 'n';  break;
            case '\r': out[j++] = '\\'; out[j++] = 'r';  break;
            case '\t': out[j++] = '\\'; out[j++] = 't';  break;
            default:   out[j++] = s[i]; break;
        }
        i++;
    }
    out[j++] = '"';
    out[j]   = '\0';
    return out;
}

/* Build {"role":"assistant","content":"<escaped>"} */
static char* _build_message_json(const char* content) {
    char* ec = _json_encode_string(content ? content : "");
    if (!ec) return NULL;
    /* "assistant" is always the role returned by inference */
    size_t len = strlen("{\"role\":\"assistant\",\"content\":}") + strlen(ec) + 1;
    char*  out = (char*)malloc(len);
    if (!out) { free(ec); return NULL; }
    snprintf(out, len, "{\"role\":\"assistant\",\"content\":%s}", ec);
    free(ec);
    return out;
}

/* Build {"role":"assistant","content":"<escaped>","sessionId":"<sid>","messageCount":<n>} */
static char* _build_object_json(const char* content,
                                 const char* session_id,
                                 int         msg_count) {
    char* ec  = _json_encode_string(content    ? content    : "");
    char* esc = _json_encode_string(session_id ? session_id : "");
    if (!ec || !esc) { free(ec); free(esc); return NULL; }
    size_t len = strlen("{\"role\":\"assistant\",\"content\":,\"sessionId\":,\"messageCount\":}")
                 + strlen(ec) + strlen(esc) + 32;
    char* out = (char*)malloc(len);
    if (!out) { free(ec); free(esc); return NULL; }
    snprintf(out, len,
             "{\"role\":\"assistant\",\"content\":%s,\"sessionId\":%s,\"messageCount\":%d}",
             ec, esc, msg_count);
    free(ec);
    free(esc);
    return out;
}

/*
 * Core session-chat helper: adds the incoming messages to the session,
 * runs inference, stores the assistant response, and returns the raw
 * assistant text (heap-allocated, caller must free).
 *
 * Message injection order: system → assistant → tool → user.
 */
static char* _session_chat_run(struct llama_engine* engine,
                                const char*          session_id,
                                const char*          system_message,
                                const char*          user_message,
                                const char*          assistant_message,
                                const char*          tool_message) {
    if (!session_id || session_id[0] == '\0') return NULL;

    struct bridge_session* s = _session_get_or_create(engine, session_id);
    if (!s) return NULL;

    /* 1. Set/update system message */
    if (system_message && system_message[0] != '\0') {
        _session_set_system(s, system_message);
    }

    /* 2. Inject prior assistant turn (few-shot / correction) */
    if (assistant_message && assistant_message[0] != '\0') {
        _session_add(s, "assistant", assistant_message);
    }

    /* 3. Inject tool response */
    if (tool_message && tool_message[0] != '\0') {
        _session_add(s, "tool", tool_message);
    }

    /* 4. Add user message */
    if (user_message && user_message[0] != '\0') {
        _session_add(s, "user", user_message);
    }

    if (s->n_msgs == 0) return NULL;

    /* 5. Build message array and run inference */
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

    /* 6. Store assistant response in session */
    if (response) {
        _session_add(s, "assistant", response);
    }

    return response; /* caller frees */
}

/* =========================================================================
 * Public API — completion
 * ====================================================================== */

llama_engine_t llama_engine_create(const char* model_path) {
    if (!model_path || model_path[0] == '\0') {
        fprintf(stderr, "[llama_bridge] llama_engine_create: empty model path\n");
        return NULL;
    }

    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    struct llama_model* model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "[llama_bridge] failed to load model: %s\n", model_path);
        return NULL;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = BRIDGE_N_CTX_DEFAULT;

    struct llama_context* ctx = llama_new_context_with_params(model, cparams);
    if (!ctx) {
        fprintf(stderr, "[llama_bridge] failed to create context\n");
        llama_model_free(model);
        return NULL;
    }

    /* calloc ensures sessions[] is zero-initialised */
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
                        const char*    session_id,
                        const char*    system_message,
                        const char*    user_message,
                        const char*    assistant_message,
                        const char*    tool_message) {
    if (!handle) return NULL;

    struct llama_engine* engine = (struct llama_engine*)handle;

    char* response = _session_chat_run(engine, session_id,
                                       system_message, user_message,
                                       assistant_message, tool_message);

    char* json = _build_message_json(response);
    free(response);
    return json;
}

char* llama_engine_chat_with_object(llama_engine_t handle,
                                     const char*    session_id,
                                     const char*    system_message,
                                     const char*    user_message,
                                     const char*    assistant_message,
                                     const char*    tool_message) {
    if (!handle) return NULL;

    struct llama_engine*   engine = (struct llama_engine*)handle;
    struct bridge_session* s      = _session_find(engine, session_id);
    /* We need message count AFTER the call, so run chat first. */

    char* response = _session_chat_run(engine, session_id,
                                       system_message, user_message,
                                       assistant_message, tool_message);

    /* Look up session again (created by _session_chat_run if it didn't exist) */
    s = _session_find(engine, session_id);
    int msg_count = s ? s->n_msgs : 0;

    char* json = _build_object_json(response, session_id, msg_count);
    free(response);
    return json;
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
