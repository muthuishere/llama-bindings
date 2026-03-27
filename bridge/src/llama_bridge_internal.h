#ifndef LLAMA_BRIDGE_INTERNAL_H
#define LLAMA_BRIDGE_INTERNAL_H

#include "llama.h"

#include <string.h>

/*
 * llama_bridge_internal.h — internal engine structure.
 *
 * This header is NEVER exposed to language bindings.
 * Bindings receive only an opaque llama_engine_t (void*) handle.
 */

/* Default generation parameters */
#define BRIDGE_N_PREDICT      512
#define BRIDGE_N_CTX_DEFAULT  2048

/* Session storage limits */
#define BRIDGE_MAX_SESSIONS      16
#define BRIDGE_MAX_SESSION_MSGS  256
#define BRIDGE_SESSION_ID_LEN    64

/* A single heap-allocated chat message stored inside a session. */
struct bridge_msg {
    char* role;    /* heap-allocated: "system" | "user" | "assistant" */
    char* content; /* heap-allocated message text */
};

/* A named conversation session keyed by session_id. */
struct bridge_session {
    char              id[BRIDGE_SESSION_ID_LEN];
    struct bridge_msg msgs[BRIDGE_MAX_SESSION_MSGS];
    int               n_msgs;
    int               active; /* 1 if slot is in use */
};

/* The opaque engine handle returned to all callers. */
struct llama_engine {
    struct llama_model*   model;
    struct llama_context* ctx;
    struct bridge_session sessions[BRIDGE_MAX_SESSIONS];
};

#endif /* LLAMA_BRIDGE_INTERNAL_H */
