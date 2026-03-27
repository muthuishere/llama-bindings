#ifndef LLAMA_BRIDGE_INTERNAL_H
#define LLAMA_BRIDGE_INTERNAL_H

#include "llama.h"

/*
 * llama_bridge_internal.h — internal engine structure.
 *
 * This struct is NEVER exposed to language bindings.
 * Bindings receive only an opaque llama_engine_t (void*) handle.
 */

struct llama_engine {
    struct llama_model*   model;
    struct llama_context* ctx;
};

/* Default generation parameters */
#define BRIDGE_N_PREDICT      512
#define BRIDGE_N_CTX_DEFAULT  2048

#endif /* LLAMA_BRIDGE_INTERNAL_H */
