/*
 * llama_bridge_chat.c
 *
 * Chat inference implementation.
 *
 * Responsibility:
 *   - parse the JSON request
 *   - validate required fields
 *   - dispatch to the correct response-mode handler
 *   - return a normalized JSON response
 *
 * TODO(integration): replace stub inference paths with real llama.cpp calls.
 */

#include "llama_bridge_internal.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ------------------------------------------------------------------ */
/* Internal helpers                                                    */
/* ------------------------------------------------------------------ */

/* Simple response-mode detection in the request JSON. */
static int detect_response_mode(const char* request_json)
{
    /* 0 = text, 1 = json_schema, 2 = tool_call */
    if (strstr(request_json, "\"json_schema\"")) return 1;
    if (strstr(request_json, "\"tool_call\""))   return 2;
    return 0;
}

/* Build a usage block for inclusion in response JSON. */
static void build_usage(char* buf, size_t buflen,
                        int prompt_tokens, int completion_tokens)
{
    snprintf(buf, buflen,
             "\"usage\":{"
             "\"prompt_tokens\":%d,"
             "\"completion_tokens\":%d,"
             "\"total_tokens\":%d"
             "}",
             prompt_tokens,
             completion_tokens,
             prompt_tokens + completion_tokens);
}

/* ------------------------------------------------------------------ */
/* Response builders                                                   */
/* ------------------------------------------------------------------ */

static char* build_text_response(const char* text,
                                 int prompt_tokens,
                                 int completion_tokens)
{
    char usage[256];
    build_usage(usage, sizeof(usage), prompt_tokens, completion_tokens);

    /* Escape the text naively for JSON (backslash and quote). */
    size_t tlen = strlen(text);
    char* escaped = (char*)malloc(tlen * 2 + 1);
    if (!escaped) return NULL;
    size_t ei = 0;
    for (size_t i = 0; i < tlen; i++) {
        if (text[i] == '"' || text[i] == '\\') {
            escaped[ei++] = '\\';
        }
        escaped[ei++] = text[i];
    }
    escaped[ei] = '\0';

    size_t buflen = 512 + tlen * 2;
    char* resp = (char*)malloc(buflen);
    if (!resp) {
        free(escaped);
        return NULL;
    }
    snprintf(resp, buflen,
             "{"
             "\"type\":\"assistant_text\","
             "\"text\":\"%s\","
             "\"finish_reason\":\"stop\","
             "%s"
             "}",
             escaped, usage);
    free(escaped);
    return resp;
}

static char* build_json_schema_response(const char* json_text,
                                        int prompt_tokens,
                                        int completion_tokens)
{
    char usage[256];
    build_usage(usage, sizeof(usage), prompt_tokens, completion_tokens);

    size_t buflen = 512 + strlen(json_text);
    char* resp = (char*)malloc(buflen);
    if (!resp) return NULL;
    snprintf(resp, buflen,
             "{"
             "\"type\":\"structured_json\","
             "\"json\":%s,"
             "\"finish_reason\":\"stop\","
             "%s"
             "}",
             json_text, usage);
    return resp;
}

static char* build_tool_call_response(const char* tool_name,
                                      const char* arguments_json,
                                      int prompt_tokens,
                                      int completion_tokens)
{
    char usage[256];
    build_usage(usage, sizeof(usage), prompt_tokens, completion_tokens);

    size_t buflen = 512 + strlen(tool_name) + strlen(arguments_json);
    char* resp = (char*)malloc(buflen);
    if (!resp) return NULL;
    snprintf(resp, buflen,
             "{"
             "\"type\":\"tool_call\","
             "\"tool_calls\":["
             "{"
             "\"id\":\"call_1\","
             "\"name\":\"%s\","
             "\"arguments\":%s"
             "}"
             "],"
             "\"finish_reason\":\"tool_call\","
             "%s"
             "}",
             tool_name, arguments_json, usage);
    return resp;
}

/* ------------------------------------------------------------------ */
/* Public inference entry point                                        */
/* ------------------------------------------------------------------ */

char* bridge_chat_infer(llama_chat_engine_impl_t* impl,
                        const char* request_json)
{
    bridge_emit(impl->on_event, impl->user_data,
                "chat_infer_start", "chat", "generation", 0,
                "Chat inference started");

    /* Validate that "messages" key is present. */
    if (!strstr(request_json, "\"messages\"")) {
        bridge_emit(impl->on_event, impl->user_data,
                    "chat_infer_failure", "chat", NULL, 0,
                    "Missing messages field");
        return bridge_json_error("INVALID_REQUEST",
                                 "messages field is required");
    }

    int mode = detect_response_mode(request_json);

    bridge_emit(impl->on_event, impl->user_data,
                "chat_progress", "chat", "generation", 50,
                "Generating response");

    char* result = NULL;

    /*
     * TODO(integration): replace stub responses with real llama.cpp
     * inference.  The mode-switch structure below is intentional –
     * each branch will eventually call a different llama.cpp path.
     */

    if (mode == 1) {
        /* json_schema mode */
        bridge_emit(impl->on_event, impl->user_data,
                    "chat_schema_validation", "chat", "generation", 60,
                    "Schema validation");
        result = build_json_schema_response("{}", 0, 0);
    } else if (mode == 2) {
        /* tool_call mode */
        result = build_tool_call_response("unknown_tool", "{}", 0, 0);
    } else {
        /* text mode (default) */
        result = build_text_response("stub response", 0, 0);
    }

    if (!result) {
        return bridge_json_error("INTERNAL_BRIDGE_ERROR",
                                 "Failed to allocate response");
    }

    bridge_emit(impl->on_event, impl->user_data,
                "chat_infer_success", "chat", "generation", 100,
                "Chat inference complete");
    bridge_emit(impl->on_event, impl->user_data,
                "chat_complete", "chat", NULL, 100, NULL);

    return result;
}
