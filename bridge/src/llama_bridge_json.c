/*
 * llama_bridge_json.c
 *
 * Minimal JSON helpers used by the bridge.
 *
 * These are intentionally simple and handle only the subset of JSON
 * needed by the bridge ABI (flat objects, string/int/double values).
 * A full JSON parser is not required at this layer.
 */

#include "llama_bridge_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ------------------------------------------------------------------ */
/* Internal: find a key's value start in a JSON object string         */
/* ------------------------------------------------------------------ */

/*
 * Locate the start of the value for `key` in `json`.
 * Returns pointer into json just past the colon, or NULL if not found.
 * This is intentionally naive and works only for flat objects.
 */
static const char* find_value(const char* json, const char* key)
{
    if (!json || !key) return NULL;

    size_t klen = strlen(key);
    /* Build the search pattern: "key" */
    char pattern[256];
    if (klen + 4 > sizeof(pattern)) return NULL;
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);

    const char* pos = strstr(json, pattern);
    if (!pos) return NULL;

    pos += strlen(pattern);
    while (*pos && (*pos == ' ' || *pos == '\t' || *pos == '\r' || *pos == '\n')) {
        pos++;
    }
    if (*pos != ':') return NULL;
    pos++;
    while (*pos && (*pos == ' ' || *pos == '\t' || *pos == '\r' || *pos == '\n')) {
        pos++;
    }
    return pos;
}

/* ------------------------------------------------------------------ */
/* Public helpers                                                      */
/* ------------------------------------------------------------------ */

char* bridge_json_get_string(const char* json, const char* key)
{
    const char* val = find_value(json, key);
    if (!val || *val != '"') return NULL;
    val++; /* skip opening quote */

    /* Find the closing quote, respecting backslash escapes. */
    const char* end = val;
    while (*end && *end != '"') {
        if (*end == '\\' && *(end + 1)) {
            end++; /* skip escaped char */
        }
        end++;
    }

    size_t len = (size_t)(end - val);
    char* result = (char*)malloc(len + 1);
    if (!result) return NULL;
    memcpy(result, val, len);
    result[len] = '\0';
    return result;
}

int bridge_json_get_int(const char* json, const char* key, int default_val)
{
    const char* val = find_value(json, key);
    if (!val) return default_val;
    if (*val == '"') {
        /* String-encoded integer */
        val++;
    }
    char* end;
    long v = strtol(val, &end, 10);
    if (end == val) return default_val;
    return (int)v;
}

double bridge_json_get_double(const char* json, const char* key,
                              double default_val)
{
    const char* val = find_value(json, key);
    if (!val) return default_val;
    char* end;
    double v = strtod(val, &end);
    if (end == val) return default_val;
    return v;
}

char* bridge_json_error(const char* code, const char* message)
{
    /* Escape code and message conservatively. */
    size_t buflen = 256 + strlen(code) + strlen(message);
    char* buf = (char*)malloc(buflen);
    if (!buf) return NULL;
    snprintf(buf, buflen,
             "{"
             "\"type\":\"error\","
             "\"error\":{"
             "\"code\":\"%s\","
             "\"message\":\"%s\""
             "}"
             "}",
             code, message);
    return buf;
}
