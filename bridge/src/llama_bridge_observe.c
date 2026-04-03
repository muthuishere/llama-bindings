/*
 * llama_bridge_observe.c
 *
 * Observability helpers: event emission and timestamp utilities.
 */

#include "llama_bridge_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ------------------------------------------------------------------ */
/* Timestamp                                                           */
/* ------------------------------------------------------------------ */

int64_t bridge_now_ms(void)
{
    struct timespec ts;
#if defined(_WIN32)
    /* Windows: use GetSystemTimeAsFileTime equivalent via timespec_get */
    timespec_get(&ts, TIME_UTC);
#else
    clock_gettime(CLOCK_REALTIME, &ts);
#endif
    return (int64_t)ts.tv_sec * 1000LL + (int64_t)(ts.tv_nsec / 1000000LL);
}

/* ------------------------------------------------------------------ */
/* Event emission                                                      */
/* ------------------------------------------------------------------ */

void bridge_emit_event(llama_event_cb cb, void* user_data,
                       const char* event_json)
{
    if (cb && event_json) {
        cb(event_json, user_data);
    }
}

void bridge_emit(llama_event_cb cb, void* user_data,
                 const char* event_name,
                 const char* engine_type,
                 const char* stage,
                 int progress,
                 const char* message)
{
    if (!cb) {
        return;
    }

    int64_t ts = bridge_now_ms();
    char buf[1024];
    int  pos = 0;

    pos += snprintf(buf + pos, sizeof(buf) - (size_t)pos,
                    "{\"event\":\"%s\",\"engine_type\":\"%s\"",
                    event_name ? event_name : "",
                    engine_type ? engine_type : "");

    if (stage && stage[0]) {
        pos += snprintf(buf + pos, sizeof(buf) - (size_t)pos,
                        ",\"stage\":\"%s\"", stage);
    }

    pos += snprintf(buf + pos, sizeof(buf) - (size_t)pos,
                    ",\"progress\":%d", progress);

    if (message && message[0]) {
        pos += snprintf(buf + pos, sizeof(buf) - (size_t)pos,
                        ",\"message\":\"%s\"", message);
    }

    pos += snprintf(buf + pos, sizeof(buf) - (size_t)pos,
                    ",\"timestamp_ms\":%lld}", (long long)ts);

    if (pos > 0 && pos < (int)sizeof(buf)) {
        cb(buf, user_data);
    }
}
