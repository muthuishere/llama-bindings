/*
 * llama_bridge_chat.cpp
 *
 * Chat inference implementation using llama.cpp.
 */

#include "llama_bridge_internal.h"
#include "llama.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <ctype.h>

/* ------------------------------------------------------------------ */
/* Message struct                                                      */
/* ------------------------------------------------------------------ */

struct ChatMessage {
    std::string role;
    std::string content;
};

/* ------------------------------------------------------------------ */
/* JSON helpers                                                        */
/* ------------------------------------------------------------------ */

/* Escape a string for embedding in a JSON value */
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() * 2);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char)c;
                }
        }
    }
    return out;
}

/* Extract a JSON object value for the given key. Returns "" if not found. */
static std::string extract_json_object(const char* json, const char* key) {
    if (!json || !key) return "";
    std::string pattern = std::string("\"") + key + "\"";
    const char* p = strstr(json, pattern.c_str());
    if (!p) return "";
    p += pattern.size();
    while (*p && *p != ':') p++;
    if (!*p) return "";
    p++;
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != '{') return "";
    const char* start = p;
    int depth = 0;
    while (*p) {
        if (*p == '"') {
            p++;
            while (*p && *p != '"') {
                if (*p == '\\') p++;
                if (*p) p++;
            }
            if (*p) p++;
            continue;
        }
        if (*p == '{') depth++;
        else if (*p == '}') {
            depth--;
            if (depth == 0) { p++; break; }
        }
        p++;
    }
    return std::string(start, p);
}

/* Parse messages array from request JSON */
static std::vector<ChatMessage> parse_messages(const char* json) {
    std::vector<ChatMessage> messages;
    if (!json) return messages;

    const char* p = strstr(json, "\"messages\"");
    if (!p) return messages;
    p += strlen("\"messages\"");
    while (*p && *p != '[') p++;
    if (!*p) return messages;
    p++; // skip '['

    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == ']' || !*p) break;
        if (*p != '{') { p++; continue; }

        /* Find end of this object tracking depth and strings */
        const char* obj_start = p;
        int depth = 0;
        const char* q = p;
        while (*q) {
            if (*q == '"') {
                q++;
                while (*q && *q != '"') {
                    if (*q == '\\') q++;
                    if (*q) q++;
                }
                if (*q) q++;
                continue;
            }
            if (*q == '{') depth++;
            else if (*q == '}') {
                depth--;
                if (depth == 0) { q++; break; }
            }
            q++;
        }

        std::string obj(obj_start, q);
        char* role    = bridge_json_get_string(obj.c_str(), "role");
        char* content = bridge_json_get_string(obj.c_str(), "content");
        if (role && content) {
            messages.push_back({std::string(role), std::string(content)});
        }
        if (role)    free(role);
        if (content) free(content);

        p = q;
        while (*p && (*p == ',' || isspace((unsigned char)*p))) p++;
    }
    return messages;
}

/* Extract tools array as a system prompt string */
static std::string extract_tools_system_prompt(const char* json) {
    if (!json) return "";
    const char* p = strstr(json, "\"tools\"");
    if (!p) return "";
    p += strlen("\"tools\"");
    while (*p && *p != '[') p++;
    if (!*p) return "";
    p++; // skip '['

    std::string prompt =
        "You have access to the following tools. "
        "When you want to call a tool, respond ONLY with a JSON object exactly like this:\n"
        "{\"name\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n\n"
        "Available tools:\n";

    while (*p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p == ']' || !*p) break;
        if (*p != '{') { p++; continue; }

        const char* obj_start = p;
        int depth = 0;
        const char* q = p;
        while (*q) {
            if (*q == '"') {
                q++;
                while (*q && *q != '"') {
                    if (*q == '\\') q++;
                    if (*q) q++;
                }
                if (*q) q++;
                continue;
            }
            if (*q == '{') depth++;
            else if (*q == '}') {
                depth--;
                if (depth == 0) { q++; break; }
            }
            q++;
        }
        std::string obj(obj_start, q);
        char* name = bridge_json_get_string(obj.c_str(), "name");
        char* desc = bridge_json_get_string(obj.c_str(), "description");
        if (name) {
            prompt += "- ";
            prompt += name;
            if (desc && *desc) { prompt += ": "; prompt += desc; }
            prompt += "\n";
        }
        if (name) free(name);
        if (desc)  free(desc);

        p = q;
        while (*p && (*p == ',' || isspace((unsigned char)*p))) p++;
    }
    prompt += "\nRespond ONLY with the JSON tool call. Do not add any other text.";
    return prompt;
}

/* Try to extract the first JSON object {...} from text */
static std::string try_extract_json_object(const std::string& text) {
    size_t start = text.find('{');
    if (start == std::string::npos) return "";
    int depth = 0;
    for (size_t i = start; i < text.size(); i++) {
        if (text[i] == '"') {
            i++;
            while (i < text.size() && text[i] != '"') {
                if (text[i] == '\\') i++;
                i++;
            }
            continue;
        }
        if (text[i] == '{') depth++;
        else if (text[i] == '}') {
            depth--;
            if (depth == 0) return text.substr(start, i - start + 1);
        }
    }
    return "";
}

/* ------------------------------------------------------------------ */
/* Response builders                                                   */
/* ------------------------------------------------------------------ */

static void build_usage(char* buf, size_t buflen, int pt, int ct) {
    snprintf(buf, buflen,
             "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}",
             pt, ct, pt + ct);
}

static char* build_text_response(const std::string& text, int pt, int ct) {
    char usage[256];
    build_usage(usage, sizeof(usage), pt, ct);
    std::string escaped = json_escape(text);
    size_t buflen = 512 + escaped.size() + strlen(usage);
    char* resp = (char*)malloc(buflen);
    if (!resp) return NULL;
    snprintf(resp, buflen,
             "{\"type\":\"assistant_text\",\"text\":\"%s\",\"finish_reason\":\"stop\",%s}",
             escaped.c_str(), usage);
    return resp;
}

static char* build_json_schema_response(const std::string& json_text, int pt, int ct) {
    char usage[256];
    build_usage(usage, sizeof(usage), pt, ct);
    size_t buflen = 512 + json_text.size() + strlen(usage);
    char* resp = (char*)malloc(buflen);
    if (!resp) return NULL;
    snprintf(resp, buflen,
             "{\"type\":\"structured_json\",\"json\":%s,\"finish_reason\":\"stop\",%s}",
             json_text.c_str(), usage);
    return resp;
}

static char* build_tool_call_response(const std::string& tool_name,
                                       const std::string& arguments_json,
                                       int pt, int ct) {
    char usage[256];
    build_usage(usage, sizeof(usage), pt, ct);
    std::string esc_name = json_escape(tool_name);
    size_t buflen = 512 + esc_name.size() + arguments_json.size() + strlen(usage);
    char* resp = (char*)malloc(buflen);
    if (!resp) return NULL;
    snprintf(resp, buflen,
             "{\"type\":\"tool_call\",\"tool_calls\":[{\"id\":\"call_1\",\"name\":\"%s\",\"arguments\":%s}],"
             "\"finish_reason\":\"tool_call\",%s}",
             esc_name.c_str(), arguments_json.c_str(), usage);
    return resp;
}

/* ------------------------------------------------------------------ */
/* Core generation                                                     */
/* ------------------------------------------------------------------ */

static std::string run_generation(struct llama_model* model,
                                   struct llama_context* ctx,
                                   const std::string& prompt,
                                   int max_tokens,
                                   float temperature,
                                   float top_p,
                                   int top_k,
                                   int* prompt_tokens_out,
                                   int* completion_tokens_out)
{
    const struct llama_vocab* vocab = llama_model_get_vocab(model);

    /* Tokenize the prompt */
    int n_ctx = (int)llama_n_ctx(ctx);
    std::vector<llama_token> tokens(n_ctx);
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                                  tokens.data(), (int32_t)tokens.size(),
                                  true, true);
    if (n_tokens < 0) {
        /* Buffer too small; resize and retry */
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t)prompt.size(),
                                  tokens.data(), (int32_t)tokens.size(),
                                  true, true);
    }
    if (n_tokens <= 0) return "";
    tokens.resize(n_tokens);

    if (prompt_tokens_out) *prompt_tokens_out = n_tokens;

    /* Clear KV cache before generation */
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) llama_memory_clear(mem, true);

    /* Decode the prompt */
    {
        struct llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
        if (llama_decode(ctx, batch) != 0) return "";
    }

    /* Build sampler chain */
    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler* sampler = llama_sampler_chain_init(sparams);
    if (top_k > 0)
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    /* Generation loop */
    std::string output;
    int n_decoded = 0;

    while (n_decoded < max_tokens) {
        llama_token tok = llama_sampler_sample(sampler, ctx, -1);
        llama_sampler_accept(sampler, tok);

        if (llama_vocab_is_eog(vocab, tok)) break;

        char piece[256];
        int n = llama_token_to_piece(vocab, tok, piece, (int32_t)sizeof(piece), 0, true);
        if (n > 0) output.append(piece, n);

        n_decoded++;

        struct llama_batch next_batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(ctx, next_batch) != 0) break;
    }

    llama_sampler_free(sampler);

    if (completion_tokens_out) *completion_tokens_out = n_decoded;
    return output;
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

    /* Validate messages */
    {
        const char* msg_ptr = strstr(request_json, "\"messages\"");
        if (!msg_ptr) {
            return bridge_json_error("INVALID_REQUEST", "messages field is required");
        }
        msg_ptr += strlen("\"messages\"");
        while (*msg_ptr == ' ' || *msg_ptr == '\t' || *msg_ptr == ':') msg_ptr++;
        if (*msg_ptr != '[' || *(msg_ptr + 1) == ']') {
            return bridge_json_error("INVALID_REQUEST",
                                     "messages array must contain at least one message");
        }
    }

    /* Detect response mode: 0=text, 1=json_schema, 2=tool_call */
    int mode = 0;
    if (strstr(request_json, "\"json_schema\"")) mode = 1;
    else if (strstr(request_json, "\"tool_call\"")) mode = 2;

    /* Parse generation parameters */
    double temperature = 0.8;
    int    max_tokens  = 512;
    float  top_p       = 0.95f;
    int    top_k       = 40;
    {
        std::string gen_json = extract_json_object(request_json, "generation");
        if (!gen_json.empty()) {
            double t = bridge_json_get_double(gen_json.c_str(), "temperature", 0.0);
            if (t > 0.0) temperature = t;
            int mt = bridge_json_get_int(gen_json.c_str(), "max_output_tokens", 0);
            if (mt > 0) max_tokens = mt;
            double tp = bridge_json_get_double(gen_json.c_str(), "top_p", 0.0);
            if (tp > 0.0) top_p = (float)tp;
            int tk = bridge_json_get_int(gen_json.c_str(), "top_k", 0);
            if (tk > 0) top_k = tk;
        }
    }

    /* Parse messages */
    std::vector<ChatMessage> messages = parse_messages(request_json);

    /* For tool_call mode, inject a system prompt about available tools */
    if (mode == 2) {
        std::string tools_prompt = extract_tools_system_prompt(request_json);
        if (!tools_prompt.empty()) {
            /* Prepend or replace system message */
            bool has_system = false;
            for (auto& m : messages) {
                if (m.role == "system") {
                    m.content = tools_prompt + "\n\n" + m.content;
                    has_system = true;
                    break;
                }
            }
            if (!has_system) {
                messages.insert(messages.begin(), {"system", tools_prompt});
            }
        }
    }

    /* For json_schema mode, inject a system prompt asking for JSON output */
    if (mode == 1) {
        std::string schema_obj = extract_json_object(request_json, "schema");
        std::string schema_prompt = "Respond with valid JSON only. No explanation, no markdown, just the JSON object.";
        if (!schema_obj.empty()) {
            schema_prompt += " The JSON must match this schema: " + schema_obj;
        }
        bool has_system = false;
        for (auto& m : messages) {
            if (m.role == "system") {
                m.content = schema_prompt + "\n\n" + m.content;
                has_system = true;
                break;
            }
        }
        if (!has_system) {
            messages.insert(messages.begin(), {"system", schema_prompt});
        }
    }

    /* Apply chat template */
    std::string formatted_prompt;
    {
        const char* tmpl = llama_model_chat_template(impl->llama_model, NULL);

        std::vector<llama_chat_message> chat_msgs;
        chat_msgs.reserve(messages.size());
        for (const auto& m : messages) {
            chat_msgs.push_back({m.role.c_str(), m.content.c_str()});
        }

        /* First call: get needed buffer size */
        int32_t needed = llama_chat_apply_template(tmpl,
                                                    chat_msgs.data(),
                                                    chat_msgs.size(),
                                                    true,
                                                    nullptr, 0);
        if (needed < 0) {
            /* Template not recognized. Detect Gemma 4 style (<|turn>/<turn|>) or
               classic Gemma (<start_of_turn>/<end_of_turn>) and build manually. */
            bool is_gemma4 = tmpl && (strstr(tmpl, "<|turn>") != nullptr ||
                                      strstr(tmpl, "<turn|>") != nullptr);
            bool is_gemma  = tmpl && strstr(tmpl, "start_of_turn") != nullptr;

            if (is_gemma4) {
                /* Gemma 4 format: <|turn>role\ncontent<turn|>\n */
                formatted_prompt = "";
                for (const auto& m : messages) {
                    std::string role = m.role;
                    if (role == "assistant") role = "model";
                    formatted_prompt += "<|turn>" + role + "\n" + m.content + "<turn|>\n";
                }
                formatted_prompt += "<|turn>model\n";
            } else if (is_gemma) {
                /* Classic Gemma 2/3 format */
                formatted_prompt = "";
                for (const auto& m : messages) {
                    std::string role = m.role;
                    if (role == "assistant") role = "model";
                    formatted_prompt += "<start_of_turn>" + role + "\n" + m.content + "<end_of_turn>\n";
                }
                formatted_prompt += "<start_of_turn>model\n";
            } else {
                /* Generic fallback: chatml */
                formatted_prompt = "";
                for (const auto& m : messages) {
                    formatted_prompt += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>\n";
                }
                formatted_prompt += "<|im_start|>assistant\n";
            }
        } else {
            formatted_prompt.resize((size_t)needed + 1, '\0');
            llama_chat_apply_template(tmpl,
                                      chat_msgs.data(),
                                      chat_msgs.size(),
                                      true,
                                      &formatted_prompt[0],
                                      (int32_t)formatted_prompt.size());
            /* Trim the null terminator off the std::string */
            if (!formatted_prompt.empty() && formatted_prompt.back() == '\0') {
                formatted_prompt.pop_back();
            }
        }
    }

    bridge_emit(impl->on_event, impl->user_data,
                "chat_progress", "chat", "generation", 30,
                "Prompt ready, generating");

    /* Run generation */
    int prompt_tokens = 0, completion_tokens = 0;
    std::string output = run_generation(
        impl->llama_model, impl->llama_ctx,
        formatted_prompt,
        max_tokens,
        (float)temperature,
        top_p, top_k,
        &prompt_tokens, &completion_tokens);

    bridge_emit(impl->on_event, impl->user_data,
                "chat_progress", "chat", "generation", 90,
                "Generation complete");

    char* result = NULL;

    if (mode == 1) {
        /* json_schema mode: try to extract a JSON object from output */
        std::string json_obj = try_extract_json_object(output);
        if (json_obj.empty()) json_obj = "{}";
        result = build_json_schema_response(json_obj, prompt_tokens, completion_tokens);
    } else if (mode == 2) {
        /* tool_call mode: parse tool call JSON from output */
        std::string json_obj = try_extract_json_object(output);
        std::string tool_name = "unknown_tool";
        std::string arguments = "{}";
        if (!json_obj.empty()) {
            char* name = bridge_json_get_string(json_obj.c_str(), "name");
            if (name) { tool_name = name; free(name); }
            std::string args = extract_json_object(json_obj.c_str(), "arguments");
            if (!args.empty()) arguments = args;
        }
        result = build_tool_call_response(tool_name, arguments,
                                           prompt_tokens, completion_tokens);
    } else {
        /* text mode */
        result = build_text_response(output, prompt_tokens, completion_tokens);
    }

    if (!result) {
        return bridge_json_error("INTERNAL_BRIDGE_ERROR", "Failed to allocate response");
    }

    bridge_emit(impl->on_event, impl->user_data,
                "chat_infer_success", "chat", "generation", 100,
                "Chat inference complete");
    bridge_emit(impl->on_event, impl->user_data,
                "chat_complete", "chat", NULL, 100, NULL);

    return result;
}
