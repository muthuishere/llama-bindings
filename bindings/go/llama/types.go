package llama

// Message represents a single chat message.
type Message struct {
	Role     string `json:"role"`               // system | user | assistant | tool
	Content  string `json:"content"`
	ToolName string `json:"tool_name,omitempty"` // set when Role == "tool"
}

// Schema describes a JSON schema for structured output mode.
type Schema struct {
	Name   string      `json:"name"`
	Schema interface{} `json:"schema"`
}

// ToolDefinition describes a callable tool exposed to the model.
type ToolDefinition struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  interface{} `json:"parameters"`
}

// GenerationOptions controls the sampling parameters for a chat request.
type GenerationOptions struct {
	Temperature     float64  `json:"temperature,omitempty"`
	MaxOutputTokens int      `json:"max_output_tokens,omitempty"`
	TopP            float64  `json:"top_p,omitempty"`
	TopK            int      `json:"top_k,omitempty"`
	Stop            []string `json:"stop,omitempty"`
}

// ResponseMode determines how the model should format its reply.
type ResponseMode string

const (
	ResponseModeText       ResponseMode = "text"
	ResponseModeJSONSchema ResponseMode = "json_schema"
	ResponseModeToolCall   ResponseMode = "tool_call"
)

// ChatRequest is the top-level input for a chat inference call.
type ChatRequest struct {
	Messages     []Message        `json:"messages"`
	ResponseMode ResponseMode     `json:"response_mode,omitempty"`
	Schema       *Schema          `json:"schema,omitempty"`
	Tools        []ToolDefinition `json:"tools,omitempty"`
	ToolChoice   string           `json:"tool_choice,omitempty"`
	Generation   GenerationOptions `json:"generation,omitempty"`
}

// UsageInfo contains token-usage accounting from a chat response.
type UsageInfo struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ToolCall represents a single tool invocation returned by the model.
type ToolCall struct {
	ID        string      `json:"id"`
	Name      string      `json:"name"`
	Arguments interface{} `json:"arguments"`
}

// ChatResponse is the normalized output from a chat inference call.
type ChatResponse struct {
	// Type is one of: "assistant_text", "structured_json", "tool_call", "error".
	Type         string      `json:"type"`
	Text         string      `json:"text,omitempty"`
	JSON         interface{} `json:"json,omitempty"`
	ToolCalls    []ToolCall  `json:"tool_calls,omitempty"`
	FinishReason string      `json:"finish_reason,omitempty"`
	Usage        UsageInfo   `json:"usage"`
	Error        *ErrorInfo  `json:"error,omitempty"`
}

// ErrorInfo carries a structured error code and message.
type ErrorInfo struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// LoadOptions are passed when creating a chat or embed engine.
type LoadOptions struct {
	// OnEvent is called for each observability event during load and inference.
	// It may be nil.
	OnEvent EventCallback
}

// ChatOptions control per-call behaviour of Chat().
type ChatOptions struct {
	Temperature     float64
	MaxOutputTokens int
	TopP            float64
	TopK            int
	Stop            []string
	// OnEvent overrides or supplements the engine-level callback for this call.
	OnEvent EventCallback
}

// EmbedOptions control per-call behaviour of Embed().
type EmbedOptions struct {
	// OnEvent overrides or supplements the engine-level callback for this call.
	OnEvent EventCallback
}
