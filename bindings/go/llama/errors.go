package llama

import "fmt"

// ErrorCode classifies bridge-level failures.
type ErrorCode string

const (
	ErrCodeModelLoadFailed      ErrorCode = "MODEL_LOAD_FAILED"
	ErrCodeInvalidRequest       ErrorCode = "INVALID_REQUEST"
	ErrCodeInferenceFailed      ErrorCode = "INFERENCE_FAILED"
	ErrCodeSchemaValidation     ErrorCode = "SCHEMA_VALIDATION_FAILED"
	ErrCodeToolValidation       ErrorCode = "TOOL_VALIDATION_FAILED"
	ErrCodeEngineClosed         ErrorCode = "ENGINE_CLOSED"
	ErrCodeInternalBridgeError  ErrorCode = "INTERNAL_BRIDGE_ERROR"
)

// LlamaError is returned when the bridge or model reports an error.
type LlamaError struct {
	Code    ErrorCode
	Message string
}

func (e *LlamaError) Error() string {
	return fmt.Sprintf("llama [%s]: %s", e.Code, e.Message)
}

// newLlamaError constructs a LlamaError from raw code and message strings.
func newLlamaError(code, message string) *LlamaError {
	return &LlamaError{
		Code:    ErrorCode(code),
		Message: message,
	}
}
