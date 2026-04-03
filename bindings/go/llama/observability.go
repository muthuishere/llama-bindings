package llama

// EventCallback is called for each observability event emitted by the bridge.
// The event parameter contains a JSON-encoded event payload (see spec §10.2).
// Callbacks must not block; they are called synchronously from CGO.
type EventCallback func(event Event)

// Event is a parsed observability event from the bridge.
type Event struct {
	// Event is the event name, e.g. "chat_infer_start".
	Event      string `json:"event"`
	EngineType string `json:"engine_type"`
	Stage      string `json:"stage,omitempty"`
	Progress   int    `json:"progress,omitempty"`
	Message    string `json:"message,omitempty"`
	// Usage is populated on completion events.
	Usage       *UsageInfo `json:"usage,omitempty"`
	PartialText string     `json:"partial_text,omitempty"`
	TimestampMs int64      `json:"timestamp_ms,omitempty"`
}
