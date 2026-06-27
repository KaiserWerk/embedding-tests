package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// LLMClient sends requests to an OpenAI-compatible chat-completions endpoint.
type LLMClient struct {
	cfg        *AppConfig
	httpClient *http.Client
}

// NewLLMClient creates a new LLM client with the given configuration.
func NewLLMClient(cfg *AppConfig) *LLMClient {
	return &LLMClient{
		cfg:        cfg,
		httpClient: &http.Client{Timeout: cfg.Timeout},
	}
}

// Chat sends a chat-completions request and returns the API response.
// Pass a non-nil ctx to control cancellation externally; the client will
// additionally enforce cfg.Timeout around every call.
func (c *LLMClient) Chat(ctx context.Context, req ChatRequest) (*ChatResponse, error) {
	req.Model = c.cfg.OpenAI.Model

	if len(req.Tools) > 0 && req.ToolChoice == "" {
		req.ToolChoice = "auto"
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("llm: marshal request: %w", err)
	}

	tctx, cancel := context.WithTimeout(ctx, c.cfg.Timeout)
	defer cancel()

	url := c.cfg.OpenAI.Endpoint + "/v1/chat/completions"
	httpReq, err := http.NewRequestWithContext(tctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("llm: build request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.cfg.OpenAI.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.cfg.OpenAI.APIKey)
	}

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("llm: http request: %w", err)
	}
	defer resp.Body.Close()

	rawBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("llm: read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("llm: API returned HTTP %d: %s", resp.StatusCode, rawBody)
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(rawBody, &chatResp); err != nil {
		return nil, fmt.Errorf("llm: unmarshal response: %w", err)
	}

	if chatResp.Error != nil {
		return nil, fmt.Errorf("llm: API error (%s): %s", chatResp.Error.Type, chatResp.Error.Message)
	}

	return &chatResp, nil
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

// ChatRequest is the body sent to /v1/chat/completions.
type ChatRequest struct {
	Model      string    `json:"model"`
	Messages   []Message `json:"messages"`
	Tools      []Tool    `json:"tools,omitempty"`
	ToolChoice string    `json:"tool_choice,omitempty"` // "auto" | "none" | "required"
}

// Message represents a single entry in the conversation.
type Message struct {
	Role       string     `json:"role"` // "system" | "user" | "assistant" | "tool"
	Content    string     `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`   // set when role == "assistant"
	ToolCallID string     `json:"tool_call_id,omitempty"` // set when role == "tool"
	Name       string     `json:"name,omitempty"`         // tool name for role == "tool"
}

// Tool is an OpenAI tool definition wrapping a function.
type Tool struct {
	Type     string   `json:"type"` // always "function"
	Function Function `json:"function"`
}

// Function carries the schema of a callable function.
type Function struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"` // JSON Schema object
}

// ToolCall is emitted by the assistant when it decides to call a tool.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"` // "function"
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction carries the tool name and its serialised arguments.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON-encoded arguments object
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

// ChatResponse is the top-level response from /v1/chat/completions.
type ChatResponse struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"`
	Model   string    `json:"model"`
	Choices []Choice  `json:"choices"`
	Error   *APIError `json:"error,omitempty"`
}

// Choice is one completion candidate.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"` // "stop" | "tool_calls" | "length" …
}

// APIError carries an error from the OpenAI-compatible API.
type APIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}
