package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type EmbeddingClient struct {
	cfg        *AppConfig
	httpClient *http.Client
}

// NewClient creates a new LLM client with the given configuration.
func NewEmbeddingClient(cfg *AppConfig) *EmbeddingClient {
	return &EmbeddingClient{
		cfg:        cfg,
		httpClient: &http.Client{Timeout: cfg.Timeout},
	}
}

type EmbeddingRequest struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

type (
	EmbeddingResponse struct {
		Error  *APIError `json:"error,omitempty"`
		Data   []Data    `json:"data"`
		Model  string    `json:"model"`
		Object string    `json:"object"`
		Usage  Usage     `json:"usage"`
	}
	Data struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
		Object    string    `json:"object"`
	}
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	}
)

func (c *EmbeddingClient) GetEmbedding(ctx context.Context, input string) (*EmbeddingResponse, error) {
	req := EmbeddingRequest{
		Input: input,
		Model: c.cfg.Embedding.EmbeddingModel,
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("llm: marshal embedding request: %w", err)
	}

	if c.cfg.Timeout <= 0 {
		c.cfg.Timeout = 30 * time.Second
	}

	tctx, cancel := context.WithTimeout(ctx, c.cfg.Timeout)
	defer cancel()

	url := c.cfg.Embedding.Endpoint + "/v1/embeddings"
	httpReq, err := http.NewRequestWithContext(tctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("llm: build request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.cfg.Embedding.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.cfg.Embedding.APIKey)
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

	var embeddingResp EmbeddingResponse
	if err := json.Unmarshal(rawBody, &embeddingResp); err != nil {
		return nil, fmt.Errorf("llm: unmarshal response: %w", err)
	}

	if embeddingResp.Error != nil {
		return nil, fmt.Errorf("llm: API error (%s): %s", embeddingResp.Error.Type, embeddingResp.Error.Message)
	}

	return &embeddingResp, nil
}
