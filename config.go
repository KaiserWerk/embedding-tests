package main

import "time"

type AppConfig struct {
	OpenAI  OpenAIConfig  `json:"openai"`
	Timeout time.Duration `json:"timeout"`
}

type OpenAIConfig struct {
	Endpoint       string `json:"endpoint"`
	Model          string `json:"model"`
	EmbeddingModel string `json:"embedding_model"`
	APIKey         string `json:"api_key"`
}
