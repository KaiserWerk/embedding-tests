package main

import (
	"bufio"
	"os"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

type AppConfig struct {
	OpenAI    OpenAIConfig    `json:"openai"`
	Embedding EmbeddingConfig `json:"embedding"`
	Timeout   time.Duration   `json:"timeout"`
}

type OpenAIConfig struct {
	Endpoint string `json:"endpoint"`
	Model    string `json:"model"`
	APIKey   string `json:"api_key"`
}

type EmbeddingConfig struct {
	Endpoint       string `json:"endpoint"`
	EmbeddingModel string `json:"embedding_model"`
	APIKey         string `json:"api_key"`
}

func LoadConfig(configFile, envFile string) (*AppConfig, error) {
	if envFile != "" {
		if err := loadEnvVars(envFile); err != nil {
			return nil, err
		}
	}

	content, err := os.ReadFile(configFile)
	if err != nil {
		return nil, err
	}

	content = []byte(os.ExpandEnv(string(content)))

	var cfg AppConfig
	if err := yaml.Unmarshal(content, &cfg); err != nil {
		return nil, err
	}

	return &cfg, nil
}

func loadEnvVars(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		value := strings.TrimSpace(parts[1])
		os.Setenv(key, value)
	}
	return scanner.Err()
}
