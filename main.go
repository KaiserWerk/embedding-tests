package main

import (
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strings"
)

type Item struct {
	Title string `xml:"title"`
	Link  string `xml:"link"`
}

type TOC struct {
	Items []Item `xml:"item"`
}

type Root struct {
	Builddate int    `xml:"builddate,attr"` // XML attribute
	Doknr     string `xml:"doknr,attr"`     // XML attribute
	Norm      []Norm `xml:"norm"`           // XML element (array)
}
type Norm struct {
	Builddate int       `xml:"builddate,attr"` // XML attribute
	Doknr     string    `xml:"doknr,attr"`     // XML attribute
	Metadaten Metadaten `xml:"metadaten"`      // XML element
	Textdaten Textdaten `xml:"textdaten"`      // XML element
}
type Textdaten struct {
	Text      Text      `xml:"text"`      // XML element
	Fussnoten Fussnoten `xml:"fussnoten"` // XML element
}
type Fussnoten struct {
	Content Content `xml:"Content"` // XML element
}
type Text struct {
	Format  string  `xml:"format,attr"` // XML attribute
	Content Content `xml:"Content"`     // XML element
}
type Content struct {
	P []string `xml:"P"` // XML element
}
type Metadaten struct {
	Jurabk            string     `xml:"jurabk"`             // XML element
	AusfertigungDatum string     `xml:"ausfertigung-datum"` // XML element
	Fundstelle        Fundstelle `xml:"fundstelle"`         // XML element
	Langue            string     `xml:"langue"`             // XML element
	Enbez             string     `xml:"enbez"`              // XML element
	Titel             string     `xml:"titel"`              // XML element
}
type Fundstelle struct {
	Typ        string `xml:"typ,attr"`   // XML attribute
	Periodikum string `xml:"periodikum"` // XML element
	Zitstelle  string `xml:"zitstelle"`  // XML element
}

func load(embeddingClient *EmbeddingClient) []StoredChunk {
	xmlData, err := os.ReadFile("gii-toc.xml")
	if err != nil {
		fmt.Println("Error reading XML file:", err)
		return nil
	}

	// unmarshal the XML data into a slice of Item structs
	var toc TOC
	err = xml.Unmarshal(xmlData, &toc)
	if err != nil {
		fmt.Println("Error unmarshalling XML:", err)
		return nil
	}

	// work with first item
	item := toc.Items[0]

	// download the zip file from the link
	zipFile, err := os.Create("file.zip")
	if err != nil {
		fmt.Println("Error creating zip file:", err)
		return nil
	}
	defer zipFile.Close()

	// use http.Get to download the file
	resp, err := http.Get(item.Link)
	if err != nil {
		fmt.Println("Error downloading file:", err)
		return nil
	}
	defer resp.Body.Close()

	_, err = io.Copy(zipFile, resp.Body)
	if err != nil {
		fmt.Println("Error saving zip file:", err)
		return nil
	}
	// unzip the file and read the XML file inside
	Unzip("file.zip", "unzipped")

	// get first file in unzipped folder
	files, err := os.ReadDir("unzipped")
	if err != nil {
		fmt.Println("Error reading unzipped folder:", err)
		return nil
	}

	if len(files) == 0 {
		fmt.Println("No files found in unzipped folder")
		return nil
	}

	// get the first file in the unzipped folder
	firstFile := files[0]
	fmt.Println("First file in unzipped folder:", firstFile.Name())

	// unmarshal the XML file into a struct
	xmlFile, err := os.Open("unzipped/" + firstFile.Name())
	if err != nil {
		fmt.Println("Error opening XML file:", err)
		return nil
	}

	defer xmlFile.Close()

	xmlData, err = io.ReadAll(xmlFile)
	if err != nil {
		fmt.Println("Error reading XML file:", err)
		return nil
	}

	var root Root
	err = xml.Unmarshal(xmlData, &root)
	if err != nil {
		fmt.Println("Error unmarshalling XML:", err)
		return nil
	}

	var allChunks []StoredChunk
	chunkCfg := ChunkConfig{
		TargetTokens:  250,
		MaxTokens:     320,
		OverlapTokens: 40,
		MinTokens:     80,
	}

	for _, norm := range root.Norm {
		title := norm.Metadaten.Titel
		if title == "" {
			title = norm.Metadaten.Langue
		}
		if title == "" {
			continue
		}

		// build chunks from paragraphs
		chunks := BuildChunksFromParagraphs(norm.Doknr, title, norm.Textdaten.Text.Content.P, chunkCfg)
		if len(chunks) == 0 {
			continue
		}

		// embed all chunks
		chunks = EmbedChunks(context.Background(), embeddingClient, chunks)
		allChunks = append(allChunks, chunks...)
	}

	return allChunks
}

type StoredNorm struct {
	Title     string
	Doknr     string
	Text      string
	Embedding []float32
	Score     float64
}

type StoredChunk struct {
	ChunkID       string
	ParentDoknr   string
	ParentTitle   string
	ChunkIndex    int
	Text          string
	Embedding     []float32
	StartPara     int
	EndPara       int
	Score         float64
	SemanticScore float64
	KeywordScore  float64
}

type ChunkConfig struct {
	TargetTokens  int
	MaxTokens     int
	OverlapTokens int
	MinTokens     int
}

func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		panic("cosine similarity: vector dimensions differ")
	}

	var dot, normA, normB float64

	for i := range a {
		af := float64(a[i])
		bf := float64(b[i])

		dot += af * bf
		normA += af * af
		normB += bf * bf
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// ApproxTokenCount returns an approximate token count for a given string containing german text.
func ApproxTokenCount(s string) int {
	words := len(strings.Fields(s))
	return int(float64(words) * 1.3)
}

// SplitIntoSentences splits text into sentences based on punctuation marks.
func SplitIntoSentences(paragraph string) []string {
	var sentences []string
	current := ""
	for _, char := range paragraph {
		current += string(char)
		if char == '.' || char == '!' || char == '?' {
			s := strings.TrimSpace(current)
			if s != "" {
				sentences = append(sentences, s)
			}
			current = ""
		}
	}
	if s := strings.TrimSpace(current); s != "" {
		sentences = append(sentences, s)
	}
	return sentences
}

type ChunkPiece struct {
	Text      string
	ParaIndex int
	Tokens    int
}

func BuildChunksFromParagraphs(
	parentDoknr string,
	parentTitle string,
	paragraphs []string,
	cfg ChunkConfig,
) []StoredChunk {
	var pieces []ChunkPiece
	for i, p := range paragraphs {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		t := ApproxTokenCount(p)
		if t <= cfg.MaxTokens {
			pieces = append(pieces, ChunkPiece{Text: p, ParaIndex: i, Tokens: t})
			continue
		}
		for _, s := range SplitIntoSentences(p) {
			s = strings.TrimSpace(s)
			if s == "" {
				continue
			}
			pieces = append(pieces, ChunkPiece{
				Text:      s,
				ParaIndex: i,
				Tokens:    ApproxTokenCount(s),
			})
		}
	}

	var out []StoredChunk
	var cur []ChunkPiece
	curTokens := 0
	chunkIndex := 0

	flush := func() {
		if len(cur) == 0 {
			return
		}
		textParts := make([]string, 0, len(cur))
		for _, cp := range cur {
			textParts = append(textParts, cp.Text)
		}
		out = append(out, StoredChunk{
			ChunkID:     fmt.Sprintf("%s:%d", parentDoknr, chunkIndex),
			ParentDoknr: parentDoknr,
			ParentTitle: parentTitle,
			ChunkIndex:  chunkIndex,
			Text:        strings.Join(textParts, " "),
			StartPara:   cur[0].ParaIndex,
			EndPara:     cur[len(cur)-1].ParaIndex,
		})
		chunkIndex++
	}

	i := 0
	for i < len(pieces) {
		next := pieces[i]
		if curTokens+next.Tokens <= cfg.MaxTokens || len(cur) == 0 {
			cur = append(cur, next)
			curTokens += next.Tokens
			i++
			continue
		}

		if curTokens < cfg.MinTokens {
			cur = append(cur, next)
			curTokens += next.Tokens
			i++
		} else {
			flush()

			overlap := []ChunkPiece{}
			overlapTokens := 0
			for j := len(cur) - 1; j >= 0; j-- {
				if overlapTokens+cur[j].Tokens > cfg.OverlapTokens && len(overlap) > 0 {
					break
				}
				overlap = append([]ChunkPiece{cur[j]}, overlap...)
				overlapTokens += cur[j].Tokens
			}
			cur = overlap
			curTokens = overlapTokens
		}
	}

	flush()
	return out
}

func EmbedChunks(ctx context.Context, embeddingClient *EmbeddingClient, chunks []StoredChunk) []StoredChunk {
	for i := range chunks {
		embedding, err := embeddingClient.GetEmbedding(ctx, chunks[i].Text)
		if err != nil {
			fmt.Printf("Error getting embedding for chunk %s: %v\n", chunks[i].ChunkID, err)
			continue
		}
		if len(embedding.Data) > 0 {
			chunks[i].Embedding = embedding.Data[0].Embedding
		}
	}
	return chunks
}

func MergeTopChunksByParent(results []StoredChunk, maxPerParent int) []StoredChunk {
	count := make(map[string]int)
	var merged []StoredChunk
	for i := range results {
		if count[results[i].ParentDoknr] < maxPerParent {
			merged = append(merged, results[i])
			count[results[i].ParentDoknr]++
		}
	}
	return merged
}

func embeddingDimension(chunks []StoredChunk) int {
	for i := range chunks {
		if len(chunks[i].Embedding) > 0 {
			return len(chunks[i].Embedding)
		}
	}
	return 0
}

func main() {
	ctx := context.Background()
	ingestMode := os.Args[1] == "ingest"
	queryMode := os.Args[1] == "query"
	input := strings.Join(os.Args[2:], " ")

	config, err := LoadConfig("config.yaml", "")
	if err != nil {
		fmt.Println("Error loading config:", err)
		return
	}

	llmClient := NewLLMClient(config)
	embeddingClient := NewEmbeddingClient(config)

	store, err := NewPostgresStore(ctx, "localhost", 5432, "postgres", "password", "postgres")
	if err != nil {
		fmt.Println("Error connecting to PostgreSQL:", err)
		return
	}
	defer store.Close()

	var chunks []StoredChunk
	if ingestMode {
		chunks = load(embeddingClient)
		if len(chunks) == 0 {
			fmt.Println("No chunks loaded from source for ingest")
			return
		}
		fmt.Printf("Loaded %d chunks from source\n\n", len(chunks))

		dims := embeddingDimension(chunks)
		if dims == 0 {
			fmt.Println("No embeddings generated for ingest")
			return
		}

		if err := store.EnsureSchema(ctx, dims); err != nil {
			fmt.Println("Error ensuring schema:", err)
			return
		}

		if err := store.UpsertChunks(ctx, chunks); err != nil {
			fmt.Println("Error upserting chunks:", err)
			return
		}
		fmt.Println("Chunks persisted to PostgreSQL")
		return
	}

	if queryMode {

		if err := store.EnsureSchema(ctx, 1536); err != nil {
			fmt.Println("Error ensuring schema:", err)
			return
		}
		chunks, err = store.LoadChunks(ctx)
		if err != nil {
			fmt.Println("Error loading chunks from PostgreSQL:", err)
			return
		}
		if len(chunks) == 0 {
			fmt.Println("No chunks loaded from PostgreSQL. Run with 'ingest' first.")
			return
		}
		fmt.Printf("Loaded %d chunks from PostgreSQL\n\n", len(chunks))

		results := find(ctx, embeddingClient, store, input)
		fmt.Println("Results found:", len(results))
		for _, result := range results {
			fmt.Printf("\tTitle: %s\n\tDoknr: %s\n\tChunk: %d\n\tHybrid: %f\n\tVector: %f\n\tKeyword: %f\n\tText: %.100s...\n\n",
				result.ParentTitle, result.ParentDoknr, result.ChunkIndex, result.Score, result.SemanticScore, result.KeywordScore, result.Text)
		}

		answer, err := answerFromHybridResults(ctx, llmClient, input, results)
		if err != nil {
			fmt.Println("Error generating final answer:", err)
			return
		}

		fmt.Println("Final answer:")
		fmt.Println(answer)
	}
}

func find(ctx context.Context, embeddingClient *EmbeddingClient, store *PostgresStore, input string) []StoredChunk {
	inputEmbedding, err := embeddingClient.GetEmbedding(ctx, input)
	if err != nil || len(inputEmbedding.Data) == 0 {
		fmt.Println("Error getting input embedding:", err)
		return nil
	}

	results, err := store.SearchHybridChunks(ctx, inputEmbedding.Data[0].Embedding, input, 8)
	if err != nil {
		fmt.Println("Error querying PostgreSQL chunks:", err)
		return nil
	}

	// Optionally merge to at most 2 results per parent norm for cleaner output.
	merged := MergeTopChunksByParent(results, 2)

	return merged
}

func answerFromHybridResults(ctx context.Context, llmClient *LLMClient, question string, chunks []StoredChunk) (string, error) {
	if strings.TrimSpace(question) == "" {
		return "", fmt.Errorf("question is empty")
	}
	if len(chunks) == 0 {
		return "", fmt.Errorf("no retrieved chunks available")
	}

	const maxChunks = 6
	const maxCharsPerChunk = 900

	var b strings.Builder
	limit := len(chunks)
	if limit > maxChunks {
		limit = maxChunks
	}

	for i := 0; i < limit; i++ {
		text := strings.TrimSpace(chunks[i].Text)
		if len(text) > maxCharsPerChunk {
			text = text[:maxCharsPerChunk] + "..."
		}

		fmt.Fprintf(&b,
			"Source %d\nDoknr: %s\nTitle: %s\nChunk: %d\nHybrid: %.6f\nVector: %.6f\nKeyword: %.6f\nText: %s\n\n",
			i+1,
			chunks[i].ParentDoknr,
			chunks[i].ParentTitle,
			chunks[i].ChunkIndex,
			chunks[i].Score,
			chunks[i].SemanticScore,
			chunks[i].KeywordScore,
			text,
		)
	}

	req := ChatRequest{
		Messages: []Message{
			{
				Role:    "system",
				Content: "You are a legal QA assistant. Answer only with information grounded in the provided retrieval context. If the context is insufficient, say clearly what is missing. Keep the answer concise and factual.",
			},
			{
				Role:    "user",
				Content: "Question:\n" + question + "\n\nRetrieved context:\n" + b.String() + "Please answer in German and mention the most relevant Doknr references.",
			},
		},
		Model: llmClient.cfg.OpenAI.Model,
	}

	resp, err := llmClient.Chat(ctx, req)
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("chat response returned no choices")
	}

	answer := strings.TrimSpace(resp.Choices[0].Message.Content)
	if answer == "" {
		return "", fmt.Errorf("chat response was empty")
	}

	return answer, nil
}
