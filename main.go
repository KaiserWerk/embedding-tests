package main

import (
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"sort"
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

func load(llmClient *Client) []StoredChunk {
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
		chunks = EmbedChunks(context.Background(), llmClient, chunks)
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
	ChunkID     string
	ParentDoknr string
	ParentTitle string
	ChunkIndex  int
	Text        string
	Embedding   []float32
	StartPara   int
	EndPara     int
	Score       float64
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

func EmbedChunks(ctx context.Context, llmClient *Client, chunks []StoredChunk) []StoredChunk {
	for i := range chunks {
		embedding, err := llmClient.GetEmbedding(ctx, chunks[i].Text)
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

func FindTopChunks(
	llmClient *Client,
	input string,
	chunks []StoredChunk,
	topK int,
	minScore float64,
) []StoredChunk {
	emb, err := llmClient.GetEmbedding(context.Background(), input)
	if err != nil || len(emb.Data) == 0 {
		return nil
	}
	q := emb.Data[0].Embedding

	scored := make([]StoredChunk, 0, len(chunks))
	for i := range chunks {
		if len(chunks[i].Embedding) == 0 {
			continue
		}
		chunks[i].Score = CosineSimilarity(q, chunks[i].Embedding)
		if chunks[i].Score >= minScore {
			scored = append(scored, chunks[i])
		}
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Score > scored[j].Score
	})

	if topK > 0 && len(scored) > topK {
		return scored[:topK]
	}
	return scored
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

func main() {
	llmClient := NewClient(&AppConfig{
		OpenAI: OpenAIConfig{
			Endpoint: "http://localhost:8080",
			APIKey:   "none",
		},
	})
	chunks := load(llmClient)
	if len(chunks) == 0 {
		fmt.Println("No chunks loaded")
		return
	}
	fmt.Printf("Loaded %d chunks\n\n", len(chunks))

	input := "Wozu dient der Stiftungsrat?"
	results := find(llmClient, input, chunks)
	fmt.Println("Results found:", len(results))
	for _, result := range results {
		fmt.Printf("\tTitle: %s\n\tDoknr: %s\n\tChunk: %d\n\tScore: %f\n\tText: %.100s...\n\n",
			result.ParentTitle, result.ParentDoknr, result.ChunkIndex, result.Score, result.Text)
	}

}

func find(llmClient *Client, input string, chunks []StoredChunk) []StoredChunk {
	// Find top chunks with semantic similarity
	results := FindTopChunks(llmClient, input, chunks, 8, 0.6)

	// Optionally merge to at most 3 results per parent norm for cleaner output
	merged := MergeTopChunksByParent(results, 2)

	return merged
}
