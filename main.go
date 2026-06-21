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

func load(llmClient *Client) []StoredNorm {
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

	// print the unmarshalled items
	// for _, item := range toc.Items {
	// 	fmt.Printf("Title: %s, Link: %s\n", item.Title, item.Link)
	// }

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

	// unmarshal the XML file into a struct and print the title of the first item
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

	var norms []StoredNorm

	for _, norm := range root.Norm {
		title := norm.Metadaten.Titel
		if title == "" {
			title = norm.Metadaten.Langue
		}
		if title == "" {
			continue
		}
		text := strings.Join(norm.Textdaten.Text.Content.P, "\n")

		embedding, err := llmClient.GetEmbedding(context.Background(), text)
		if err != nil {
			fmt.Println("Error getting embedding for title:", err)
			continue
		}

		norms = append(norms, StoredNorm{
			Title:     title,
			Doknr:     norm.Doknr,
			Text:      text,
			Embedding: embedding.Data[0].Embedding,
		})
		//fmt.Printf("Titel: %s\nDoknr: %s\nText: %v\n\n", title, norm.Doknr, norm.Textdaten.Text.Content.P)
	}

	return norms
}

type StoredNorm struct {
	Title     string
	Doknr     string
	Text      string
	Embedding []float32
	Score     float64
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

func main() {
	llmClient := NewClient(&AppConfig{
		OpenAI: OpenAIConfig{
			Endpoint: "http://localhost:8080",
			APIKey:   "none",
		},
	})
	norms := load(llmClient)
	input := "Welche Mitglieder hat der Stiftungsrat im Rahmen der 1-DM-Goldmünz-Gesetze?"
	results := find(llmClient, input, norms)
	fmt.Println("Result found:", len(results))
	for _, result := range results {
		fmt.Printf("Title: %s\nDoknr: %s\nScore: %f\n\n", result.Title, result.Doknr, result.Score)
	}
	
}

func find(llmClient *Client, input string, norms []StoredNorm) []StoredNorm {
	// 1. Get embedding for input

	inputEmbedding, err := llmClient.GetEmbedding(context.Background(), input)
	if err != nil {
		fmt.Println("Error getting embedding:", err)
		return nil
	}
	// 2. Compare with stored norms and return the most similar ones
	var results []StoredNorm
	for _, norm := range norms {
		norm.Score = CosineSimilarity(inputEmbedding.Data[0].Embedding, norm.Embedding)
		if norm.Score > 0.75 {
			results = append(results, norm)
		} else {
			fmt.Printf("Norm '%s' has low similarity score: %f\n", norm.Title, norm.Score)
		}
	}
	return results
}
