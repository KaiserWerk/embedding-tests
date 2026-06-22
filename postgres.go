package main

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
	_ "github.com/pgvector/pgvector-go/pgx"
)

type PostgresStore struct {
	pool *pgxpool.Pool
}

func NewPostgresStore(ctx context.Context, host string, port int, user, password, dbname string) (*PostgresStore, error) {
	dsn := fmt.Sprintf("postgres://%s:%s@%s:%d/%s", user, password, host, port, dbname)
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("postgres: create pool: %w", err)
	}

	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("postgres: ping: %w", err)
	}

	return &PostgresStore{pool: pool}, nil
}

func (s *PostgresStore) Close() {
	if s.pool != nil {
		s.pool.Close()
	}
}

func (s *PostgresStore) EnsureSchema(ctx context.Context, embeddingDims int) error {
	if embeddingDims <= 0 {
		return fmt.Errorf("postgres: invalid embedding dimension: %d", embeddingDims)
	}

	if _, err := s.pool.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector"); err != nil {
		return fmt.Errorf("postgres: create extension vector: %w", err)
	}

	createTableSQL := fmt.Sprintf(`
CREATE TABLE IF NOT EXISTS norm_chunks (
	chunk_id      TEXT PRIMARY KEY,
	parent_doknr  TEXT NOT NULL,
	parent_title  TEXT NOT NULL,
	chunk_index   INT NOT NULL,
	text          TEXT NOT NULL,
	start_para    INT NOT NULL,
	end_para      INT NOT NULL,
	embedding     vector(%d) NOT NULL
)`, embeddingDims)
	if _, err := s.pool.Exec(ctx, createTableSQL); err != nil {
		return fmt.Errorf("postgres: create table norm_chunks: %w", err)
	}

	if _, err := s.pool.Exec(ctx, `
CREATE INDEX IF NOT EXISTS idx_norm_chunks_embedding_cosine
ON norm_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100)`); err != nil {
		return fmt.Errorf("postgres: create vector index: %w", err)
	}

	if _, err := s.pool.Exec(ctx, `
CREATE INDEX IF NOT EXISTS idx_norm_chunks_parent_doknr
ON norm_chunks(parent_doknr)`); err != nil {
		return fmt.Errorf("postgres: create parent_doknr index: %w", err)
	}

	return nil
}

func (s *PostgresStore) UpsertChunks(ctx context.Context, chunks []StoredChunk) error {
	if len(chunks) == 0 {
		return nil
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("postgres: begin tx: %w", err)
	}
	defer tx.Rollback(ctx)

	batch := &pgx.Batch{}
	for i := range chunks {
		if len(chunks[i].Embedding) == 0 {
			continue
		}

		batch.Queue(`
INSERT INTO norm_chunks (
	chunk_id, parent_doknr, parent_title, chunk_index, text, start_para, end_para, embedding
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT (chunk_id) DO UPDATE SET
	parent_doknr = EXCLUDED.parent_doknr,
	parent_title = EXCLUDED.parent_title,
	chunk_index  = EXCLUDED.chunk_index,
	text         = EXCLUDED.text,
	start_para   = EXCLUDED.start_para,
	end_para     = EXCLUDED.end_para,
	embedding    = EXCLUDED.embedding
`,
			chunks[i].ChunkID,
			chunks[i].ParentDoknr,
			chunks[i].ParentTitle,
			chunks[i].ChunkIndex,
			chunks[i].Text,
			chunks[i].StartPara,
			chunks[i].EndPara,
			pgvector.NewVector(chunks[i].Embedding),
		)
	}

	br := tx.SendBatch(ctx, batch)
	if err := br.Close(); err != nil {
		return fmt.Errorf("postgres: execute upsert batch: %w", err)
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("postgres: commit tx: %w", err)
	}

	return nil
}

func (s *PostgresStore) LoadChunks(ctx context.Context) ([]StoredChunk, error) {
	rows, err := s.pool.Query(ctx, `
SELECT
	chunk_id,
	parent_doknr,
	parent_title,
	chunk_index,
	text,
	start_para,
	end_para,
	embedding
FROM norm_chunks
ORDER BY parent_doknr, chunk_index`)
	if err != nil {
		return nil, fmt.Errorf("postgres: load chunks: %w", err)
	}
	defer rows.Close()

	var chunks []StoredChunk
	for rows.Next() {
		var chunk StoredChunk
		var embedding pgvector.Vector
		if err := rows.Scan(
			&chunk.ChunkID,
			&chunk.ParentDoknr,
			&chunk.ParentTitle,
			&chunk.ChunkIndex,
			&chunk.Text,
			&chunk.StartPara,
			&chunk.EndPara,
			&embedding,
		); err != nil {
			return nil, fmt.Errorf("postgres: scan stored chunk: %w", err)
		}
		chunk.Embedding = embedding.Slice()
		chunks = append(chunks, chunk)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("postgres: iterate stored chunks: %w", err)
	}

	return chunks, nil
}

func (s *PostgresStore) SearchTopChunks(ctx context.Context, queryEmbedding []float32, topK int, minScore float64) ([]StoredChunk, error) {
	if len(queryEmbedding) == 0 {
		return nil, nil
	}
	if topK <= 0 {
		topK = 8
	}

	rows, err := s.pool.Query(ctx, `
SELECT
	chunk_id,
	parent_doknr,
	parent_title,
	chunk_index,
	text,
	start_para,
	end_para,
	1 - (embedding <=> $1::vector) AS score
FROM norm_chunks
WHERE 1 - (embedding <=> $1::vector) >= $2
ORDER BY embedding <=> $1::vector
LIMIT $3
`, pgvector.NewVector(queryEmbedding), minScore, topK)
	if err != nil {
		return nil, fmt.Errorf("postgres: query top chunks: %w", err)
	}
	defer rows.Close()

	var results []StoredChunk
	for rows.Next() {
		var c StoredChunk
		if err := rows.Scan(
			&c.ChunkID,
			&c.ParentDoknr,
			&c.ParentTitle,
			&c.ChunkIndex,
			&c.Text,
			&c.StartPara,
			&c.EndPara,
			&c.Score,
		); err != nil {
			return nil, fmt.Errorf("postgres: scan result row: %w", err)
		}
		results = append(results, c)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("postgres: row iteration error: %w", err)
	}

	return results, nil
}
