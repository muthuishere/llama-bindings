// Package knowledge provides a persistent vector + full-text knowledge store
// backed by SQLite (modernc.org/sqlite – pure Go, no CGO).
//
// The store supports hybrid retrieval that combines cosine-similarity vector
// search with SQLite FTS5 keyword search, and fuses the two result lists using
// Reciprocal Rank Fusion (RRF).
package knowledge

import (
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"sync"

	_ "modernc.org/sqlite"
)

// Document is a retrieved knowledge item.
type Document struct {
	ID        int64
	Text      string
	Score     float64
	Embedding []float32
}

// Store is a persistent knowledge store backed by SQLite.
// Create with New; always call Close when done.
type Store struct {
	mu   sync.Mutex
	db   *sql.DB
	path string
}

// New opens (or creates) a SQLite knowledge store at dsn.
// Use ":memory:" for an in-memory store (useful in tests).
func New(dsn string) (*Store, error) {
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("knowledge: open db: %w", err)
	}
	db.SetMaxOpenConns(1) // SQLite is single-writer

	s := &Store{db: db, path: dsn}
	if err := s.migrate(); err != nil {
		db.Close()
		return nil, err
	}
	return s, nil
}

// migrate creates the schema if it does not exist.
func (s *Store) migrate() error {
	_, err := s.db.Exec(`
		CREATE TABLE IF NOT EXISTS documents (
			id        INTEGER PRIMARY KEY AUTOINCREMENT,
			text      TEXT    NOT NULL,
			embedding BLOB    NOT NULL
		);
		CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
			USING fts5(text, content=documents, content_rowid=id);
		CREATE TRIGGER IF NOT EXISTS documents_ai
			AFTER INSERT ON documents BEGIN
				INSERT INTO documents_fts(rowid, text) VALUES (new.id, new.text);
			END;
		CREATE TRIGGER IF NOT EXISTS documents_ad
			AFTER DELETE ON documents BEGIN
				INSERT INTO documents_fts(documents_fts, rowid, text)
					VALUES ('delete', old.id, old.text);
			END;
	`)
	return err
}

// Add stores text together with its pre-computed embedding vector.
func (s *Store) Add(text string, embedding []float32) error {
	if text == "" {
		return fmt.Errorf("knowledge: text must not be empty")
	}
	if len(embedding) == 0 {
		return fmt.Errorf("knowledge: embedding must not be empty")
	}
	blob := float32SliceToBytes(embedding)

	s.mu.Lock()
	defer s.mu.Unlock()

	_, err := s.db.Exec(`INSERT INTO documents(text, embedding) VALUES (?, ?)`, text, blob)
	return err
}

// Search performs hybrid RRF retrieval: vector similarity + FTS keyword search.
// Returns up to limit documents ordered by descending RRF score.
func (s *Store) Search(queryVec []float32, queryText string, limit int) ([]Document, error) {
	if len(queryVec) == 0 {
		return nil, fmt.Errorf("knowledge: query vector must not be empty")
	}
	if limit <= 0 {
		limit = 5
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// 1. Vector similarity: full-table scan with cosine similarity.
	type vectorResult struct {
		id    int64
		text  string
		score float64
	}

	rows, err := s.db.Query(`SELECT id, text, embedding FROM documents`)
	if err != nil {
		return nil, fmt.Errorf("knowledge: vector query: %w", err)
	}
	defer rows.Close()

	var vecResults []vectorResult
	for rows.Next() {
		var id int64
		var text string
		var blob []byte
		if err := rows.Scan(&id, &text, &blob); err != nil {
			return nil, err
		}
		stored := bytesToFloat32Slice(blob)
		score := cosineSimilarity(queryVec, stored)
		vecResults = append(vecResults, vectorResult{id, text, score})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	sort.Slice(vecResults, func(i, j int) bool {
		return vecResults[i].score > vecResults[j].score
	})

	// Build rank maps.
	vecRank := make(map[int64]int, len(vecResults))
	for i, r := range vecResults {
		vecRank[r.id] = i + 1
	}

	// 2. FTS keyword search (only when a non-empty query text is provided).
	ftsRank := make(map[int64]int)
	if queryText != "" {
		ftsRows, err := s.db.Query(
			`SELECT rowid FROM documents_fts WHERE documents_fts MATCH ? ORDER BY rank LIMIT ?`,
			queryText, limit*3,
		)
		if err == nil {
			rank := 1
			for ftsRows.Next() {
				var id int64
				if err := ftsRows.Scan(&id); err == nil {
					ftsRank[id] = rank
					rank++
				}
			}
			ftsRows.Close()
		}
		// FTS errors are non-fatal; fall back to vector-only.
	}

	// 3. RRF fusion: score = 1/(k+vecRank) + 1/(k+ftsRank).
	const k = 60
	type rrfEntry struct {
		id   int64
		text string
		rrf  float64
	}

	// Collect all candidate IDs.
	seen := make(map[int64]string, len(vecResults))
	for _, r := range vecResults {
		seen[r.id] = r.text
	}

	entries := make([]rrfEntry, 0, len(seen))
	for id, text := range seen {
		vr, hasVec := vecRank[id]
		fr, hasFts := ftsRank[id]

		rrf := 0.0
		if hasVec {
			rrf += 1.0 / float64(k+vr)
		}
		if hasFts {
			rrf += 1.0 / float64(k+fr)
		}
		entries = append(entries, rrfEntry{id, text, rrf})
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].rrf > entries[j].rrf
	})

	if len(entries) > limit {
		entries = entries[:limit]
	}

	docs := make([]Document, len(entries))
	for i, e := range entries {
		docs[i] = Document{ID: e.id, Text: e.text, Score: e.rrf}
	}
	return docs, nil
}

// All returns all documents in insertion order.
func (s *Store) All() ([]Document, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	rows, err := s.db.Query(`SELECT id, text, embedding FROM documents ORDER BY id`)
	if err != nil {
		return nil, fmt.Errorf("knowledge: all docs query: %w", err)
	}
	defer rows.Close()

	var docs []Document
	for rows.Next() {
		var id int64
		var text string
		var blob []byte
		if err := rows.Scan(&id, &text, &blob); err != nil {
			return nil, err
		}
		docs = append(docs, Document{
			ID:        id,
			Text:      text,
			Embedding: bytesToFloat32Slice(blob),
		})
	}
	return docs, rows.Err()
}

// Path returns the SQLite database path.
func (s *Store) Path() string {
	return s.path
}

// Close closes the underlying SQLite database.
func (s *Store) Close() error {
	return s.db.Close()
}

// ─────────────────────────────────────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────────────────────────────────────

func cosineSimilarity(a, b []float32) float64 {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	var dot, normA, normB float64
	for i := 0; i < n; i++ {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// ─────────────────────────────────────────────────────────────────────────────
// Binary serialisation helpers
// ─────────────────────────────────────────────────────────────────────────────

func float32SliceToBytes(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	return b
}

func bytesToFloat32Slice(b []byte) []float32 {
	n := len(b) / 4
	v := make([]float32, n)
	for i := range v {
		bits := binary.LittleEndian.Uint32(b[i*4:])
		v[i] = math.Float32frombits(bits)
	}
	return v
}
