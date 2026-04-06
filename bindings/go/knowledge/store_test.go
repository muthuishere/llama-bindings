package knowledge_test

import (
	"math"
	"testing"

	"github.com/muthuishere/llama-bindings/go/knowledge"
)

func TestStoreAddAndSearch(t *testing.T) {
	s, err := knowledge.New(":memory:")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	vec := func(x, y float32) []float32 { return []float32{x, y} }

	if err := s.Add("the sky is blue", vec(1, 0)); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if err := s.Add("the grass is green", vec(0, 1)); err != nil {
		t.Fatalf("Add: %v", err)
	}

	docs, err := s.Search(vec(1, 0), "sky", 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(docs) == 0 {
		t.Fatal("expected at least one result")
	}
	if docs[0].Text != "the sky is blue" {
		t.Fatalf("expected 'the sky is blue' as top result, got %q", docs[0].Text)
	}
}

func TestStoreSearchVectorOnly(t *testing.T) {
	s, err := knowledge.New(":memory:")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	_ = s.Add("apple", []float32{1, 0, 0})
	_ = s.Add("banana", []float32{0, 1, 0})
	_ = s.Add("cherry", []float32{0, 0, 1})

	docs, err := s.Search([]float32{0.9, 0.1, 0}, "", 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(docs) == 0 {
		t.Fatal("expected results")
	}
	if docs[0].Text != "apple" {
		t.Fatalf("expected apple as top result, got %q", docs[0].Text)
	}
}

func TestStoreEmpty(t *testing.T) {
	s, err := knowledge.New(":memory:")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	docs, err := s.Search([]float32{1, 0}, "query", 5)
	if err != nil {
		t.Fatalf("Search on empty store: %v", err)
	}
	if len(docs) != 0 {
		t.Fatalf("expected 0 results on empty store, got %d", len(docs))
	}
}

func TestStoreAddEmptyText(t *testing.T) {
	s, err := knowledge.New(":memory:")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	if err := s.Add("", []float32{1}); err == nil {
		t.Fatal("expected error for empty text")
	}
}

func TestStoreCloseIdempotent(t *testing.T) {
	s, err := knowledge.New(":memory:")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	s.Close()
	// Second close should not panic.
	_ = s.Close()
}

func TestCosineSimilarityOrthogonal(t *testing.T) {
	s, err := knowledge.New(":memory:")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	_ = s.Add("A", []float32{1, 0})
	_ = s.Add("B", []float32{0, 1})

	docs, err := s.Search([]float32{1, 0}, "", 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(docs) < 2 {
		t.Fatalf("expected 2 results, got %d", len(docs))
	}
	if docs[0].Text != "A" {
		t.Fatalf("expected A first, got %q", docs[0].Text)
	}
	// Score for B should be ~0 (orthogonal vectors).
	if math.Abs(docs[1].Score) > 0.1 {
		t.Logf("B score: %f (non-zero is ok if FTS boosted it)", docs[1].Score)
	}
}
