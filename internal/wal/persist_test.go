package wal

import (
	"path/filepath"
	"testing"

	"github.com/Rishishukla070702/quiver/internal/index"
	"github.com/Rishishukla070702/quiver/internal/vector"
)

// TestPersistenceSurvivesRestart is the payoff: data added in one "process"
// must reappear in a fresh index after reopening the same log.
func TestPersistenceSurvivesRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "index.wal")

	// Process 1: open a fresh index, add three vectors, close cleanly.
	p1, err := OpenIndex(path, index.NewFlat(3, index.Cosine))
	if err != nil {
		t.Fatalf("OpenIndex #1: %v", err)
	}
	writes := []Record{
		{ID: "a", Vector: vector.Vector{1, 0, 0}},
		{ID: "b", Vector: vector.Vector{0, 1, 0}},
		{ID: "c", Vector: vector.Vector{1, 1, 0}},
	}
	for _, r := range writes {
		if err := p1.Add(r.ID, r.Vector); err != nil {
			t.Fatalf("Add %s: %v", r.ID, err)
		}
	}
	if err := p1.Close(); err != nil {
		t.Fatalf("Close #1: %v", err)
	}

	// Process 2: reopen the SAME log into a brand-new, empty index.
	p2, err := OpenIndex(path, index.NewFlat(3, index.Cosine))
	if err != nil {
		t.Fatalf("OpenIndex #2: %v", err)
	}
	if p2.Len() != 3 {
		t.Fatalf("after restart Len() = %d, want 3 (the writes should have survived)", p2.Len())
	}
	got, err := p2.Search(vector.Vector{1, 0, 0}, 1)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(got) != 1 || got[0].ID != "a" {
		t.Fatalf("nearest to [1,0,0] = %+v, want id \"a\"", got)
	}
}
