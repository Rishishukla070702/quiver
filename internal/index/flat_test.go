package index

import (
	"errors"
	"testing"

	"github.com/Rishishukla070702/quiver/internal/vector"
)

func TestFlatAdd(t *testing.T) {
	idx := NewFlat(3, Cosine) // this index stores 3-dimensional vectors

	// Happy path: correctly-sized vectors are accepted and the index grows.
	if err := idx.Add("a", vector.Vector{1, 0, 0}); err != nil {
		t.Fatalf("Add(a) unexpected error: %v", err)
	}
	if err := idx.Add("b", vector.Vector{0, 1, 0}); err != nil {
		t.Fatalf("Add(b) unexpected error: %v", err)
	}
	if got := idx.Len(); got != 2 {
		t.Errorf("Len() = %d, want 2", got)
	}

	// Wrong dimension must fail loud (2 dims into a 3-dim index).
	if err := idx.Add("bad", vector.Vector{1, 1}); !errors.Is(err, vector.ErrDimMismatch) {
		t.Errorf("Add(wrong dim) error = %v, want ErrDimMismatch", err)
	}

	// ...and a rejected Add must NOT have been stored (validate BEFORE you mutate).
	if got := idx.Len(); got != 2 {
		t.Errorf("after rejected Add, Len() = %d, want 2 (rejected vector must not be stored)", got)
	}
}

func TestFlatSearch(t *testing.T) {
	idx := NewFlat(2, Cosine) // this index stores 2-dimensional vectors
	mustAdd(t, idx, "east", vector.Vector{1, 0})
	mustAdd(t, idx, "north", vector.Vector{0, 1})
	mustAdd(t, idx, "northeast", vector.Vector{1, 1})
	mustAdd(t, idx, "west", vector.Vector{-1, 0})

	// A query pointing "east". By cosine similarity the ranking must be:
	//   east (1.0) > northeast (~0.707) > north (0) > west (-1)
	got, err := idx.Search(vector.Vector{1, 0}, 2)
	if err != nil {
		t.Fatalf("Search unexpected error: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("Search returned %d results, want 2", len(got))
	}
	if got[0].ID != "east" || got[1].ID != "northeast" {
		t.Errorf("ranking = [%s, %s], want [east, northeast]", got[0].ID, got[1].ID)
	}

	// k larger than the index returns everything (4), clamped — not an error.
	all, err := idx.Search(vector.Vector{1, 0}, 100)
	if err != nil {
		t.Fatalf("Search(k=100) unexpected error: %v", err)
	}
	if len(all) != 4 {
		t.Errorf("Search(k=100) returned %d, want 4 (clamped to index size)", len(all))
	}

	// A negative (or zero) k must NOT panic — it just returns nothing.
	none, err := idx.Search(vector.Vector{1, 0}, -1)
	if err != nil {
		t.Fatalf("Search(k=-1) unexpected error: %v", err)
	}
	if len(none) != 0 {
		t.Errorf("Search(k=-1) returned %d results, want 0", len(none))
	}

	// A query of the wrong dimension fails loud.
	if _, err := idx.Search(vector.Vector{1, 2, 3}, 1); !errors.Is(err, vector.ErrDimMismatch) {
		t.Errorf("Search(wrong dim) error = %v, want ErrDimMismatch", err)
	}
}

// mustAdd is a test helper: it fails the test if Add errors, so the actual test
// bodies stay focused. t.Helper() makes failures point at the caller's line.
func mustAdd(t *testing.T, idx *FlatIndex, id string, v vector.Vector) {
	t.Helper()
	if err := idx.Add(id, v); err != nil {
		t.Fatalf("setup Add(%s) failed: %v", id, err)
	}
}

// A wrong-dimension query must fail loud EVEN when the index is empty. If the
// only dimension check lives inside the per-entry loop, an empty index skips
// the loop entirely and silently returns (nil, nil) — accepting a malformed
// query. The guard belongs up front, on the query itself.
func TestFlatSearchValidatesDimOnEmptyIndex(t *testing.T) {
	idx := NewFlat(2, Cosine) // empty — nothing added

	if _, err := idx.Search(vector.Vector{1, 2, 3}, 5); !errors.Is(err, vector.ErrDimMismatch) {
		t.Errorf("Search(wrong-dim query, empty index) error = %v, want ErrDimMismatch", err)
	}
}

func TestFlatSearchWithNewPluggability(t *testing.T) {
	idx := NewFlat(2, L2)
	mustAdd(t, idx, "near", vector.Vector{1, 0})
	mustAdd(t, idx, "mid", vector.Vector{0, 3})
	mustAdd(t, idx, "far", vector.Vector{3, 4})

	got, err := idx.Search(vector.Vector{1, 0}, 3)
	if err != nil {
		t.Fatalf("Search unexpected error: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("Search returned %d results, want 3", len(got))
	}
	want := []string{"near", "mid", "far"}
	for i, id := range want {
		if got[i].ID != id {
			t.Errorf("rank %d = %q, want %q", i, got[i].ID, id)
		}
	}
}
