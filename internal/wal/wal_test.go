package wal

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/Rishishukla070702/quiver/internal/vector"
)

func TestWALRoundTrip(t *testing.T) {
	// t.TempDir() gives a temp directory that's auto-removed when the test ends.
	path := filepath.Join(t.TempDir(), "test.wal")

	recs := []Record{
		{ID: "a", Vector: vector.Vector{1, 0, 0}},
		{ID: "b", Vector: vector.Vector{0, 1, 0}},
		{ID: "c", Vector: vector.Vector{0, 0, 1}},
	}

	// Write phase: append three records, then close.
	w, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	for _, r := range recs {
		if err := w.Append(r); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Replay phase: simulate a restart by reading the records back.
	var got []Record
	err = Replay(path, func(r Record) error {
		got = append(got, r)
		return nil
	})
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	if !reflect.DeepEqual(got, recs) {
		t.Errorf("replayed %+v, want %+v", got, recs)
	}
}

func TestReplayMissingFileIsOK(t *testing.T) {
	path := filepath.Join(t.TempDir(), "does-not-exist.wal")
	calls := 0
	err := Replay(path, func(Record) error { calls++; return nil })
	if err != nil {
		t.Errorf("Replay of missing file = %v, want nil", err)
	}
	if calls != 0 {
		t.Errorf("fn called %d times on a missing file, want 0", calls)
	}
}
