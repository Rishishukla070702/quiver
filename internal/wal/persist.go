package wal

import (
	"github.com/Rishishukla070702/quiver/internal/index"
	"github.com/Rishishukla070702/quiver/internal/vector"
)

// PersistentIndex wraps an index.Index with a write-ahead log so its contents
// survive a restart. It satisfies index.Index itself, so callers (like the HTTP
// server) can't tell it apart from a plain in-memory index.
type PersistentIndex struct {
	idx index.Index
	wal *WAL
}

// Compile-time proof that PersistentIndex is a drop-in index.Index.
var _ index.Index = (*PersistentIndex)(nil)

// OpenIndex replays any existing log at path into idx (rebuilding its state from
// the record of past writes), then returns a PersistentIndex that appends future
// Adds to that same log.
func OpenIndex(path string, idx index.Index) (*PersistentIndex, error) {
	replayErr := Replay(path, func(rec Record) error {
		return idx.Add(rec.ID, rec.Vector)
	})
	if replayErr != nil {
		return nil, replayErr
	}

	wal, err := Open(path)
	if err != nil {
		return nil, err
	}
	return &PersistentIndex{idx: idx, wal: wal}, nil
}

// Add applies the write to the wrapped index first (so invalid input is rejected
// before it can enter the log), then records it in the write-ahead log — only
// accepted writes are ever logged. Strict write-ahead would log first for maximum
// durability, but risks logging records the index would reject.
func (p *PersistentIndex) Add(id string, vec vector.Vector) error {
	p.idx.Add(id, vec)
	err := p.wal.Append(Record{ID: id, Vector: vec})
	if err != nil {
		return err
	}
	return nil
}

// Search delegates to the wrapped index.
func (p *PersistentIndex) Search(query vector.Vector, k int) ([]index.Result, error) {
	return p.idx.Search(query, k)
}

// Len delegates to the wrapped index.
func (p *PersistentIndex) Len() int { return p.idx.Len() }

// Close closes the underlying log file.
func (p *PersistentIndex) Close() error { return p.wal.Close() }
