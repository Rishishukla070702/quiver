// Package wal provides a write-ahead log: an append-only record of every Add,
// so a Quiver index can be rebuilt exactly after a restart or a crash.
package wal

import (
	"bufio"
	"encoding/json"
	"os"

	"github.com/Rishishukla070702/quiver/internal/vector"
)

// Record is one logged mutation: a vector added under an id.
type Record struct {
	ID     string        `json:"id"`
	Vector vector.Vector `json:"vector"`
}

// WAL is an append-only write-ahead log, stored as one JSON object per line.
type WAL struct {
	f *os.File
}

// Open opens the log at path for appending, creating it if it doesn't exist.
// O_APPEND sends every write to the end of the file; O_CREATE makes the file if
// it's missing; 0o644 is the file permission mode (owner read/write, others read).
func Open(path string) (*WAL, error) {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	return &WAL{f: f}, nil
}

// Close closes the underlying file.
func (w *WAL) Close() error { return w.f.Close() }

// Append writes rec to the log as a JSON line and flushes it to disk with Sync
// (fsync), so a crash immediately after loses nothing — that durability is the
// defining guarantee of a write-ahead log.
func (w *WAL) Append(rec Record) error {
	err := json.NewEncoder(w.f).Encode(rec)
	if err != nil {
		return err
	}
	err = w.f.Sync()
	if err != nil {
		return err
	}
	return nil
}

// Replay reads every record in the log at path from start to end, calling fn for
// each in order. A missing file is NOT an error (nothing has been logged yet).
func Replay(path string, fn func(Record) error) error {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		var rec Record
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			return err
		}
		if err := fn(rec); err != nil {
			return err
		}
	}
	return scanner.Err()
}
