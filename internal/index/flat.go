// Package index holds Quiver's vector indexes: the structures that store
// vectors and answer "find the k most similar" queries.
//
// FlatIndex is the exact, brute-force baseline. A future HNSW index will live in
// this package as a faster, approximate alternative benchmarked against it.
package index

import (
	"sort"

	"github.com/Rishishukla070702/quiver/internal/vector"
)

// Result is one search hit: the ID of a stored vector and its similarity score.
// Search returns results best-first.
type Result struct {
	ID    string
	Score float32
}

// entry is one stored vector together with its ID.
type entry struct {
	id  string
	vec vector.Vector
}

// FlatIndex stores every vector in a slice and answers a query by comparing it
// against all of them. It is exact but runs in O(n) per query.
type FlatIndex struct {
	dim     int     // required dimensionality; every added vector must match this
	entries []entry // all stored vectors, in insertion order
	metric  Metric
}

type Metric struct {
	fn            func(a, b vector.Vector) (float32, error)
	sortDirection string
}

var (
	Cosine = Metric{fn: vector.Cosine, sortDirection: "desc"}
	L2     = Metric{fn: vector.L2, sortDirection: "asc"}
)

// NewFlat returns an empty index that stores dim-dimensional vectors.
func NewFlat(dim int, metric Metric) *FlatIndex {
	return &FlatIndex{
		dim:    dim,
		metric: metric,
	}
}

// Len reports how many vectors are stored.
func (idx *FlatIndex) Len() int {
	return len(idx.entries)
}

// Add inserts vec under the given id. It returns vector.ErrDimMismatch if vec's
// length does not match the index dimension.
func (idx *FlatIndex) Add(id string, vec vector.Vector) error {
	if len(vec) != idx.dim {
		return vector.ErrDimMismatch
	}
	idx.entries = append(idx.entries, entry{id: id, vec: vec})
	return nil
}

// Search returns the k stored vectors most similar to query, ordered best-first
// by cosine similarity. If k exceeds the number of stored vectors, all are
// returned; if k <= 0, none are.
//
// It returns vector.ErrDimMismatch if query's length does not match the index
// dimension, and propagates any error from the similarity computation (for
// example, vector.ErrZeroVector for a zero-magnitude query).
func (idx *FlatIndex) Search(query vector.Vector, k int) ([]Result, error) {
	if len(query) != idx.dim {
		return nil, vector.ErrDimMismatch
	}
	results := make([]Result, 0, len(idx.entries))
	for _, value := range idx.entries {
		score, err := idx.metric.fn(query, value.vec)
		if err != nil {
			return nil, err
		}
		results = append(results, Result{ID: value.id, Score: score})
	}
	sort.Slice(results, func(i, j int) bool {
		if idx.metric.sortDirection == "asc" {
			return results[i].Score < results[j].Score
		}
		return results[i].Score > results[j].Score
	})
	n := min(max(k, 0), len(results))
	return results[:n], nil
}
