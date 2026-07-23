package index

import "github.com/Rishishukla070702/quiver/internal/vector"

type Index interface {
	// Add inserts a vector under the given ID. It returns vector.ErrDimMismatch if
	// the vector's length does not match the index dimension.
	Add(id string, vec vector.Vector) error

	// Search returns the k stored vectors most similar to query, ordered best-first
	// by cosine similarity. If k exceeds the number of stored vectors, all are
	// returned; if k <= 0, none are.
	//
	// It returns vector.ErrDimMismatch if query's length does not match the index
	// dimension, and propagates any error from the similarity computation (for
	// example, if a vector contains NaN).
	Search(query vector.Vector, k int) ([]Result, error)
	Len() int
}

var (
	_ Index = (*FlatIndex)(nil)
	_ Index = (*HNSW)(nil)
)
