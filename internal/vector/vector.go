// Package vector defines Quiver's core embedding type and the distance and
// similarity functions the rest of the engine is built on.
package vector

import (
	"errors"
	"math"
)

// Vector is a dense, fixed-dimension embedding.
//
// It is stored as a slice of float32 rather than float64 because embedding
// models emit float32 and it halves the memory needed to hold large numbers of
// vectors.
type Vector []float32

// Errors returned by the distance functions. They are package-level values so
// callers can match them with errors.Is.
var (
	ErrDimMismatch = errors.New("vector: dimension mismatch")
	ErrZeroVector  = errors.New("vector: zero vector")
)

// Dim returns the number of dimensions (the length of the vector).
func (v Vector) Dim() int { return len(v) }

// Dot returns the dot product of a and b: sum(a[i]*b[i]). It is the primitive
// the other metrics are built on. It returns ErrDimMismatch if the lengths
// differ.
func Dot(a, b Vector) (float32, error) {
	if len(a) != len(b) {
		return 0, ErrDimMismatch
	}
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum, nil
}

// L2 returns the Euclidean distance between a and b: sqrt(sum((a[i]-b[i])^2)).
// Smaller means more similar (0 for identical vectors). It returns
// ErrDimMismatch if the lengths differ.
func L2(a, b Vector) (float32, error) {
	if len(a) != len(b) {
		return 0, ErrDimMismatch
	}
	var sum float32
	for i := range a {
		sum += (a[i] - b[i]) * (a[i] - b[i])
	}
	return float32(math.Sqrt(float64(sum))), nil
}

// Cosine returns the cosine similarity of a and b — the cosine of the angle
// between them — in the range [-1, 1], where 1 is the same direction, 0 is
// orthogonal, and -1 is opposite:
//
//	cosine(a, b) = Dot(a, b) / (||a|| * ||b||)
//
// Because it compares direction and ignores magnitude, it is the metric most
// commonly used for text-embedding search. It returns ErrDimMismatch if the
// lengths differ, and ErrZeroVector if either vector has zero magnitude (cosine
// is undefined in that case).
func Cosine(a, b Vector) (float32, error) {
	dot, err := Dot(a, b)
	if err != nil {
		return 0, err
	}
	normA, err := Dot(a, a)
	normB, err := Dot(b, b)
	if normA == 0 || normB == 0 {
		return 0, ErrZeroVector
	}
	return dot / (float32(math.Sqrt(float64(normA) * float64(normB)))), nil
}
