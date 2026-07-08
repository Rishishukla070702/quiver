// Package vector defines the core embedding type and the distance/similarity
// functions everything else in Quiver is built on. This is the math core
// (bottom layer of the architecture in ROADMAP.md).
package vector

import (
	"errors"
	"math"
)

// Vector is a dense, fixed-dimension embedding.
//
// It's just a slice of float32. Two deliberate choices:
//   - float32 (not float64): embedding models emit float32, and it halves memory.
//     When you're holding millions of vectors in RAM (see M4/M7), that matters.
//   - a named type over []float32 (not a struct): gives us methods and readable
//     signatures while staying a plain slice under the hood (zero overhead).
type Vector []float32

// ErrDimMismatch is returned when an operation gets vectors of different lengths.
// In Go, errors are values: we define them once as package-level variables so
// callers can compare against them with errors.Is.
var ErrDimMismatch = errors.New("vector: dimension mismatch")

// Dim returns the number of dimensions (the length of the vector).
func (v Vector) Dim() int { return len(v) }

// Dot returns the dot product of a and b: sum(a[i]*b[i]).
//
// The dot product is the building block for every other metric we'll add in M1
// (cosine similarity, and it relates to L2 distance too). For normalized
// vectors, a larger dot product means "more similar".
func Dot(a, b Vector) (float32, error) {
	if len(a) != len(b) {
		return 0, ErrDimMismatch
	}
	// `range` over a slice yields index, value. We only want the index here so
	// we can read the same position from both a and b.
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum, nil
}

// L2 returns the Euclidean (straight-line) distance between a and b:
//
//	sqrt( sum( (a[i]-b[i])^2 ) )
//
// Note the flip in meaning vs. Dot: here SMALLER means MORE similar
// (0 == identical vectors). L2 and cosine are the two metrics our flat index
// will support in M1.
//
// TODO(rishi): implement this. Hints, not answers:
//   - Return ErrDimMismatch when the lengths differ (mirror Dot above).
//   - Accumulate the sum of squared differences in a loop, like Dot does.
//   - Go's math.Sqrt takes and returns float64, but our data is float32 — so
//     you'll convert at the end: float32(math.Sqrt(float64(sum))).
//     That means you also need to add "math" to the import block at the top.
//   - Run `make test` — watch TestL2 go from red to green.
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
// between them — in the range [-1, 1]:
//
//	cosine(a, b) = Dot(a, b) / ( ||a|| * ||b|| )
//
// where ||v|| (the magnitude, or "L2 norm", of v) = sqrt( sum( v[i]^2 ) ).
//
// Like Dot (and unlike L2), LARGER means MORE similar: 1 = same direction,
// 0 = orthogonal/unrelated, -1 = opposite. This is the metric most text-
// embedding search uses, because it compares *direction* and ignores magnitude.
//
// TODO(rishi): implement this. Hints, not answers:
//   - Reuse what you built: the numerator is literally Dot(a, b). Call it, and
//     handle the error it returns — that gives you the dimension check for free.
//   - A vector's norm is sqrt(sum of its squares). Notice sum-of-squares of v
//     is just Dot(v, v) — so you can reuse Dot again for each norm if you like.
//   - Ignore the zero-vector case for now (a zero norm means divide-by-zero).
//     Do the happy path; we'll design that edge case together in review.
func Cosine(a, b Vector) (float32, error) {
	panic("TODO: implement Cosine (see the comment above)")
}
