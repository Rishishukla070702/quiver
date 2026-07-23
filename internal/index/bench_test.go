package index

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/Rishishukla070702/quiver/internal/vector"
)

// Scale for the comparison. Bump benchN up to see the gap widen.
const (
	benchN   = 5000 // vectors in the index (bump this to watch the flat-vs-HNSW gap widen)
	benchDim = 32   // dimensions per vector
	benchK   = 10   // neighbours per query
	benchM   = 16   // HNSW graph degree
	benchEf  = 64   // HNSW beam width
)

// randomVectors returns n deterministic random vectors of the given dim.
func randomVectors(n, dim int, seed int64) []vector.Vector {
	rng := rand.New(rand.NewSource(seed))
	vs := make([]vector.Vector, n)
	for i := range vs {
		v := make(vector.Vector, dim)
		for j := range v {
			v[j] = rng.Float32()
		}
		vs[i] = v
	}
	return vs
}

// BenchmarkFlatSearch times an exact brute-force query. The index is built once,
// OUTSIDE the timed loop; b.ResetTimer() drops that setup from the measurement,
// and the loop runs b.N times (Go picks b.N so the timed part lasts ~1s).
func BenchmarkFlatSearch(b *testing.B) {
	data := randomVectors(benchN, benchDim, 1)
	idx := NewFlat(benchDim, L2)
	for i, v := range data {
		_ = idx.Add(fmt.Sprintf("v%d", i), v)
	}
	queries := randomVectors(100, benchDim, 2)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = idx.Search(queries[i%len(queries)], benchK)
	}
}

// BenchmarkHNSWSearch times the same queries through the HNSW index.
func BenchmarkHNSWSearch(b *testing.B) {
	data := randomVectors(benchN, benchDim, 1)
	idx := NewHNSW(benchDim, L2, benchM, benchEf)
	for i, v := range data {
		_ = idx.Add(fmt.Sprintf("v%d", i), v)
	}
	queries := randomVectors(100, benchDim, 2)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = idx.Search(queries[i%len(queries)], benchK)
	}
}

// TestHNSWRecallAtScale reports HNSW recall vs the exact FlatIndex at bench scale,
// so the speed numbers come with an accuracy number attached.
func TestHNSWRecallAtScale(t *testing.T) {
	data := randomVectors(benchN, benchDim, 1)
	flat := NewFlat(benchDim, L2)
	hnsw := NewHNSW(benchDim, L2, benchM, benchEf)
	for i, v := range data {
		id := fmt.Sprintf("v%d", i)
		_ = flat.Add(id, v)
		_ = hnsw.Add(id, v)
	}

	queries := randomVectors(50, benchDim, 2)
	var total float64
	for _, q := range queries {
		want, _ := flat.Search(q, benchK)
		got, _ := hnsw.Search(q, benchK)
		truth := make(map[string]bool, len(want))
		for _, r := range want {
			truth[r.ID] = true
		}
		hits := 0
		for _, r := range got {
			if truth[r.ID] {
				hits++
			}
		}
		total += float64(hits) / float64(benchK)
	}
	t.Logf("recall@%d at n=%d, dim=%d, M=%d, ef=%d: %.3f",
		benchK, benchN, benchDim, benchM, benchEf, total/float64(len(queries)))
}
