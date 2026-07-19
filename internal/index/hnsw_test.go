package index

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/Rishishukla070702/quiver/internal/vector"
)

// A hand-built graph of 1-D points laid out along a line and chained as
// neighbours:  0 — 1 — 2 — 3 — 4.  Using L2 (lower distance = more similar), a
// greedy walk from node 0 toward a query at 4 should hop all the way to node 4.
func TestGreedySearch(t *testing.T) {
	nodes := []node{
		{id: "0", vec: vector.Vector{0}, neighbors: []int{1}},
		{id: "1", vec: vector.Vector{1}, neighbors: []int{0, 2}},
		{id: "2", vec: vector.Vector{2}, neighbors: []int{1, 3}},
		{id: "3", vec: vector.Vector{3}, neighbors: []int{2, 4}},
		{id: "4", vec: vector.Vector{4}, neighbors: []int{3}},
	}

	got, err := greedySearch(nodes, 0, vector.Vector{4}, L2)
	if err != nil {
		t.Fatalf("greedySearch error: %v", err)
	}
	if nodes[got].id != "4" {
		t.Errorf("greedySearch reached node %q, want %q", nodes[got].id, "4")
	}

	// From node 4 toward a query at 0, it should walk back down to node 0.
	got, err = greedySearch(nodes, 4, vector.Vector{0}, L2)
	if err != nil {
		t.Fatalf("greedySearch error: %v", err)
	}
	if nodes[got].id != "0" {
		t.Errorf("greedySearch reached node %q, want %q", nodes[got].id, "0")
	}
}

func TestSearchLayer(t *testing.T) {
	nodes := []node{
		{id: "0", vec: vector.Vector{0}, neighbors: []int{1}},
		{id: "1", vec: vector.Vector{1}, neighbors: []int{0, 2}},
		{id: "2", vec: vector.Vector{2}, neighbors: []int{1, 3}},
		{id: "3", vec: vector.Vector{3}, neighbors: []int{2, 4}},
		{id: "4", vec: vector.Vector{4}, neighbors: []int{3}},
	}

	// The top-2 nearest to a query at 4 must be node 4 then node 3, best-first.
	got, err := searchLayer(nodes, 0, vector.Vector{4}, L2, 2)
	if err != nil {
		t.Fatalf("searchLayer error: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d results, want 2", len(got))
	}
	if nodes[got[0].index].id != "4" || nodes[got[1].index].id != "3" {
		t.Errorf("top-2 = [%s, %s], want [4, 3]",
			nodes[got[0].index].id, nodes[got[1].index].id)
	}

	// ef=1 returns only the single best match (node 4).
	got, err = searchLayer(nodes, 0, vector.Vector{4}, L2, 1)
	if err != nil {
		t.Fatalf("searchLayer error: %v", err)
	}
	if len(got) != 1 || nodes[got[0].index].id != "4" {
		t.Fatalf("ef=1 result = %v, want a single node 4", got)
	}
}

// TestHNSWRecall builds a FlatIndex (exact) and an HNSW over the same random
// vectors, then measures how often HNSW's top-k matches the true top-k. This is
// the "grade against the exact oracle" idea made concrete — recall is the number
// that goes in the resume bullet.
func TestHNSWRecall(t *testing.T) {
	const (
		dim     = 16
		n       = 300
		k       = 10
		queries = 20
		m       = 16
		ef      = 100
	)
	rng := rand.New(rand.NewSource(42)) // fixed seed → deterministic test
	randVec := func() vector.Vector {
		v := make(vector.Vector, dim)
		for i := range v {
			v[i] = rng.Float32()
		}
		return v
	}

	flat := NewFlat(dim, L2)
	hnsw := NewHNSW(dim, L2, m, ef)
	for i := 0; i < n; i++ {
		v := randVec()
		id := fmt.Sprintf("v%d", i)
		if err := flat.Add(id, v); err != nil {
			t.Fatalf("flat.Add: %v", err)
		}
		if err := hnsw.Add(id, v); err != nil {
			t.Fatalf("hnsw.Add: %v", err)
		}
	}

	var total float64
	for q := 0; q < queries; q++ {
		query := randVec()
		want, err := flat.Search(query, k) // exact ground truth
		if err != nil {
			t.Fatalf("flat.Search: %v", err)
		}
		got, err := hnsw.Search(query, k)
		if err != nil {
			t.Fatalf("hnsw.Search: %v", err)
		}
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
		total += float64(hits) / float64(k)
	}
	recall := total / float64(queries)
	t.Logf("HNSW recall@%d over %d queries: %.3f", k, queries, recall)
	if recall < 0.85 {
		t.Errorf("recall@%d = %.3f, want >= 0.85", k, recall)
	}
}
