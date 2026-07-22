package index

import (
	"math"
	"math/rand"
	"sort"

	"github.com/Rishishukla070702/quiver/internal/vector"
)

// node is one vector in the HNSW proximity graph. It keeps a separate adjacency
// list per layer: neighbors[layer] holds the indices this node links to on that
// layer. A node lives on layers 0..len(neighbors)-1; layer 0 is the dense bottom
// layer that holds every node.
type node struct {
	id        string
	vec       vector.Vector
	neighbors [][]int
}

// better reports whether score a is "more similar" than score b under this
// metric's direction: higher is better for cosine, lower for L2.
func (m Metric) better(a, b float32) bool {
	if m.sortDirection == "asc" {
		return a < b
	}
	return a > b
}

// greedySearch walks a single layer of the proximity graph from entry toward
// query, always hopping to the neighbour most similar to query, and returns the
// index of the closest node it can reach (a local minimum). It is the greedy,
// single-node walk (ef=1) used to descend the sparse upper layers.
func greedySearch(nodes []node, entry int, query vector.Vector, metric Metric, layer int) (int, error) {
	current := entry
	bestScore, err := metric.fn(query, nodes[current].vec)
	if err != nil {
		return -1, err
	}

	for {
		var bestNeighbor int
		foundBetter := false

		for _, neighbor := range nodes[current].neighbors[layer] {
			score, err := metric.fn(query, nodes[neighbor].vec)
			if err != nil {
				return -1, err
			}
			if metric.better(score, bestScore) {
				bestScore = score
				bestNeighbor = neighbor
				foundBetter = true
			}
		}

		if !foundBetter {
			break
		}
		current = bestNeighbor
	}

	return current, nil
}

// candidate is a (node index, score) pair used by the beam search.
type candidate struct {
	index int
	score float32
}

// popBest removes and returns the most-similar candidate from c (under the
// metric), along with the remaining slice.
func popBest(c []candidate, m Metric) (candidate, []candidate) {
	best := 0
	for i := 1; i < len(c); i++ {
		if m.better(c[i].score, c[best].score) {
			best = i
		}
	}
	chosen := c[best]
	c[best] = c[len(c)-1] // swap-with-last, then shrink (order doesn't matter here)
	return chosen, c[:len(c)-1]
}

// worst returns the least-similar candidate in c. Assumes len(c) > 0.
func worst(c []candidate, m Metric) candidate {
	w := 0
	for i := 1; i < len(c); i++ {
		if m.better(c[w].score, c[i].score) {
			w = i
		}
	}
	return c[w]
}

// dropWorst removes the least-similar candidate from c and returns the result.
func dropWorst(c []candidate, m Metric) []candidate {
	w := 0
	for i := 1; i < len(c); i++ {
		if m.better(c[w].score, c[i].score) {
			w = i
		}
	}
	c[w] = c[len(c)-1]
	return c[:len(c)-1]
}

// searchLayer runs a beam search across one layer from entry toward query and
// returns the ef best candidates found, ordered best-first. ef is the beam
// width: a larger ef explores more of the graph, trading speed for recall. It
// propagates any error from the metric.
func searchLayer(nodes []node, entry int, query vector.Vector, metric Metric, ef, layer int) ([]candidate, error) {
	s0, err := metric.fn(query, nodes[entry].vec)
	if err != nil {
		return nil, err
	}
	visited := map[int]bool{entry: true}
	results := []candidate{{entry, s0}}
	frontier := []candidate{{entry, s0}}

	for len(frontier) > 0 {
		c, newFrontier := popBest(frontier, metric)
		frontier = newFrontier

		if len(results) == ef && metric.better(worst(results, metric).score, c.score) {
			break
		}

		for _, nb := range nodes[c.index].neighbors[layer] {
			if !visited[nb] {
				visited[nb] = true
				s, err := metric.fn(query, nodes[nb].vec)
				if err != nil {
					return nil, err
				}
				if len(results) < ef || metric.better(s, worst(results, metric).score) {
					frontier = append(frontier, candidate{nb, s})
					results = append(results, candidate{nb, s})
					if len(results) > ef {
						results = dropWorst(results, metric)
					}
				}
			}
		}
	}

	// Sort results best-first.
	sort.Slice(results, func(i, j int) bool {
		return metric.better(results[i].score, results[j].score)
	})

	return results, nil
}

// randomLevel draws a node's maximum layer from an exponentially-decaying
// distribution: most nodes get 0 and each higher layer is ~1/M as likely — the
// skip-list "coin flip" that keeps the upper layers sparse. mL is a
// normalisation factor, conventionally 1/ln(M).
func randomLevel(rng *rand.Rand, mL float64) int {
	level := -math.Log(rng.Float64()) * mL
	return int(level)
}

// HNSW is a hierarchical proximity-graph index (Hierarchical Navigable Small
// World). Nodes live on a random number of layers — many on layer 0, few up top
// — so a search descends the sparse upper layers in big jumps, then refines on
// the dense bottom layer, examining ~log(n) nodes instead of all n. It offers
// the same Add/Search shape as FlatIndex, so the two are directly comparable.
type HNSW struct {
	dim      int
	metric   Metric
	nodes    []node
	entry    int     // index of the top entry node, or -1 when empty
	maxLevel int     // highest layer currently populated (== level of entry node)
	m        int     // max neighbours to connect per layer
	ef       int     // beam width for construction and search
	mL       float64 // level-distribution normaliser, 1/ln(m)
	rng      *rand.Rand
}

// NewHNSW returns an empty index. m controls graph degree; ef is the beam width.
func NewHNSW(dim int, metric Metric, m, ef int) *HNSW {
	return &HNSW{
		dim:    dim,
		metric: metric,
		entry:  -1,
		m:      m,
		ef:     ef,
		mL:     1 / math.Log(float64(m)),
		rng:    rand.New(rand.NewSource(1)), // fixed seed → reproducible builds
	}
}

// Len reports how many vectors are stored.
func (h *HNSW) Len() int { return len(h.nodes) }

// Search returns the k nearest stored vectors to query, best-first. It descends
// the sparse upper layers greedily to get close, then beam-searches the bottom
// layer for the top matches.
func (h *HNSW) Search(query vector.Vector, k int) ([]Result, error) {
	if len(query) != h.dim {
		return nil, vector.ErrDimMismatch
	}
	if h.entry == -1 {
		return nil, nil // empty index
	}

	// Phase 1: from the top, greedily hop toward query on each upper layer,
	// carrying the closest node down as the entry point for the next layer.
	ep := h.entry
	for l := h.maxLevel; l > 0; l-- {
		next, err := greedySearch(h.nodes, ep, query, h.metric, l)
		if err != nil {
			return nil, err
		}
		ep = next
	}

	// Phase 2: a full beam search on the dense bottom layer.
	cands, err := searchLayer(h.nodes, ep, query, h.metric, h.ef, 0)
	if err != nil {
		return nil, err
	}
	n := min(max(k, 0), len(cands))
	out := make([]Result, n)
	for i := 0; i < n; i++ {
		out[i] = Result{ID: h.nodes[cands[i].index].id, Score: cands[i].score}
	}
	return out, nil
}

// Add inserts vec under id and links it into the hierarchy. It rolls a random
// level for the node, descends from the top entry to that level, then connects
// the node to its nearest neighbours on every layer from its level down to 0.
// It returns vector.ErrDimMismatch on a dimension mismatch.
//
// Node degree is currently unbounded; capping it is a later optimisation.
func (h *HNSW) Add(id string, vec vector.Vector) error {
	if len(vec) != h.dim {
		return vector.ErrDimMismatch
	}

	level := randomLevel(h.rng, h.mL)
	newIdx := len(h.nodes)
	h.nodes = append(h.nodes, node{
		id:        id,
		vec:       vec,
		neighbors: make([][]int, level+1), // one adjacency list per layer 0..level
	})

	// First node ever: it is the whole graph and the entry point.
	if h.entry == -1 {
		h.entry = newIdx
		h.maxLevel = level
		return nil
	}

	// Phase 1: descend the layers ABOVE the new node's level, greedily, to find
	// a good entry point to start inserting from.
	ep := h.entry
	for l := h.maxLevel; l > level; l-- {
		next, err := greedySearch(h.nodes, ep, vec, h.metric, l)
		if err != nil {
			return err
		}
		ep = next
	}

	// Phase 2: on each layer the node shares with the existing graph (from
	// min(level, maxLevel) down to 0), find neighbours and link both ways.
	for l := min(level, h.maxLevel); l >= 0; l-- {
		cands, err := searchLayer(h.nodes, ep, vec, h.metric, h.ef, l)
		if err != nil {
			return err
		}
		limit := min(h.m, len(cands))
		for i := 0; i < limit; i++ {
			nb := cands[i].index
			h.nodes[newIdx].neighbors[l] = append(h.nodes[newIdx].neighbors[l], nb)
			h.nodes[nb].neighbors[l] = append(h.nodes[nb].neighbors[l], newIdx)
		}
		ep = cands[0].index // carry the closest node down to the next layer
	}

	// If this node is taller than anything so far, it becomes the new top entry.
	if level > h.maxLevel {
		h.entry = newIdx
		h.maxLevel = level
	}
	return nil
}
