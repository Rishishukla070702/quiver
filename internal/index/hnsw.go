package index

import (
	"sort"

	"github.com/Rishishukla070702/quiver/internal/vector"
)

// node is one vector in the HNSW proximity graph. neighbors holds indices into
// the graph's node slice — an adjacency list. Single-layer for now; the layer
// hierarchy arrives in a later stage.
type node struct {
	id        string
	vec       vector.Vector
	neighbors []int
}

// better reports whether score a is "more similar" than score b under this
// metric's direction: higher is better for cosine, lower for L2.
func (m Metric) better(a, b float32) bool {
	if m.sortDirection == "asc" {
		return a < b
	}
	return a > b
}

// greedySearch walks the proximity graph from entry toward query, always hopping
// to the neighbour most similar to query, and returns the index of the closest
// node it can reach (a local minimum). It is the greedy, single-node walk (ef=1);
// the layer hierarchy uses it to descend the sparse upper layers, while
// searchLayer is the wider beam search used on the dense bottom layer.
func greedySearch(nodes []node, entry int, query vector.Vector, metric Metric) (int, error) {
	current := entry
	bestScore, err := metric.fn(query, nodes[current].vec)
	if err != nil {
		return -1, err
	}

	for {
		var bestNeighbor int
		foundBetter := false

		for _, neighbor := range nodes[current].neighbors {
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

// searchLayer runs a beam search from entry toward query and returns the ef best
// candidates found, ordered best-first. ef is the beam width: a larger ef explores
// more of the graph, trading speed for recall. It propagates any error from the
// metric (for example a dimension mismatch or a zero vector).
func searchLayer(nodes []node, entry int, query vector.Vector, metric Metric, ef int) ([]candidate, error) {
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

		for _, nb := range nodes[c.index].neighbors {
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

// HNSW is a single-layer proximity-graph index (the layer hierarchy arrives in
// the next stage). It offers the same Add/Search shape as FlatIndex, so the two
// are interchangeable and directly comparable on recall.
type HNSW struct {
	dim    int
	metric Metric
	nodes  []node
	entry  int // index of the entry node, or -1 when the index is empty
	m      int // max neighbours to connect a new node to
	ef     int // beam width, used for both construction and search
}

// NewHNSW returns an empty index. m controls graph degree; ef is the beam width.
func NewHNSW(dim int, metric Metric, m, ef int) *HNSW {
	return &HNSW{dim: dim, metric: metric, entry: -1, m: m, ef: ef}
}

// Search returns the k nearest stored vectors to query, best-first.
func (h *HNSW) Search(query vector.Vector, k int) ([]Result, error) {
	if len(query) != h.dim {
		return nil, vector.ErrDimMismatch
	}
	if h.entry == -1 {
		return nil, nil // empty index
	}
	cands, err := searchLayer(h.nodes, h.entry, query, h.metric, h.ef)
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

// Add inserts vec under id and links it into the proximity graph: it finds the
// nearest existing nodes with searchLayer and connects the new node to the m
// closest via bidirectional edges. It returns vector.ErrDimMismatch if vec's
// length does not match the index dimension.
//
// Node degree is currently unbounded; capping it is a later optimisation.
func (h *HNSW) Add(id string, vec vector.Vector) error {
	if len(vec) != h.dim {
		return vector.ErrDimMismatch
	}
	if h.entry == -1 {
		h.nodes = append(h.nodes, node{id: id, vec: vec})
		h.entry = 0
		return nil
	}

	newIdx := len(h.nodes)
	h.nodes = append(h.nodes, node{id: id, vec: vec})

	cands, err := searchLayer(h.nodes, h.entry, vec, h.metric, h.ef)
	if err != nil {
		return err
	}

	limit := min(h.m, len(cands)) // connect to at most m (maybe fewer exist)
	for i := 0; i < limit; i++ {
		cand := cands[i] // one of the nearest existing nodes
		// edge in BOTH directions:
		h.nodes[newIdx].neighbors = append(h.nodes[newIdx].neighbors, cand.index)
		h.nodes[cand.index].neighbors = append(h.nodes[cand.index].neighbors, newIdx)
	}
	return nil
}
