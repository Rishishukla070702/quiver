// Command siftbench benchmarks Quiver's HNSW index on a SIFT-format dataset
// (TEXMEX .fvecs / .ivecs), reporting build time, query throughput, and recall@k
// against the dataset's provided ground truth. Point it at siftsmall (10K) or the
// full SIFT1M — the harness is identical.
//
// Usage:
//
//	go run ./cmd/siftbench \
//	  -base  siftsmall/siftsmall_base.fvecs \
//	  -query siftsmall/siftsmall_query.fvecs \
//	  -truth siftsmall/siftsmall_groundtruth.ivecs
package main

import (
	"bufio"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"time"

	"github.com/Rishishukla070702/quiver/internal/index"
	"github.com/Rishishukla070702/quiver/internal/vector"
)

func main() {
	base := flag.String("base", "", "path to base .fvecs")
	query := flag.String("query", "", "path to query .fvecs")
	truth := flag.String("truth", "", "path to ground-truth .ivecs")
	k := flag.Int("k", 10, "neighbours per query")
	m := flag.Int("m", 16, "HNSW M (graph degree)")
	ef := flag.Int("ef", 64, "HNSW beam width")
	flag.Parse()
	if *base == "" || *query == "" || *truth == "" {
		log.Fatal("need -base, -query and -truth")
	}

	baseVecs, err := readFvecs(*base)
	if err != nil {
		log.Fatalf("read base: %v", err)
	}
	queryVecs, err := readFvecs(*query)
	if err != nil {
		log.Fatalf("read query: %v", err)
	}
	gt, err := readIvecs(*truth)
	if err != nil {
		log.Fatalf("read ground truth: %v", err)
	}
	dim := len(baseVecs[0])
	fmt.Printf("loaded: %d base vectors, %d queries, dim %d (metric L2)\n",
		len(baseVecs), len(queryVecs), dim)

	// Build the index (SIFT is an L2 / Euclidean benchmark).
	idx := index.NewHNSW(dim, index.L2, *m, *ef)
	start := time.Now()
	for i, v := range baseVecs {
		if err := idx.Add(strconv.Itoa(i), v); err != nil {
			log.Fatalf("add %d: %v", i, err)
		}
	}
	build := time.Since(start)
	fmt.Printf("build: %s (%.0f vectors/sec), M=%d ef=%d\n",
		build.Round(time.Millisecond), float64(len(baseVecs))/build.Seconds(), *m, *ef)

	// Query: measure recall@k against ground truth, and throughput.
	var recallSum float64
	start = time.Now()
	for qi, q := range queryVecs {
		res, err := idx.Search(q, *k)
		if err != nil {
			log.Fatalf("search %d: %v", qi, err)
		}
		want := make(map[int]bool, *k)
		for i := 0; i < *k && i < len(gt[qi]); i++ {
			want[gt[qi][i]] = true
		}
		hits := 0
		for _, r := range res {
			if id, _ := strconv.Atoi(r.ID); want[id] {
				hits++
			}
		}
		recallSum += float64(hits) / float64(*k)
	}
	elapsed := time.Since(start)
	fmt.Printf("recall@%d: %.4f\n", *k, recallSum/float64(len(queryVecs)))
	fmt.Printf("queries: %d in %s → %.0f QPS, %.3f ms/query\n",
		len(queryVecs), elapsed.Round(time.Millisecond),
		float64(len(queryVecs))/elapsed.Seconds(),
		elapsed.Seconds()*1000/float64(len(queryVecs)))
}

// readFvecs reads a TEXMEX .fvecs file: each record is an int32 dimension
// followed by that many little-endian float32s.
func readFvecs(path string) ([]vector.Vector, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReader(f)
	var out []vector.Vector
	for {
		var dim int32
		if err := binary.Read(r, binary.LittleEndian, &dim); err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		v := make(vector.Vector, dim)
		if err := binary.Read(r, binary.LittleEndian, v); err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, nil
}

// readIvecs reads a TEXMEX .ivecs file: an int32 dimension followed by that many
// little-endian int32s (the ground-truth neighbour indices per query).
func readIvecs(path string) ([][]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := bufio.NewReader(f)
	var out [][]int
	for {
		var dim int32
		if err := binary.Read(r, binary.LittleEndian, &dim); err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		raw := make([]int32, dim)
		if err := binary.Read(r, binary.LittleEndian, raw); err != nil {
			return nil, err
		}
		row := make([]int, dim)
		for i, x := range raw {
			row[i] = int(x)
		}
		out = append(out, row)
	}
	return out, nil
}
