# Quiver

A **vector database built from scratch in Go** — it stores high-dimensional
embeddings and answers *"find the k most similar vectors"* in **sub-linear time**
via a hand-rolled HNSW index, served over an HTTP/JSON API, with write-ahead-log
persistence. **The core uses only the Go standard library** — no third-party
vector-search or web frameworks.

> **≈6.4× faster than exact brute-force search at 20K vectors, while keeping
> 96.6% recall@10** — measured against a self-built exact baseline.

Built as a from-first-principles systems project: the goal was to understand the
data structures, algorithms, storage, networking, and concurrency that power AI
search by *building* them, not importing them.

---

## What's inside

- **Two index backends behind one `Index` interface**, swappable at runtime:
  - `FlatIndex` — exact brute-force search (the correctness oracle).
  - `HNSW` — Hierarchical Navigable Small World graph: a layered proximity graph
    with greedy descent through sparse upper layers + beam search on the dense
    bottom layer, giving ~`log(n)` queries.
- **HTTP/JSON API** (`net/http`): `POST /vectors`, `POST /query`, `GET /health`.
- **Concurrency-safe** — `RWMutex`-guarded, verified with `go test -race`.
- **Durability** — a write-ahead log (`fsync` on every write, replayed on startup)
  so the index survives crashes and restarts with zero data loss.
- **Pluggable distance metrics** — cosine and L2.
- **Real semantic search** — a demo that embeds a text corpus with
  `sentence-transformers` and queries Quiver by *meaning*, not keywords.

## Benchmarks

### On the standard SIFT dataset (SIFT10K, real ground truth)

10,000 × 128-dim base vectors, 100 queries, **recall@10 measured against the
dataset's provided ground truth**. Single-threaded, `M=16`. The `ef` dial trades
recall for throughput:

| `ef` | recall@10 | throughput   | latency   |
|-----:|----------:|-------------:|----------:|
|   16 | 0.919     | 16,100 QPS   | 0.06 ms/q |
|   64 | **0.988** |  5,000 QPS   | 0.20 ms/q |
|  128 | 0.990     |  2,470 QPS   | 0.41 ms/q |
|  256 | 0.990     |    960 QPS   | 1.05 ms/q |

```sh
# Reproducible; point the same tool at full SIFT1M for the million-scale run.
go run ./cmd/siftbench -base siftsmall_base.fvecs \
  -query siftsmall_query.fvecs -truth siftsmall_groundtruth.ivecs -ef 64
```

### vs. exact brute force (synthetic, showing the scaling law)

`FlatIndex` (exact) vs. `HNSW`, dim 32, `M=16`, `ef=64`, L2:

| Vectors | FlatIndex | HNSW    | Speedup  |
|--------:|----------:|--------:|---------:|
|   5,000 | 0.63 ms   | 0.33 ms | 1.9×     |
|  20,000 | 2.82 ms   | 0.44 ms | **6.4×** |

Flat search is O(n); HNSW is ~O(log n), so the gap widens with scale.

## How it works

HNSW stacks proximity graphs: every node sits on layer 0, and a random,
exponentially-thinning subset also lives on higher "express-lane" layers. A search
takes big greedy hops down the sparse top layers to get *near*, then one wide beam
search on the bottom layer to get the exact top-k.

📖 **Full visual walkthrough (diagrams + insertion/search traces):
[`docs/hnsw.md`](./docs/hnsw.md).**

## Quickstart

```sh
# Run the server (HNSW-backed, with persistence)
go run ./cmd/quiver -index=hnsw -dim=3 -wal=/tmp/quiver.wal

# In another terminal:
curl -X POST localhost:8080/vectors -d '{"id":"a","vector":[1,0,0]}'
curl -X POST localhost:8080/vectors -d '{"id":"b","vector":[0,1,0]}'
curl -s -X POST localhost:8080/query  -d '{"vector":[1,0,0],"k":2}'
# → {"results":[{"id":"a","score":1},{"id":"b","score":0}]}
```

Kill the server and start it again on the same `-wal` file — your vectors are
still there.

### Semantic search demo (real embeddings)

```sh
pip install sentence-transformers        # one-time
go run ./cmd/quiver                      # dim defaults to 384 (all-MiniLM-L6-v2)
python scripts/embed_and_load.py         # embed a corpus, load it in
python scripts/search.py "a pet that purrs"
# → ranks "A cat curls up on the windowsill and purrs..." first — by meaning, not keywords
```

## Project layout

```
cmd/quiver/       HTTP server entrypoint (flags: -index, -dim, -m, -ef, -wal, -addr)
internal/vector/  distance metrics — dot product, L2, cosine
internal/index/   FlatIndex (exact) + HNSW (approximate) behind the Index interface
internal/server/  HTTP/JSON API + handlers
internal/wal/     write-ahead log + PersistentIndex (durability)
scripts/          embedding + search demo (Python)
docs/hnsw.md      visual explainer
```

## Development

```sh
make check   # gofmt + go vet + tests
make race    # tests under the race detector
```

## Status & roadmap

**v1 complete:** distance metrics → exact index → HTTP API → real embeddings →
HNSW → concurrency → WAL persistence → degree-capping → SIFT benchmarks.
**Possible extensions:** heap-backed candidate lists (faster inner loop), a full
SIFT1M run, a RAG "chat with docs" demo. See [`ROADMAP.md`](./ROADMAP.md).
