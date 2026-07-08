# Quiver — a vector database from scratch, in Go

> Working name: **quiver** (a quiver holds arrows; arrows are vectors). Rename anytime.

A from-scratch vector database: it stores high-dimensional embeddings and answers
"find the *k* most similar vectors to this query" — fast, over millions of vectors.
Built to learn CS fundamentals from first principles, in Go, as a portfolio anchor.

---

## Why this project

- **Trending:** vector DBs are the backbone of every RAG / AI-search app in 2026.
- **Fundamentals:** forces data structures, algorithms, storage engines, networking,
  concurrency — the internals that day-jobs let you skip.
- **Standout:** almost everyone builds *on top of* a vector DB. Building one from
  scratch, with benchmarks vs. pgvector/FAISS, is a senior-tier interview story.
- **Cheap:** builds and runs entirely on a laptop. ~$0. (Optional ~$5/mo VPS for a live demo.)

## The one-line resume bullet we're working toward

> *Built a vector database from scratch in Go — HNSW ANN index, WAL-backed persistence,
> concurrent reads/writes — achieving **~X% recall@10 at Y QPS (Zms p99)** on SIFT1M,
> within N% of pgvector.*

(The X/Y/Z get filled in by Milestone 7. Metrics are the whole point.)

---

## What a vector database actually does (orientation)

1. An **embedding model** turns text/images into vectors (arrays of floats). *Not our job to build* — we use a local model.
2. We **store** those vectors + their IDs + metadata.
3. Given a **query vector**, we return the *k* nearest (most similar) stored vectors.
4. "Nearest" = a **distance metric** (cosine / L2 / dot product).
5. The hard part: doing step 3 in **milliseconds over millions of vectors** without
   comparing against every one. That's the **ANN index** (we build **HNSW**).

## Architecture (what we'll build, layer by layer)

```
                 ┌─────────────────────────────────────┐
   client  ───►  │  API layer (HTTP/JSON, later gRPC)   │   insert / query / delete
                 ├─────────────────────────────────────┤
                 │  Engine: index + metadata + filters  │
                 │   ├─ Flat index (exact, baseline)    │
                 │   └─ HNSW index (approximate, fast)  │   ◄── the centerpiece
                 ├─────────────────────────────────────┤
                 │  Distance metrics (cosine/L2/dot)    │   ◄── the math core
                 ├─────────────────────────────────────┤
                 │  Storage: WAL + snapshots on disk    │   ◄── durability
                 └─────────────────────────────────────┘
```

---

## Tech stack

- **Language:** Go (single binary, great concurrency, big in infra hiring).
- **API:** start with net/http + JSON; add gRPC later if we want.
- **Embeddings:** generated locally with Python `sentence-transformers`
  (`all-MiniLM-L6-v2`, 384-dim) — free, offline. Quiver itself never calls an API.
- **Benchmark datasets:** SIFT1M / GloVe from [ann-benchmarks](https://github.com/erikbern/ann-benchmarks).
- **Profiling:** Go `pprof` + `testing.B` benchmarks.
- **No cloud required.** Optional live demo on Fly.io free tier or a ~$5 Hetzner VPS.

---

## Milestone plan (~3–4 months, evenings/weekends)

Each milestone is independently demoable. Never a broken tree.

### M0 — Go bootcamp + repo scaffold  ·  ~1 week
- Install Go; do the [Tour of Go](https://go.dev/tour/). Learn: structs, interfaces,
  slices, maps, errors, goroutines/channels, `go test`.
- Scaffold: `go mod init`, package layout, first passing test, linter, Makefile.
- **Deliverable:** repo builds; `go test ./...` green; a `Vector` type + dot-product function.
- **Fundamentals:** the Go language & tooling.

### M1 — Distance metrics + flat (brute-force) index  ·  ~1 week
- Implement cosine similarity, L2, dot product (with unit tests on known vectors).
- Flat index: store vectors in a slice, brute-force top-*k* using `container/heap`.
- **Deliverable:** insert vectors, get **exact** top-*k*. A correct (slow) vector search.
- **Fundamentals:** numerical computing, heaps, algorithmic complexity (O(n·d) per query).

### M2 — Make it a server  ·  ~1 week
- HTTP/JSON API: `POST /vectors`, `POST /query`, `DELETE /vectors/{id}`.
- A tiny CLI/curl client; request validation; structured errors.
- **Deliverable:** a running vector-search **server** you talk to over the network.
- **Fundamentals:** networking, HTTP, serialization, API design.

### M3 — Real embeddings + real data  ·  ~1 week
- Python script: embed a corpus (e.g. Wikipedia/arXiv abstracts) → export vectors.
- Bulk-load into Quiver; run semantic queries.
- **Deliverable:** semantic search over a real corpus. First "wow" demo.
- **Fundamentals:** the embedding pipeline; batch ingestion (you've done this at scale — good bridge).

### M4 — HNSW index  ·  ~2–3 weeks  ·  ⭐ the heart
- Read the [HNSW paper](https://arxiv.org/abs/1603.09320); understand the intuition
  (navigable small-world graph + skip-list-like layers).
- Implement: graph structure, insertion (greedy search + neighbor heuristic), search.
- Tune params M / efConstruction / efSearch; measure **recall vs. the flat index**.
- **Deliverable:** fast approximate search. The centerpiece.
- **Fundamentals:** graph data structures, probabilistic structures, greedy search.

### M5 — Persistence & durability  ·  ~2 weeks
- Serialize/load the index; a **write-ahead log** (WAL); crash recovery (replay on start); snapshots.
- **Deliverable:** the DB survives restarts and crashes without data loss.
- **Fundamentals:** storage engines, serialization, WAL, crash recovery, file I/O.

### M6 — Concurrency + production concerns  ·  ~2 weeks
- Safe concurrent reads/writes (RWMutex / per-segment locking); metadata filtering; batch ops.
- **Deliverable:** correct under concurrent load (race detector clean: `go test -race`).
- **Fundamentals:** Go concurrency model, locking, data-race avoidance.

### M7 — Benchmarks + optimization  ·  ~2 weeks  ·  💰 the resume metrics
- Harness measuring **recall@k, QPS, p50/p95/p99 latency, memory**.
- Run on SIFT1M; **compare vs. pgvector / FAISS**. Profile with pprof; optimize hot paths.
- Optional: SIMD-accelerated distance; scalar/product **quantization** to cut memory.
- **Deliverable:** the numbers that fill in the resume bullet.
- **Fundamentals:** benchmarking methodology, profiling, performance engineering.

### M8 — Polish & showcase  ·  ongoing
- README with architecture diagrams, benchmark charts, "why I built this."
- A blog post explaining HNSW (teaching it proves mastery).
- Optional: deploy a live demo; build a small **RAG "chat with docs"** on top of Quiver
  (closes the AI-gap loop; the LLM call is the *only* thing that could cost pennies).
- **Deliverable:** portfolio-ready — repo + writeup + (optional) live demo.

---

## Stretch goals (post-v1, each its own resume line)
- Rewrite the HNSW hot path in **Rust**, benchmark the delta ("I profiled and rewrote…").
- Sharding / a second node → distributed vector search.
- Alternative indexes (IVF, LSH) with a comparison writeup.

## Definition of done (v1)
- [ ] Insert / query / delete over HTTP
- [ ] HNSW index with measured recall
- [ ] Survives restart (WAL + snapshot)
- [ ] Concurrency-safe (`-race` clean)
- [ ] Benchmarked on SIFT1M with published numbers vs. a real vector DB
- [ ] README + architecture diagram + writeup
