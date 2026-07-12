# Quiver

A **vector database, built from scratch in Go** — stores high-dimensional embeddings
and answers *"find the k most similar vectors"* over large collections, fast.

Built as a from-first-principles learning project: HNSW approximate-nearest-neighbor
index, WAL-backed persistence, concurrent access, benchmarked against production
vector databases. See [`ROADMAP.md`](./ROADMAP.md) for the full plan and progress.

> Status: **M2 — concurrent HTTP server** ✅ · next: M3 (real embeddings + data)

## Quick start

```sh
make check   # format, vet, and test
make test    # just run tests
```

## Why

Most engineers build *on top of* a vector database. This builds one — to learn the
data structures, storage, networking, and concurrency that power AI search.
