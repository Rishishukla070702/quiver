"""Embed scripts/corpus.txt with all-MiniLM-L6-v2 and load the vectors into Quiver.

Prerequisite (one-time):  pip install sentence-transformers
Run Quiver first (default dim is 384, matching this model):  go run ./cmd/quiver
Then:  python scripts/embed_and_load.py
"""

import json
import sys
import urllib.request

from sentence_transformers import SentenceTransformer

QUIVER = "http://localhost:8080"
MODEL = "all-MiniLM-L6-v2"  # emits 384-dimensional vectors


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        QUIVER + path, data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return resp.status


def main():
    with open("scripts/corpus.txt") as f:
        sentences = [line.strip() for line in f if line.strip()]

    print(f"Embedding {len(sentences)} sentences with {MODEL} ...")
    model = SentenceTransformer(MODEL)
    vectors = model.encode(sentences)  # -> numpy array, shape (n, 384)

    print("Loading vectors into Quiver ...")
    loaded = 0
    for text, vec in zip(sentences, vectors):
        # Use the sentence itself as the id, so search results show the text directly.
        status = post("/vectors", {"id": text, "vector": vec.tolist()})
        if status == 201:
            loaded += 1
        else:
            print(f"  WARN: got {status} for {text!r}", file=sys.stderr)
    print(f"Done. Loaded {loaded}/{len(sentences)} vectors.")


if __name__ == "__main__":
    main()
