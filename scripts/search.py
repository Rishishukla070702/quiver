"""Embed a query string with the same model and search Quiver for nearest sentences.

Usage:  python scripts/search.py "a pet that purrs" [k]
"""

import json
import sys
import urllib.request

from sentence_transformers import SentenceTransformer

QUIVER = "http://localhost:8080"
MODEL = "all-MiniLM-L6-v2"  # MUST match the model used to load the corpus


def main():
    if len(sys.argv) < 2:
        print('usage: python scripts/search.py "your query" [k]', file=sys.stderr)
        sys.exit(1)
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    model = SentenceTransformer(MODEL)
    vec = model.encode(query).tolist()

    data = json.dumps({"vector": vec, "k": k}).encode()
    req = urllib.request.Request(
        QUIVER + "/query", data=data, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        results = json.load(resp)["results"]

    print(f'\nQuery: {query!r}\n')
    for r in results:
        print(f"  {r['score']:.3f}  {r['id']}")
    print()


if __name__ == "__main__":
    main()
