// Command quiver runs the Quiver vector database as an HTTP server.
package main

import (
	"flag"
	"log"
	"net/http"

	"github.com/Rishishukla070702/quiver/internal/index"
	"github.com/Rishishukla070702/quiver/internal/server"
)

func main() {
	// A small fixed dimension for now — easy to hand-type vectors when testing
	// with curl. Later this becomes configurable (flag/env) and matches the
	// embedding model's output size.
	indexKind := flag.String("index", "flat", "index type: flat or hnsw")
	m := flag.Int("m", 16, "HNSW graph degree")
	ef := flag.Int("ef", 64, "HNSW beam width")

	dim := flag.Int("dim", 384, "vector dimesionality")
	addr := flag.String("addr", ":8080", "address to listen on")
	flag.Parse()
	var idx index.Index
	switch *indexKind {
	case "flat":
		idx = index.NewFlat(*dim, index.Cosine)
	default:
		idx = index.NewHNSW(*dim, index.Cosine, *m, *ef)
	}
	srv := server.New(idx)
	log.Printf("Quiver listening on %s", *addr)

	// ListenAndServe blocks, serving each request in its own goroutine, until
	// the process is killed or it returns an error.
	if err := http.ListenAndServe(*addr, srv.Handler()); err != nil {
		log.Fatal(err)
	}
}
