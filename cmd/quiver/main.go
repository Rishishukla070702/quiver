// Command quiver runs the Quiver vector database as an HTTP server.
package main

import (
	"log"
	"net/http"

	"github.com/Rishishukla070702/quiver/internal/index"
	"github.com/Rishishukla070702/quiver/internal/server"
)

func main() {
	// A small fixed dimension for now — easy to hand-type vectors when testing
	// with curl. Later this becomes configurable (flag/env) and matches the
	// embedding model's output size.
	idx := index.NewFlat(3, index.Cosine)
	srv := server.New(idx)

	const addr = ":8080"
	log.Printf("Quiver listening on %s", addr)

	// ListenAndServe blocks, serving each request in its own goroutine, until
	// the process is killed or it returns an error.
	if err := http.ListenAndServe(addr, srv.Handler()); err != nil {
		log.Fatal(err)
	}
}
