// Package server exposes a vector index over HTTP with a small JSON API.
package server

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/Rishishukla070702/quiver/internal/helpers"
	"github.com/Rishishukla070702/quiver/internal/index"
	"github.com/Rishishukla070702/quiver/internal/vector"
)

// Server wraps a vector index and serves it over HTTP.
type Server struct {
	idx index.Index
}

// New returns a Server backed by idx.
func New(idx index.Index) *Server {
	return &Server{idx: idx}
}

// Handler registers the routes and returns the root HTTP handler.
//
// The "METHOD /path" pattern syntax is built into net/http's ServeMux
// (Go 1.22+), so Quiver needs no third-party router.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("POST /vectors", s.handleAddVector)
	mux.HandleFunc("POST /query", s.handleQuery)
	return mux
}

// handleHealth reports that the server is running: HTTP 200 with {"status":"ok"}.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// addRequest is the JSON body accepted by POST /vectors.
type addRequest struct {
	ID     string        `json:"id"`
	Vector vector.Vector `json:"vector"`
}

// queryRequest is the JSON body accepted by POST /query.
type queryRequest struct {
	Vector vector.Vector `json:"vector"`
	K      int           `json:"k"`
}

// hit is one search result in the JSON response.
type hit struct {
	ID    string  `json:"id"`
	Score float32 `json:"score"`
}

// queryResponse is the JSON body returned by POST /query.
type queryResponse struct {
	Results []hit `json:"results"`
}

// handleAddVector inserts a vector sent as JSON. Bad input → 400, success → 201.
func (s *Server) handleAddVector(w http.ResponseWriter, r *http.Request) {
	var req addRequest
	if err := helpers.ReadJson(r, &req); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	if err := s.idx.Add(req.ID, req.Vector); err != nil {
		if errors.Is(err, vector.ErrDimMismatch) {
			http.Error(w, "vector dimension mismatch", http.StatusBadRequest)
			return
		}
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	helpers.WriteJson(w, http.StatusCreated, map[string]string{"status": "created"})
}

// handleQuery runs a nearest-neighbour search for the posted query vector and
// returns the ranked matches. Bad input → 400, success → 200.
func (s *Server) handleQuery(w http.ResponseWriter, r *http.Request) {
	var req queryRequest
	if err := helpers.ReadJson(r, &req); err != nil {
		http.Error(w, "invalid JSON body", http.StatusBadRequest)
		return
	}
	results, err := s.idx.Search(req.Vector, req.K)
	if err != nil {
		if errors.Is(err, vector.ErrDimMismatch) {
			http.Error(w, "vector dimension mismatch", http.StatusBadRequest)
			return
		}
		if errors.Is(err, vector.ErrZeroVector) {
			http.Error(w, "zero vector is not allowed", http.StatusBadRequest)
			return
		}
		http.Error(w, "internal error", http.StatusInternalServerError)
		return
	}
	hits := make([]hit, len(results))
	for i, res := range results {
		hits[i] = hit{ID: res.ID, Score: res.Score}
	}
	helpers.WriteJson(w, http.StatusOK, queryResponse{Results: hits})
}
