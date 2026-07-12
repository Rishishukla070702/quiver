package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Rishishukla070702/quiver/internal/index"
)

func TestHealth(t *testing.T) {
	s := New(index.NewFlat(3, index.Cosine))

	// httptest exercises a handler without opening a real network port:
	// NewRequest builds a fake request; NewRecorder captures what the handler
	// writes back. ServeHTTP runs the request through the router.
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()

	s.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", rec.Code, http.StatusOK)
	}
	var body map[string]string
	if err := json.NewDecoder(rec.Body).Decode(&body); err != nil {
		t.Fatalf("decoding response body: %v", err)
	}
	if body["status"] != "ok" {
		t.Errorf("status field = %q, want %q", body["status"], "ok")
	}
}

// postJSON sends body to path as a POST request and returns the recorder.
func postJSON(s *Server, path, body string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	return rec
}

func TestAddVector(t *testing.T) {
	s := New(index.NewFlat(3, index.Cosine))

	// A valid 3-dim vector is accepted → 201 Created, and the index grows.
	rec := postJSON(s, "/vectors", `{"id":"a","vector":[1,0,0]}`)
	if rec.Code != http.StatusCreated {
		t.Fatalf("valid insert: status = %d, want %d", rec.Code, http.StatusCreated)
	}
	if s.idx.Len() != 1 {
		t.Errorf("after valid insert, index has %d vectors, want 1", s.idx.Len())
	}

	// A wrong-dimension vector is the client's fault → 400 Bad Request,
	// and it must not be stored.
	rec = postJSON(s, "/vectors", `{"id":"b","vector":[1,0]}`)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("wrong-dim insert: status = %d, want %d", rec.Code, http.StatusBadRequest)
	}
	if s.idx.Len() != 1 {
		t.Errorf("after rejected insert, index has %d vectors, want 1", s.idx.Len())
	}
}

func TestQuery(t *testing.T) {
	s := New(index.NewFlat(2, index.Cosine))
	if postJSON(s, "/vectors", `{"id":"east","vector":[1,0]}`).Code != http.StatusCreated {
		t.Fatal("setup: failed to add east")
	}
	if postJSON(s, "/vectors", `{"id":"north","vector":[0,1]}`).Code != http.StatusCreated {
		t.Fatal("setup: failed to add north")
	}

	// A query pointing east should rank "east" first.
	rec := postJSON(s, "/query", `{"vector":[1,0],"k":1}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("query status = %d, want %d", rec.Code, http.StatusOK)
	}
	var resp struct {
		Results []struct {
			ID    string  `json:"id"`
			Score float32 `json:"score"`
		} `json:"results"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("decoding response: %v", err)
	}
	if len(resp.Results) != 1 || resp.Results[0].ID != "east" {
		t.Fatalf("results = %+v, want a single result [east]", resp.Results)
	}

	// A wrong-dimension query is the client's fault → 400.
	if rec := postJSON(s, "/query", `{"vector":[1,2,3],"k":1}`); rec.Code != http.StatusBadRequest {
		t.Errorf("wrong-dim query: status = %d, want %d", rec.Code, http.StatusBadRequest)
	}
}
