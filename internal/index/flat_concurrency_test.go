package index

import (
	"fmt"
	"sync"
	"testing"

	"github.com/Rishishukla070702/quiver/internal/vector"
)

// TestFlatConcurrentAccess hammers the index from many goroutines at once —
// writers (Add) and readers (Search) running simultaneously, exactly like the
// HTTP server does. Run with `go test -race` to surface the data race on the
// shared entries slice. Once FlatIndex is properly locked, this passes clean.
func TestFlatConcurrentAccess(t *testing.T) {
	idx := NewFlat(3, Cosine)
	q := vector.Vector{1, 0, 0}

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(2)
		go func(n int) {
			defer wg.Done()
			_ = idx.Add(fmt.Sprintf("v%d", n), vector.Vector{1, 0, 0})
		}(i)
		go func() {
			defer wg.Done()
			_, _ = idx.Search(q, 5)
		}()
	}
	wg.Wait()
}
