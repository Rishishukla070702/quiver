package vector

import (
	"errors"
	"testing"
)

// TestDot uses a "table-driven test" — the idiomatic Go pattern. Instead of one
// function per case, we list cases in a slice and loop over them. t.Run gives
// each case its own name in the output, so a failure tells you exactly which one.
func TestDot(t *testing.T) {
	tests := []struct {
		name    string
		a, b    Vector
		want    float32
		wantErr error
	}{
		{name: "orthogonal", a: Vector{1, 0}, b: Vector{0, 1}, want: 0},
		{name: "parallel", a: Vector{1, 2, 3}, b: Vector{1, 2, 3}, want: 14},  // 1+4+9
		{name: "with negatives", a: Vector{1, -2}, b: Vector{3, 4}, want: -5}, // 3-8
		{name: "dimension mismatch", a: Vector{1, 2}, b: Vector{1}, wantErr: ErrDimMismatch},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Dot(tt.a, tt.b)

			// errors.Is is how you compare errors in Go (handles wrapping).
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("Dot() error = %v, want %v", err, tt.wantErr)
			}
			if tt.wantErr == nil && got != tt.want {
				t.Errorf("Dot() = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestL2 defines what a correct L2 looks like. It fails until you implement L2.
// (I picked cases whose answers are exact whole numbers, so we can compare
// floats directly without worrying about rounding for now.)
func TestL2(t *testing.T) {
	tests := []struct {
		name    string
		a, b    Vector
		want    float32
		wantErr error
	}{
		{name: "identical", a: Vector{1, 2, 3}, b: Vector{1, 2, 3}, want: 0},
		{name: "3-4-5 triangle", a: Vector{0, 0}, b: Vector{3, 4}, want: 5}, // sqrt(9+16)=5
		{name: "unit apart", a: Vector{1, 0}, b: Vector{0, 0}, want: 1},
		{name: "dimension mismatch", a: Vector{1, 2}, b: Vector{1}, wantErr: ErrDimMismatch},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := L2(tt.a, tt.b)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("L2() error = %v, want %v", err, tt.wantErr)
			}
			if tt.wantErr == nil && got != tt.want {
				t.Errorf("L2() = %v, want %v", got, tt.want)
			}
		})
	}
}

// approxEqual compares floats with a small tolerance (epsilon). You almost never
// compare floats with == in real code: operations like sqrt introduce tiny
// rounding errors, so a "true" 1.0 might come out as 0.99999994. We assert
// "close enough" instead. Cosine needs this; L2's cases were exact integers.
func approxEqual(a, b, eps float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= eps
}

func TestCosine(t *testing.T) {
	const eps = 1e-6
	tests := []struct {
		name    string
		a, b    Vector
		want    float32
		wantErr error
	}{
		{name: "identical direction", a: Vector{1, 0}, b: Vector{1, 0}, want: 1},
		{name: "orthogonal", a: Vector{1, 0}, b: Vector{0, 1}, want: 0},
		{name: "opposite", a: Vector{2, 0}, b: Vector{-3, 0}, want: -1},
		{name: "same direction, bigger magnitude", a: Vector{1, 1}, b: Vector{2, 2}, want: 1},
		{name: "dimension mismatch", a: Vector{1, 2}, b: Vector{1}, wantErr: ErrDimMismatch},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Cosine(tt.a, tt.b)
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("Cosine() error = %v, want %v", err, tt.wantErr)
			}
			if tt.wantErr == nil && !approxEqual(got, tt.want, eps) {
				t.Errorf("Cosine() = %v, want %v (±%v)", got, tt.want, eps)
			}
		})
	}
}
