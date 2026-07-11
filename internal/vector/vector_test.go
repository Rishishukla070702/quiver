package vector

import (
	"errors"
	"testing"
)

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
			if !errors.Is(err, tt.wantErr) {
				t.Fatalf("Dot() error = %v, want %v", err, tt.wantErr)
			}
			if tt.wantErr == nil && got != tt.want {
				t.Errorf("Dot() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestL2(t *testing.T) {
	tests := []struct {
		name    string
		a, b    Vector
		want    float32
		wantErr error
	}{
		{name: "identical", a: Vector{1, 2, 3}, b: Vector{1, 2, 3}, want: 0},
		{name: "3-4-5 triangle", a: Vector{0, 0}, b: Vector{3, 4}, want: 5}, // sqrt(9+16)
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

// approxEqual reports whether a and b are within eps of each other. Cosine
// results are compared with a tolerance because operations like sqrt introduce
// small floating-point rounding errors that make exact equality unreliable.
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
		{name: "zero vector (a)", a: Vector{0, 0}, b: Vector{1, 0}, wantErr: ErrZeroVector},
		{name: "zero vector (b)", a: Vector{1, 0}, b: Vector{0, 0}, wantErr: ErrZeroVector},
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
