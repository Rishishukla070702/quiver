# Common tasks. Run `make test` etc. from the project root.
.PHONY: build test race fmt vet check

build:      ## compile everything
	go build ./...

test:       ## run all tests
	go test ./...

race:       ## run tests with the data-race detector (we'll rely on this in M6)
	go test -race ./...

fmt:        ## auto-format all code (gofmt is non-negotiable in Go)
	go fmt ./...

vet:        ## static analysis for common mistakes
	go vet ./...

check: fmt vet test  ## the pre-commit gate: format, vet, test
