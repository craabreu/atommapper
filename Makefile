PYTHON ?= python
RUFF ?= ruff
TY ?= ty
PATHS = src tests

.PHONY: check lint types test format rust build

check: lint types rust build test

lint:
	$(RUFF) check $(PATHS)
	$(RUFF) format --check $(PATHS)

format:
	$(RUFF) format $(PATHS)
	$(RUFF) check --fix --exit-zero $(PATHS)
	cd rust && cargo fmt

types:
	$(TY) check src tests

# The Stage 1 engine. PYO3_PYTHON points the binding's build at the same interpreter.
rust:
	cd rust && cargo fmt --check
	cd rust && PYO3_PYTHON=$$(command -v $(PYTHON)) cargo clippy --all-targets --features python -- -D warnings
	cd rust && cargo test --release

# Rebuild the extension in place, so the tests run against the current Rust source.
build:
	$(PYTHON) -m pip install -q -e . --no-deps

test:
	$(PYTHON) -m pytest -q
