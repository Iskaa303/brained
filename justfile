# List all recipes by default
default: list

# List all available recipes
list:
    @just --list

# Format code using rustfmt
fmt:
    cargo fmt --all

# Lint code using clippy
lint:
    cargo clippy --all-targets --all-features -- -D warnings

# Build the project
build mode="debug":
    cargo build {{ if mode == "release" { "--release" } else { "" } }}

# Run the project (with args if needed)
run mode="debug" *args:
    cargo run {{ if mode == "release" { "--release" } else { "" } }} -- {{ args }}

# Build and run the project
build-run mode="debug" *args:
    cargo build {{ if mode == "release" { "--release" } else { "" } }}
    cargo run {{ if mode == "release" { "--release" } else { "" } }} -- {{ args }}

# Run tests
test:
    cargo test

# Remove compiled artifacts
clean:
    cargo clean

# Run format, lint, and tests
check: fmt lint test