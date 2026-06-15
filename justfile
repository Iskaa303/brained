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

# Run the project example
example mode="debug" example="basic_simulation" *args:
    cargo run --example {{ example }} {{ if mode == "release" { "--release" } else { "" } }} -- {{ args }}

# Build and run tests
test:
    cargo test --all-features

# Remove compiled artifacts
clean:
    cargo clean

# Run format, lint, and tests
check: fmt lint test

# Generate documentation and open it in the browser immediately
doc:
    cargo doc --no-deps --open