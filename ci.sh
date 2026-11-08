#!/bin/sh
# CI levels:
#   ./ci.sh --fast   fmt + release unit tests (lib only)                       ~ seconds (incremental)
#   ./ci.sh          fmt + clippy -D warnings + all release tests              ~ 20 s + build
#   ./ci.sh --full   default + ignored (long) tests + debug tests (overflow/debug assertions)
# Every level first fetches the stored test data named in testdata.lock (no-op when present;
# TAPESPEED_TESTDATA=/path/archive.tar.zst for an offline copy), see tools/fetch-testdata.sh.
set -e
mode="${1:-default}"
"$(dirname "$0")/tools/fetch-testdata.sh"
cargo fmt --all --check
case "$mode" in
  --fast)
    cargo test -q --workspace --release --lib
    ;;
  default)
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test -q --workspace --release
    ;;
  --full)
    cargo clippy --workspace --all-targets -- -D warnings
    cargo test -q --workspace --release -- --include-ignored
    cargo test -q --workspace
    ;;
  *)
    echo "usage: $0 [--fast|--full]" >&2
    exit 2
    ;;
esac
