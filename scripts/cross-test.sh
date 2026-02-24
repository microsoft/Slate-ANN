#!/usr/bin/env bash
# cross-test.sh — run the slate-ann workspace test suite under multiple CPU
# architectures using Docker + QEMU (binfmt_misc transparent emulation).
#
# WHY: the SIMD distance kernels have arch-specific code paths
# (x86_64: AVX2/AVX-512, aarch64: NEON). The dev box is x86_64, so the aarch64
# NEON path is otherwise never executed. Running the tests inside an aarch64
# container validates those kernels against the scalar oracle for real.
#
# HOW: the official `rust` image is multi-arch. `docker run --platform` pulls
# the right variant; aarch64 binaries run transparently via the
# binfmt-registered qemu-aarch64-static interpreter. Each run uses a
# per-arch CARGO_TARGET_DIR inside the container so host x86_64 build
# artifacts in ./target are never mixed with emulated aarch64 ones.
#
# Usage:
#   scripts/cross-test.sh                 # test all default arches
#   scripts/cross-test.sh arm64           # test only arm64
#   scripts/cross-test.sh amd64 arm64     # explicit list
#
# Env:
#   RUST_IMAGE   override the base image (default: rust:1-bookworm)
#   CARGO_CMD    override the in-container command
#                (default: cargo test --workspace)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUST_IMAGE="${RUST_IMAGE:-rust:1-bookworm}"
CARGO_CMD="${CARGO_CMD:-cargo test --workspace}"

# Default arch set if none given on the command line.
if [ "$#" -gt 0 ]; then
  ARCHES=("$@")
else
  ARCHES=(amd64 arm64)
fi

# Pick a container runtime: prefer docker, fall back to podman.
if command -v docker >/dev/null 2>&1; then
  RUNTIME=docker
elif command -v podman >/dev/null 2>&1; then
  RUNTIME=podman
else
  echo "error: neither docker nor podman found on PATH" >&2
  exit 1
fi

echo "==> repo:    $REPO_ROOT"
echo "==> runtime: $RUNTIME"
echo "==> image:   $RUST_IMAGE"
echo "==> command: $CARGO_CMD"
echo "==> arches:  ${ARCHES[*]}"
echo

fail=0
for arch in "${ARCHES[@]}"; do
  echo "============================================================"
  echo "  ARCH: linux/$arch"
  echo "============================================================"

  # - Mount the repo read-write at /work (cargo needs to write target).
  # - Redirect CARGO_TARGET_DIR to a container-only path so the emulated
  #   aarch64 artifacts never collide with the host x86_64 ./target.
  # - --platform selects the image variant; QEMU handles execution.
  if "$RUNTIME" run --rm \
      --platform "linux/$arch" \
      -v "$REPO_ROOT":/work \
      -w /work \
      -e CARGO_TARGET_DIR=/tmp/target \
      -e CARGO_TERM_COLOR=always \
      "$RUST_IMAGE" \
      bash -c "set -e; echo \"running on \$(uname -m)\"; $CARGO_CMD"; then
    echo "  [PASS] linux/$arch"
  else
    echo "  [FAIL] linux/$arch"
    fail=1
  fi
  echo
done

if [ "$fail" -ne 0 ]; then
  echo "==> cross-arch tests FAILED"
  exit 1
fi
echo "==> cross-arch tests PASSED on: ${ARCHES[*]}"
