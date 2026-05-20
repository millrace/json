#!/usr/bin/env bash
# tools/run_sanitizer_tests.sh -- json sanitizer harness.
#
# AOT-compiles a curated list of FFI- and lifetime-heavy test files
# with `mojo build --sanitize <kind>` and runs the resulting binaries
# one at a time, failing fast on the first error. Modeled after the
# flare harness; the curated list focuses on the surfaces that are
# the most likely sources of memory bugs:
#
#   * simdjson FFI (OwnedDLHandle + external_call into libsimdjson)
#   * tape-backed Document / Value lifetime + COW mutation
#   * the two-pass parser's UnsafePointer arithmetic
#   * the GPU adapter's index merge under the Apple-fallback flag
#   * the reflection serde walker (allocates per-field)
#
# Driven by `pixi run -e dev tests-asan` from `pixi.toml`. Standalone
# usage:
#
#   tools/run_sanitizer_tests.sh asan
#   tools/run_sanitizer_tests.sh asan tests/test_value.mojo  # single file
#
# Why AOT?
# --------
# JIT (`mojo run --sanitize address ...`) does not work because the
# JIT cannot resolve `__asan_*` runtime symbols statically. AOT
# (`mojo build --sanitize address ...`) emits a real binary that the
# linker can resolve against libasan.
#
# macOS note
# ----------
# At the time of writing, Mojo's bundled libasan expects
# `__asan_version_mismatch_check_v8`, while the system clang on
# Apple Silicon ships `__asan_version_mismatch_check_apple_clang_1700`.
# That version skew makes the linker step fail. Until Mojo and the
# host clang's ASan versions converge, this harness is Linux-only;
# on macOS it prints a clear skip message and exits 0 so dev-box
# `pixi run` invocations don't fail.

set -euo pipefail

KIND="${1:-asan}"
shift || true

case "${KIND}" in
  asan)
    SAN_FLAG="--sanitize address"
    SUFFIX="_asan"
    # detect_leaks=0   -- mute LSan exit-time chatter for one-shot
    #                     test binaries (we run each test in a fresh
    #                     process; libasan would otherwise complain
    #                     about per-test leaks that don't matter for
    #                     a use-after-free / OOB read sweep).
    # abort_on_error=1 -- turn recoverable findings into hard exits
    #                     so CI fails fast on the first hit.
    # verify_asan_link_order=0 -- disable the runtime preload-order
    #                     check; conda's LD_LIBRARY_PATH injects
    #                     libstdc++ ahead of libasan and tripping
    #                     this guard kills the test before it runs.
    SAN_ENV="ASAN_OPTIONS=detect_leaks=0:abort_on_error=1:symbolize_inlines=1:verify_asan_link_order=0"
    ;;
  *)
    echo "usage: $0 asan [test_file...]" >&2
    exit 2
    ;;
esac

KIND_UPPER=$(echo "${KIND}" | tr '[:lower:]' '[:upper:]')

# macOS skip gate (see header comment). bash 3.2 on stock macOS lacks
# the ${VAR^^} expansion, hence the explicit tr above.
if [[ "$(uname)" == "Darwin" ]]; then
  echo "── ${KIND_UPPER}: skipped on macOS"
  echo "   Mojo's bundled ${KIND} expects __asan_version_mismatch_check_v8;"
  echo "   Apple Silicon clang ships __asan_version_mismatch_check_apple_clang_1700."
  echo "   Linux CI exercises the same harness end-to-end."
  exit 0
fi

# ── default test inventory ────────────────────────────────────────
# Curated list -- keep aligned with the surfaces enumerated in the
# header. Order is roughly bottom-up so the first failure points at
# the most primitive layer.
ASAN_TESTS=(
  # Document + Value tape lifetime
  "tests/test_document.mojo"
  "tests/test_value.mojo"
  "tests/test_value_mutation.mojo"
  # Two-pass CPU parser (stage 1 + stage 2 walker, UnsafePointer math)
  "tests/test_parser.mojo"
  "tests/test_stage1_equivalence.mojo"
  # simdjson FFI boundary
  "tests/test_backend_equivalence.mojo"
  # Round-trip serializer + reflection walker (allocates per-field)
  "tests/test_serialize.mojo"
  "tests/test_reflection.mojo"
  "tests/test_serde.mojo"
  # RFC-driven query / patch / validate paths
  "tests/test_jsonpath.mojo"
  "tests/test_patch.mojo"
  "tests/test_schema.mojo"
  # End-to-end mix (drives both backends + reflection on real fixtures)
  "tests/test_e2e.mojo"
)

# Allow caller to override the test list.
if [[ $# -gt 0 ]]; then
  TESTS=( "$@" )
else
  TESTS=( "${ASAN_TESTS[@]}" )
fi

mkdir -p target/sanitize

PASS=0
FAIL=0
START_NS=$(date +%s%N)

for test_file in "${TESTS[@]}"; do
  base=$(basename "${test_file}" .mojo)
  out="target/sanitize/${base}${SUFFIX}"

  printf '── %-40s build (%s) … ' "${base}" "${KIND}"
  if ! pixi run -e dev mojo build ${SAN_FLAG} -D ASSERT=all -I . "${test_file}" -o "${out}" \
       > "target/sanitize/${base}${SUFFIX}.build.log" 2>&1; then
    echo "BUILD FAILED"
    cat "target/sanitize/${base}${SUFFIX}.build.log"
    FAIL=$((FAIL + 1))
    continue
  fi
  echo "ok"

  printf '   %-40s run   (%s) … ' "${base}" "${KIND}"
  if env ${SAN_ENV} "./${out}" > "target/sanitize/${base}${SUFFIX}.run.log" 2>&1; then
    summary=$(grep -E '^Summary' "target/sanitize/${base}${SUFFIX}.run.log" | tail -1 || true)
    echo "PASS — ${summary:-no summary}"
    PASS=$((PASS + 1))
  else
    echo "FAILED"
    tail -40 "target/sanitize/${base}${SUFFIX}.run.log"
    FAIL=$((FAIL + 1))
  fi
done

END_NS=$(date +%s%N)
ELAPSED_S=$(( (END_NS - START_NS) / 1000000000 ))

echo
echo "── ${KIND_UPPER} summary: ${PASS} passed, ${FAIL} failed in ${ELAPSED_S}s"

if [[ ${FAIL} -gt 0 ]]; then
  exit 1
fi
