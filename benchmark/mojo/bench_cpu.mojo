# Mojo CPU benchmark for the json library.
#
# Reports throughput for two parser paths under two access patterns:
#
#   - json_cpu/scalar           : lazy v0.1 Value, top-level only
#   - json_cpu/simd             : lazy v0.1 Value, top-level only
#   - json_cpu/tape             : eager Document, top-level only
#   - json_cpu/scalar_traverse  : lazy + full DOM traversal
#   - json_cpu/simd_traverse    : lazy + full DOM traversal
#   - json_cpu/tape_traverse    : eager + full DOM traversal
#
# Top-level rows measure "parse, then peek the root" — the lazy paths
# only validate structure here; the eager tape path pays full
# materialisation cost. The traversal rows measure parse + walk every
# value, which is the fair comparison for code that actually reads
# the whole document.
#
# Usage:
#   mojo -I . benchmark/mojo/bench_cpu.mojo [json_file]

from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    ThroughputMeasure,
    BenchMetric,
)
from std.pathlib import Path
from std.sys import argv
from json.cpu import parse_cpu_native_tape
from json.value import Value


def _walk_strict(v: Value) -> Bool:
    """Strict counterpart to `_walk`: returns False if any access
    raises. Used once outside the bench loop to surface correctness
    bugs the timing-only walk would otherwise hide."""
    if v.is_object():
        try:
            for pair in v.object_items():
                if not _walk_strict(pair[1]):
                    return False
        except:
            return False
        return True
    if v.is_array():
        try:
            for child in v.array_items():
                if not _walk_strict(child):
                    return False
        except:
            return False
        return True
    if v.is_string():
        try:
            _ = v.string_value()
        except:
            return False
        return True
    if v.is_int():
        try:
            _ = v.int_value()
        except:
            return False
        return True
    if v.is_float():
        try:
            _ = v.float_value()
        except:
            return False
        return True
    if v.is_bool():
        try:
            _ = v.bool_value()
        except:
            return False
        return True
    return True


def _walk(v: Value) -> Int:
    """Recursively visit every node, returning a checksum so the
    traversal cannot be optimised away.

    For each leaf we mix one int's worth of payload into the
    accumulator; for containers we recurse. Errors during access are
    swallowed so the bench reports throughput even when a backend has
    a correctness bug under deep traversal — we can spot the bug
    separately, but we want the wall-clock comparison either way.
    """
    if v.is_object():
        var sum = 0
        try:
            for pair in v.object_items():
                sum += pair[0].byte_length()
                sum += _walk(pair[1])
        except:
            pass
        return sum
    if v.is_array():
        var sum = 0
        try:
            for child in v.array_items():
                sum += _walk(child)
        except:
            pass
        return sum
    if v.is_string():
        try:
            return v.string_value().byte_length()
        except:
            return 0
    if v.is_int():
        try:
            return Int(v.int_value())
        except:
            return 0
    if v.is_float():
        try:
            return Int(v.float_value())
        except:
            return 0
    if v.is_bool():
        try:
            return 1 if v.bool_value() else 0
        except:
            return 0
    return 0


def main() raises:
    var args = argv()
    var path: String
    if len(args) > 1:
        path = String(args[1])
    else:
        path = "benchmark/simdjson/jsonexamples/twitter.json"

    print(
        "========================================================================"
    )
    print("json CPU Benchmark")
    print(
        "========================================================================"
    )
    print()

    var json_str = Path(path).read_text()

    var file_size = len(json_str.as_bytes())
    var file_size_kb = Float64(file_size) / 1024.0

    print("File:", path)
    print("Size:", file_size, "bytes (", file_size_kb, "KB )")
    print()

    var bench = Bench(BenchConfig(max_iters=100))

    @parameter
    @always_inline
    def bench_scalar(mut b: Bencher) raises capturing:
        @parameter
        @always_inline
        def call_fn() raises:
            # Scalar stage 1 + tape-emitting stage 2.
            var v = parse_cpu_native_tape[force_scalar=True](json_str.copy())
            _ = v.is_object()

        b.iter[call_fn]()

    @parameter
    @always_inline
    def bench_simd(mut b: Bencher) raises capturing:
        @parameter
        @always_inline
        def call_fn() raises:
            # SIMD stage 1 + tape-emitting stage 2 (default).
            var v = parse_cpu_native_tape(json_str.copy())
            _ = v.is_object()

        b.iter[call_fn]()

    @parameter
    @always_inline
    def bench_scalar_traverse(mut b: Bencher) raises capturing:
        @parameter
        @always_inline
        def call_fn() raises:
            var v = parse_cpu_native_tape[force_scalar=True](json_str.copy())
            _ = _walk(v)

        b.iter[call_fn]()

    @parameter
    @always_inline
    def bench_simd_traverse(mut b: Bencher) raises capturing:
        @parameter
        @always_inline
        def call_fn() raises:
            var v = parse_cpu_native_tape(json_str.copy())
            _ = _walk(v)

        b.iter[call_fn]()

    # `_walk` swallows access errors so we can still time the
    # traversal even when a backend has a correctness bug under deep
    # recursion. As a separate signal print whether the simd path
    # makes it through `citm_catalog`-style payloads cleanly, so the
    # bench reader can correlate raw throughput with correctness. We
    # only check this once at startup; the bench loop itself uses the
    # error-swallowing walk.
    var diag_v = parse_cpu_native_tape(json_str.copy())
    var diag_ok = _walk_strict(diag_v)
    print("traversal correctness:")
    if diag_ok:
        print("  tape (simd):  OK")
    else:
        print("  tape (simd):  FAILED -- access error during walk")
    print()

    var measures = List[ThroughputMeasure]()
    measures.append(ThroughputMeasure(BenchMetric.bytes, file_size))
    bench.bench_function[bench_scalar](BenchId("json_cpu", "scalar"), measures)
    bench.bench_function[bench_simd](BenchId("json_cpu", "simd"), measures)
    bench.bench_function[bench_scalar_traverse](
        BenchId("json_cpu", "scalar_traverse"), measures
    )
    bench.bench_function[bench_simd_traverse](
        BenchId("json_cpu", "simd_traverse"), measures
    )

    print(bench)
