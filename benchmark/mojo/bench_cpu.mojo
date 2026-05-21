# Mojo CPU benchmark for the json library.
#
# Reports throughput for both stage 1 paths side by side:
#
#   - json_cpu/scalar : the byte-by-byte structural index
#   - json_cpu/simd   : the 32-byte `pack_bits`-based structural index
#
# Same stage 2 walker for both, so the delta is purely stage 1.
#
# Usage:
#   mojo -I . benchmark/mojo/bench_cpu.mojo [json_file]
#
# Default file: benchmark/simdjson/jsonexamples/twitter.json

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
from json.cpu.stage2 import parse_two_pass


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
            var v = parse_two_pass[force_scalar=True](json_str)
            _ = v.is_object()

        b.iter[call_fn]()

    @parameter
    @always_inline
    def bench_simd(mut b: Bencher) raises capturing:
        @parameter
        @always_inline
        def call_fn() raises:
            var v = parse_two_pass[force_scalar=False](json_str)
            _ = v.is_object()

        b.iter[call_fn]()

    var measures = List[ThroughputMeasure]()
    measures.append(ThroughputMeasure(BenchMetric.bytes, file_size))
    bench.bench_function[bench_scalar](BenchId("json_cpu", "scalar"), measures)
    bench.bench_function[bench_simd](BenchId("json_cpu", "simd"), measures)

    print(bench)
