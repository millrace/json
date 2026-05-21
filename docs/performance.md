# Performance Deep Dive

This document explains why json is faster than existing parsers and the key optimizations that make it possible.

## GPU: 2x Faster than cuJSON

On NVIDIA B200 with 804MB `twitter_large_record.json`:

| Parser | Throughput | Time | Speedup |
|--------|------------|------|---------|
| cuJSON (CUDA C++) | 3.6 GB/s | 236 ms | baseline |
| **json GPU** | **7.0 GB/s** | **121 ms** | **2.0x** |

*Based on warmed-up runs. Pinned memory path (comparable scope to cuJSON).*

## Key Optimizations

| Optimization | Impact | Description |
|--------------|--------|-------------|
| **GPU Stream Compaction** | 🔥 **Main speedup** | Reduces D2H transfer from ~160ms to minimal overhead |
| **Pinned Memory** | H2D: ~15ms | Uses `HostBuffer` for fast host-to-device transfer |
| **Hierarchical Prefix Sums** | GPU: efficient | Parallel scans using block primitives |
| **Fused Kernels** | Lower overhead | Single-pass quote detection + structural bitmap |

## Why json is Faster: The Stream Compaction Advantage

### The Problem with cuJSON

cuJSON transfers **all structural character data** back to CPU:

- Input: 804MB JSON file
- Structural chars: ~58% of input = **465MB transfer**
- D2H time: **~160ms** (bottleneck)

### json's Solution

json uses **GPU stream compaction** to extract only position indices:

- Input: 804MB JSON file
- Position array: ~1 million positions × 4 bytes = **4MB transfer**
- D2H time: **minimal** (116x smaller data transfer)

This is the primary reason for the 2x overall speedup.

## Detailed Timing Breakdown

### cuJSON Pipeline (~236ms total)

```
cuJSON breakdown (average):
├─ H2D transfer:       ~15 ms   (804MB → GPU)
├─ Validation:          ~2 ms   (GPU)
├─ Tokenization:        ~6 ms   (GPU)
├─ Parser:              ~2 ms   (GPU)
└─ D2H transfer:      ~160 ms   (465MB → CPU, bottleneck)
────────────────────────────────
TOTAL:                ~236 ms
Throughput:           3.6 GB/s
```

### json GPU Pipeline (~121ms total)

```
json pinned breakdown (average):
├─ H2D transfer:       ~15 ms   (804MB → GPU, pinned memory)
├─ GPU kernels:        ~30 ms   (quote detection + prefix sums + bitmap)
├─ Stream compact:     ~50 ms   (GPU position extraction)
├─ D2H transfer:       ~15 ms   (4MB positions → CPU)
└─ Bracket matching:   ~11 ms   (CPU)
────────────────────────────────
TOTAL:                ~121 ms
Throughput:           7.0 GB/s
```

## Architecture Comparison

| Aspect | cuJSON | json |
|--------|--------|--------|
| **Input memory** | Pinned (cudaMallocHost) | Pinned (HostBuffer) |
| **H2D transfer** | ✓ (15ms) | ✓ (15ms) |
| **GPU kernels** | Validation + Tokenization | Quote detection + Prefix sums + Bitmap |
| **Position extraction** | ❌ (transfers all data) | ✅ **GPU stream compaction** |
| **D2H transfer** | 465MB (~160ms) | 4MB (~15ms) |
| **Bracket matching** | GPU (Parser kernel) | CPU (stack algorithm) |

## Performance Metrics Explained

`pixi run bench-gpu` reports a single `std.benchmark.Bench` table with
four rows so you can see where time goes across the pipeline:

| Row | What It Includes | Use Case |
|-----|------------------|----------|
| **from host bytes: memcpy + parse (wall-clock)** | host→pinned memcpy + `parse_json_gpu_from_pinned` | Realistic "bytes in memory → parsed" cost |
| **parse_json_gpu_from_pinned (pinned, wall-clock)** | H2D + GPU kernels + stream compaction + D2H + CPU bracket matching | Apples-to-apples comparison with cuJSON (both assume pinned input) |
| **parse_json_gpu_from_pinned (device-only)** | Same call, timed via `DeviceContext.execution_time` (CUDA events) | Pure device-queue time, excludes host-side CPU post-processing |
| **loads[target='gpu']** | Everything + `Value` tree construction on CPU | Real-world application performance |

### Why Four Rows?

1. **Pinned wall-clock (~121 ms, 7.0 GB/s):** apples-to-apples with cuJSON
   (both assume pinned input). This is the headline GPU-parse number.
2. **Pinned device-only (~100 ms, ~8 GB/s):** drops the host-side
   bracket-matching and list-build work. Use this to compare against
   kernel-only timings from other frameworks.
3. **from host bytes (~280 ms, ~2.9 GB/s):** adds the realistic
   host→pinned memcpy (~120 ms for 804 MB on DDR5).
4. **Full `loads[target='gpu']` (~900 ms, ~1.0 GB/s):** adds the
   CPU-bound `Value` tree construction on top of everything.

Pass `--debug-timing` to get a per-phase breakdown (H2D, GPU kernels,
position extraction, bracket matching, total) printed alongside the
summary table:

```bash
pixi run bench-gpu -- --debug-timing benchmark/datasets/twitter_large_record.json
```

## Benchmark Results

### GPU Performance (NVIDIA B200)

**Important:** GPU benchmarks are only meaningful for large files (>100MB). For smaller files, GPU launch overhead dominates and results are not representative of actual performance.

| Dataset | Size | Pinned Path | Speedup vs cuJSON |
|---------|------|-------------|-------------------|
| twitter_large_record.json | 804 MB | 7.0 GB/s | **2.0x** |

GPU parallelism shines with large files where the overhead is amortized.

## CPU Performance

json has two CPU code paths, both served by `loads(target='cpu')`:

| Path | DOM model | What it costs |
|---|---|---|
| **tape** (default) | Eager `Document` tape: every primitive becomes a typed tape entry at parse time. SIMD stage 1 by default; opt into the scalar oracle with `parse_cpu_native_tape[force_scalar=True]` | A bit slower to parse than the legacy lazy approach we shipped historically; dramatically faster and correct on traversal of real-world JSON |
| simdjson (FFI) | C++ simdjson via the `target='cpu-simdjson'` shim. The FFI output is translated into the same tape representation | Reference parser at the cost of FFI marshalling |

### Two workloads, two answers

`pixi run -e dev bench-cpu <file>` reports both:

* `scalar` / `simd` -- "parse, then peek the root with
  `is_object()`". Children are not touched, so this measures
  the parse cost in isolation.
* `scalar_traverse` / `simd_traverse` -- "parse, then recursively
  visit every value via the public API
  (`array_items` / `object_items` / `*_value`)". This is what almost
  any real consumer ends up doing.

#### Historical comparison (Apple Silicon, M-series, this dev box)

The numbers below were collected while the legacy v0.1 lazy `Value`
representation still shipped as a `-D JSON_USE_LAZY_VALUE=1`
fallback. They are kept here as evidence for why the tape path is
the only path the library now ships:

| Corpus | Size | simdjson C++ | simd lazy (peek) | tape (peek) | simd lazy (traverse) | **tape (traverse)** |
|---|---|---|---|---|---|---|
| `twitter.json` | 617 KB | 2.66 GB/s | 1.18 GB/s | 0.23 GB/s | 142.9 ms | **4.17 ms (34x faster)** |
| `citm_catalog.json` | 1.7 MB | 3.13 GB/s | 1.33 GB/s | 0.23 GB/s | **701 ms ❌ buggy** | **11.38 ms ✅ correct (62x faster)** |
| `twitter_large_record.json` | 804 MB | 1.47 GB/s | 0.73 GB/s | 0.15 GB/s | (not run; quadratic-ish) | (eager parse only -- bench measures `peek`) |

The `citm_catalog` row was the punchline: the legacy lazy path
raised `Key not found in JSON object` mid-walk because
`object_items()` re-scanned the raw substring for each key it
remembered, and that second scan could disagree with the first on
documents with duplicate keys or non-trivial escape patterns. **The
tape path is the only one that walks `citm_catalog` correctly**, and
it does so 62x faster than the (buggy) lazy walk on the same input.

### Why tape is the only CPU path

The lazy v0.1 representation was fast for "parse and ignore", which
is why it shipped first. But its on-access re-parse model was a
documented silent-bug source: nested mutations didn't propagate
through `Value` views, duplicate keys collapsed, and traversal cost
was super-linear because every `object_items()` rescanned the raw
substring. Every fix to those made the lazy path slower without
making it correct, so the lazy `Value` representation was removed
in favor of the tape:

* Each value gets exactly one tape entry. No re-parsing, no
  re-scanning, no key-collision lottery.
* Strings are stored as `(offset, length)` slices into the original
  input -- zero-copy when the bytes don't need unescaping.
* The GPU pipeline emits the same tape, so CPU and GPU now agree on
  one DOM representation.
* Because every `Value` is a stable index into the same tape,
  copy-on-write mutation can correctly propagate through nested
  containers (Phase 2d).

In other words: tape pays for correctness and post-parse traversal
speed at parse time. That's the right trade for almost every real
consumer. The legacy lazy path stays gated for now to unblock callers
that haven't migrated, and will be removed entirely in a follow-up
once the v0.1 `_raw` fields can come out of `Value`.

## When to Use GPU vs CPU

| File Size | Recommended Backend | Reason |
|-----------|---------------------|--------|
| < 1 MB | **CPU (simdjson)** | GPU launch overhead dominates |
| 1-100 MB | **CPU or GPU** | Comparable performance |
| > 100 MB | **GPU** | 2x faster than cuJSON, 3-5x faster than CPU |

## Optimization Techniques

### 1. GPU Stream Compaction

**Problem:** After identifying structural characters on GPU, we need their positions on CPU for bracket matching.

**Naive approach:** Transfer entire structural character bitmap (58% of input size)

**Optimized approach:**
1. Create position bitmap on GPU
2. Use parallel prefix sum to compute output positions
3. Compact positions into dense array on GPU
4. Transfer only compact position array to CPU

**Result:** 116x reduction in D2H transfer size (465MB → 4MB)

### 2. Pinned Memory

Using `HostBuffer` (pinned memory) for H2D transfers:

- Pinned: ~15ms for 804MB
- Pageable: ~110ms for 804MB
- **Speedup:** 7.3x faster

### 3. Hierarchical Prefix Sums

For computing in-string regions, we use block-level prefix sums:

1. Each block computes local prefix sum using `block.prefix_sum`
2. Last value from each block propagates to next block
3. Single-pass algorithm, minimal synchronization

### 4. Fused Kernels

Combine multiple operations in single kernel launches:

- Quote detection + escape handling
- Structural character extraction + bitmap creation
- Reduces kernel launch overhead

### 5. Minimize Memory Allocations

- Pre-allocate GPU buffers based on input size
- Reuse `DeviceContext` across operations
- Use `String(unsafe_from_utf8=bytes^)` for bulk string construction

### 6. Hybrid GPU/CPU Pipeline

- **GPU:** Parallel bitmap operations (where GPU excels)
- **CPU:** Sequential bracket matching (where CPU is sufficient)
- **Key insight:** Don't force everything on GPU; use the right tool for each step

## Performance Variance

GPU performance can vary between runs due to:

- **Cold-start overhead:** First GPU run ~200ms slower (GPU initialization)
- **Thermal throttling:** GPU frequency varies with temperature
- **Scheduling:** CUDA stream scheduling can introduce variance

**Solution:** Always measure with warm-up runs and report averages.

## Future Optimizations

Potential improvements for even better performance:

1. **GPU bracket matching:** Could eliminate CPU bottleneck (~11ms)
2. **Multi-GPU support:** For files > 1GB
3. **Streaming parser:** Process chunks as they arrive
4. **Zero-copy Value tree:** Build tree directly on GPU memory

## Benchmark Reproducibility

All benchmarks are reproducible using pinned git submodules:

```bash
# Clone the repo
git clone https://github.com/ehsanmok/json.git && cd json

# Clone cuJSON (optional, for the head-to-head benchmark)
cd benchmark && git clone https://github.com/AutomataLab/cuJSON.git && cd ..

# Build comparison benchmark (lives in the dev feature)
pixi run -e dev build-cujson

# Run benchmarks
pixi run bench-gpu-cujson benchmark/datasets/twitter_large_record.json
```

See [benchmark/readme.md](../benchmark/readme.md) for complete setup instructions.

## Hardware Requirements

- **GPU:** NVIDIA GPU with CUDA support (tested on B200, H100, A100) or Apple Silicon
- **CUDA:** Latest CUDA toolkit (for NVIDIA)
- **Memory:** At least 2x your largest JSON file size (for GPU buffers)

## References

- [simdjson](https://github.com/simdjson/simdjson) - CPU JSON parser
- [cuJSON](https://github.com/AutomataLab/cuJSON) - GPU JSON parser (baseline comparison)
- [GPU stream compaction](https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back) - Decoupled look-back algorithm
