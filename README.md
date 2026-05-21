# json

[![CI](https://github.com/ehsanmok/json/actions/workflows/ci.yml/badge.svg)](https://github.com/ehsanmok/json/actions/workflows/ci.yml)
[![Docs](https://github.com/ehsanmok/json/actions/workflows/docs.yaml/badge.svg)](https://ehsanmok.github.io/json/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-performance JSON library for Mojo with GPU acceleration.

- **Python-like API:** `loads`, `dumps`, `load`, `dump`
- **Reflection serde:** zero-boilerplate struct serialization via compile-time reflection
- **GPU accelerated:** 2-4x faster than [cuJSON](https://github.com/AutomataLab/cuJSON) on large files (NVIDIA, AMD)
- **Tape-backed Value:** v0.2 stores parsed JSON as a packed tape inside a `Document`; `Value` is a view; nested mutation propagates correctly
- **Two-pass CPU parser:** stage 1 builds a structural index (scalar oracle + SIMD), stage 2 walks it; **~1.2 GB/s on `twitter.json` / 0.7 GB/s on the 804 MB record-shaped corpus** (Apple Silicon, M-series), zero FFI
- **Streaming and lazy parsing:** handle files larger than memory
- **JSONPath and Schema:** query and validate JSON documents
- **RFC compliant:** JSON Patch, Merge Patch, JSON Pointer

## Quick Start

```mojo
from json import loads, dumps, load, dump

# Parse & serialize strings
var data = loads('{"name": "Alice", "scores": [95, 87, 92]}')
print(data["name"].string_value())  # Alice
print(data["scores"][0].int_value())  # 95
print(dumps(data, indent="  "))  # Pretty print

# File I/O
var config = load("config.json")
var logs = load[format="ndjson"]("events.ndjson")  # Returns List[Value]

# Explicit GPU parsing (NVIDIA / AMD; Apple Silicon raises by default)
var big = load[target="gpu"]("large.json")
```

### One-line prelude

For everyday code, `json.prelude` re-exports the names you reach for
all the time -- `loads`, `dumps`, `load`, `dump`, `Value`, `Null`,
`ParserConfig`, `SerializerConfig`, and the reflection serde
shortcuts. The canonical import block collapses to one line:

```mojo
from json.prelude import *

@fieldwise_init
struct Person(Defaultable, Movable):
    var name: String
    var age:  Int
    def __init__(out self):
        self.name = ""
        self.age = 0

def main() raises:
    var p = deserialize_json[Person]('{"name":"Alice","age":30}')
    print(p.name, p.age)
    print(dumps(serialize_value(p), indent="  "))
```

Domain-specific surfaces (jsonpath / patch / schema / lazy / streaming
/ manual serde traits / simdjson FFI) are intentionally **not** in the
prelude -- import those from the matching module so the import block
of a real file still documents which features it actually uses. See
[`json/prelude.mojo`](./json/prelude.mojo) for the exact list.

## Installation

Add json to your project's `pixi.toml`:

```toml
[workspace]
channels = ["https://conda.modular.com/max-nightly", "conda-forge"]
preview = ["pixi-build"]

[dependencies]
json = { git = "https://github.com/ehsanmok/json.git", tag = "v0.2.0" }
```

Then run:

```bash
pixi install
```

Requires [pixi](https://pixi.sh) (pulls Mojo nightly automatically).

For the latest development version:

```toml
[dependencies]
json = { git = "https://github.com/ehsanmok/json.git", branch = "main" }
```

> **Note:** `mojo-compiler` and `simdjson` are automatically installed as dependencies.

**GPU (optional):** NVIDIA CUDA 7.0+, AMD ROCm 6+, or Apple Silicon. See [GPU requirements](https://docs.modular.com/max/packages#gpu-compatibility).

## Performance

### GPU (804MB `twitter_large_record.json`)

| Platform | Throughput | vs cuJSON |
|----------|------------|-----------|
| AMD MI355X | 13 GB/s | **3.6x faster** |
| NVIDIA B200 | 8 GB/s | **1.8x faster** |
| Apple M3 Pro | 3.9 GB/s | N/A |

*GPU only beneficial for files >100MB.*

### CPU (Apple Silicon, M-series)

`pixi run bench-cpu <file>` runs simdjson C++ first, then the three
Mojo CPU paths under both `parse + peek` and `parse + traverse-every-value`
workloads.

**Parse + peek** (lazy paths short-circuit; tape pays full materialisation cost):

| File | Size | simdjson C++ | simd (lazy) | scalar (lazy) | tape (eager) |
|---|---|---|---|---|---|
| `twitter.json` | 617 KB | 2.66 GB/s | **1.18 GB/s** | 0.60 GB/s | 0.23 GB/s |
| `citm_catalog.json` | 1.7 MB | 3.13 GB/s | **1.33 GB/s** | 0.62 GB/s | 0.23 GB/s |
| `twitter_large_record.json` | 804 MB | 1.47 GB/s | **0.73 GB/s** | 0.51 GB/s | 0.15 GB/s |

**Parse + traverse every value** (the realistic workload):

| File | simd_traverse (lazy) | **tape_traverse (eager)** |
|---|---|---|
| `twitter.json` | 142.9 ms | **4.17 ms (34x faster)** |
| `citm_catalog.json` | **701 ms, but buggy ❌** | **11.38 ms, correct ✅ (62x faster)** |

The lazy path raises `Key not found in JSON object` mid-walk on
`citm_catalog` -- `object_items()` re-scans the raw substring for
every remembered key, and the second scan can disagree with the first
on documents with duplicate keys or non-trivial escapes. **Tape is
the only path that walks `citm_catalog` correctly**, and it's 30-60×
faster than the (buggy) lazy walk on the same input.

The lazy paths still beat tape on `parse + peek` because they don't
decode children -- they store array/object bodies as raw substrings
and rescan on access. That makes them cheap when you ignore the
parsed result and pathologically slow / incorrect when you don't.
**As of v0.2, `loads(target='cpu')` routes through the tape path by
default** -- typed tape entries, zero-copy strings, one DOM
representation shared with the GPU pipeline, and correct nested
mutation. The legacy lazy path is still in tree behind
`-D JSON_USE_LAZY_VALUE=1` so callers that haven't migrated stay
unblocked, but it's documented to be slower and to disagree with
itself on documents like `citm_catalog`.

```bash
# Download large dataset first (required for meaningful GPU benchmarks).
# `download-*` lives in the dev feature because it needs gdown, so pass -e dev.
pixi run -e dev download-twitter-large

# Run GPU benchmark (only use large files)
pixi run bench-gpu benchmark/datasets/twitter_large_record.json

# CPU bench: simdjson C++ vs Mojo simd / scalar / tape
pixi run -e dev bench-cpu                                            # twitter.json
pixi run -e dev bench-cpu benchmark/datasets/citm_catalog.json
pixi run -e dev bench-cpu benchmark/datasets/twitter_large_record.json
```

## Reflection-Based Serde (Zero Boilerplate)

Automatically serialize and deserialize structs using compile-time reflection. No hand-written `to_json()` or `from_json()` methods needed.

```mojo
from json import serialize_json, deserialize_json

@fieldwise_init
struct Person(Defaultable, Movable):
    var name: String
    var age: Int
    var active: Bool
    def __init__(out self):
        self.name = ""
        self.age = 0
        self.active = False

# Serialize: one function, zero boilerplate
var json = serialize_json(Person(name="Alice", age=30, active=True))
# {"name":"Alice","age":30,"active":true}

# Deserialize: just specify the type
var person = deserialize_json[Person](json)
print(person.name)  # Alice

# Pretty print
print(serialize_json[pretty=True](person))

# GPU-accelerated parsing, CPU struct extraction
var fast = deserialize_json[Person, target="gpu"](json)

# Non-raising variant (returns Optional)
from json import try_deserialize_json
var maybe = try_deserialize_json[Person]('{"bad json')  # None
```

### Supported Field Types

| Category | Types |
|----------|-------|
| Scalars | `Int`, `Int64`, `Bool`, `Float64`, `Float32`, `String` |
| Containers | `List[T]`, `Optional[T]` (where T is a scalar) |
| Nested | Any struct that is `Defaultable & Movable` |
| Raw JSON | `Value` (pass-through, no conversion) |

### Custom Serialization

Override reflection behavior for specific types:

```mojo
from json import JsonSerializable, JsonDeserializable

struct Color(JsonSerializable, Defaultable, Movable):
    var r: Int
    var g: Int
    var b: Int

    def to_json_value(self) raises -> Value:
        """Serialize as "rgb(r,g,b)" instead of {"r":...,"g":...,"b":...}."""
        ...

struct RGBArray(JsonDeserializable, Defaultable, Movable):
    var r: Int
    var g: Int
    var b: Int

    @staticmethod
    def from_json_value(json: Value) raises -> Self:
        """Deserialize from JSON array [r, g, b] instead of object."""
        ...
```

Full API reference: [ehsanmok.github.io/json](https://ehsanmok.github.io/json/)

## Examples

Examples are organised by progression -- start in `basic/`, move to
`intermediate/` when you need typed serde / JSONPath / Schema / Patch,
and dip into `advanced/` for lazy and GPU paths. See
[examples/README.md](./examples/README.md) for the full guided tour.

```bash
pixi run examples                                  # run every tier
pixi run example-parsing                           # one example by name
pixi run mojo -I . examples/basic/parsing.mojo     # or invoke mojo directly
```

### basic/ -- first ten lines of code

| Example | Description |
|---------|-------------|
| [basic/parsing](./examples/basic/parsing.mojo)               | Parse, serialize, type handling |
| [basic/file_io](./examples/basic/file_io.mojo)               | Read/write JSON files |
| [basic/value_types](./examples/basic/value_types.mojo)       | Type checking, value extraction |
| [basic/error_handling](./examples/basic/error_handling.mojo) | `try` / `except` patterns and recovery |

### intermediate/ -- real-world feature usage

| Example | Description |
|---------|-------------|
| [intermediate/reflection_serde](./examples/intermediate/reflection_serde.mojo) | Zero-boilerplate struct serde via reflection |
| [intermediate/struct_serde](./examples/intermediate/struct_serde.mojo)         | Manual `Serializable` / `Deserializable` traits |
| [intermediate/ndjson](./examples/intermediate/ndjson.mojo)                     | NDJSON parsing & streaming |
| [intermediate/jsonpath](./examples/intermediate/jsonpath.mojo)                 | RFC 9535 JSONPath queries |
| [intermediate/schema_validation](./examples/intermediate/schema_validation.mojo) | JSON Schema validation |
| [intermediate/json_patch](./examples/intermediate/json_patch.mojo)             | JSON Patch (RFC 6902) & Merge Patch (RFC 7396) |

### advanced/ -- perf-focused

| Example | Description |
|---------|-------------|
| [advanced/lazy_parsing](./examples/advanced/lazy_parsing.mojo) | On-demand lazy parsing for large documents |
| [advanced/gpu_parsing](./examples/advanced/gpu_parsing.mojo)   | GPU-accelerated parsing (Apple-Silicon-aware) |

## Development

```bash
git clone https://github.com/ehsanmok/json.git && cd json
pixi install                 # lean default env (mojo, simdjson, gxx, sysroot)
pixi run tests-cpu           # or: tests-gpu / tests-e2e / tests-e2e-gpu
pixi run bench-gpu           # optional; builds and runs the GPU benchmark
```

### Pixi environments

The project layers three environments; pick the one that matches the
job you're doing.

| Env       | What's in it                                                | When to use it                                                                |
|-----------|-------------------------------------------------------------|-------------------------------------------------------------------------------|
| `default` | mojo, simdjson, gxx, sysroot (Linux)                        | Run the library, run tests, run examples. End users stop here.                |
| `dev`     | `default` + python, gdown, pre-commit, mojodoc, sysroot pin | Format, build docs, build benchmark binaries, AOT sanitizer builds, datasets. |
| `fuzz`    | `dev` + mozz                                                | Run mozz fuzz harnesses against the FFI / parser / access surfaces.           |

Switch envs by passing `-e <name>` to `pixi run`, or enter a shell with
`pixi shell -e <name>`. Tasks defined only in a non-default env auto-route
when there is no ambiguity, but explicit `-e` is always safe.

### Common dev tasks

```bash
pixi run -e dev format          # mojo format + pre-commit hook install
pixi run -e dev format-check    # used by CI
pixi run -e dev docs            # mojodoc + open in browser
pixi run -e dev docs-build      # build into target/doc (used by Pages workflow)
pixi run -e dev download-twitter-large  # gdown cuJSON benchmark datasets
```

### Sanitizer testing (ASan)

A curated, FFI- and lifetime-heavy slice of the test suite is
exercised under LLVM AddressSanitizer to catch use-after-free,
out-of-bounds, and lifetime regressions in the simdjson FFI shim, the
tape-backed `Document` / `Value` view, and the COW mutation path.
Driven by [`tools/run_sanitizer_tests.sh`](./tools/run_sanitizer_tests.sh).

```bash
pixi run -e dev tests-asan      # AOT-build with --sanitize address, run, fail-fast
pixi run -e dev tests-asan tests/test_value.mojo   # one file
```

> **macOS note.** Mojo's bundled libasan expects
> `__asan_version_mismatch_check_v8`; the system clang on Apple
> Silicon ships `__asan_version_mismatch_check_apple_clang_1700`. The
> harness detects this and skips cleanly with a clear message, so a
> dev-box `pixi run tests-asan` doesn't fail. Linux CI runs the full
> harness end-to-end.

### Fuzz testing (mozz)

Mozz harnesses live in [`fuzz/`](./fuzz/) and exercise:

- `fuzz-loads` -- the default Mojo two-pass parser, with a
  parse -> dumps -> parse idempotence property.
- `fuzz-simdjson` -- the simdjson FFI boundary
  (`OwnedDLHandle` + `external_call`), with a differential property
  that the simdjson and Mojo native parsers must agree on canonical
  `dumps` output when both succeed.
- `fuzz-value-access` -- the `Value` access API + COW mutation.
- `fuzz-jsonpath` -- the RFC 9535 JSONPath engine.
- `fuzz-ndjson` -- the `loads[format='ndjson']` line splitter.

```bash
pixi run -e fuzz fuzz-simdjson  # 200k iters; locks crash repros under .mozz_crashes/
pixi run -e fuzz fuzz-all       # run every harness back-to-back
```

The `fuzz/corpus/<harness>/` directory is gitignored and populated at
runtime; copy interesting crashes there manually to lock in
regressions.

### GPU on Apple Silicon

Mojo's Metal backend currently rejects the raw-pointer kernels the
GPU pipeline relies on (`Metal Compiler failed to compile metallib`).
The library handles this honestly: `loads[target='gpu']` raises by
default on Apple Silicon, and recompiling with
`-D JSON_GPU_ALLOW_APPLE_FALLBACK=1` opts back into the silent CPU
fallback. The pixi tasks set this flag for the cases where it makes
sense:

| Task                  | Apple Silicon behaviour                                                         |
|-----------------------|---------------------------------------------------------------------------------|
| `tests-gpu`           | Builds + runs (21 / 21 pass under the fallback flag)                            |
| `tests-e2e-gpu`       | Builds + runs (38 / 38 pass under the fallback flag)                            |
| `example-gpu`         | Runs end-to-end under the fallback flag                                         |
| `bench-gpu`           | Skips with a clean "Metal lacks raw-pointer kernel support" message, exits 0    |

On Linux + NVIDIA / AMD the flag is harmless and the real GPU path is
taken before the fallback gate.

`pixi run <task>` without `-e` auto-routes dev / fuzz tasks to the
matching env when the task is unambiguous. End users consuming the
library as a package never need the `dev` or `fuzz` features.

Further documentation:

- [Architecture](./docs/architecture.md): CPU/GPU backend design
- [Performance](./docs/performance.md): optimization deep dive
- [Benchmarks](./benchmark/README.md): reproducible benchmarks
- [Examples guided tour](./examples/README.md): basic / intermediate / advanced

## License

[MIT](./LICENSE)
