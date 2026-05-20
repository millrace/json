# json

[![CI](https://github.com/ehsanmok/json/actions/workflows/ci.yml/badge.svg)](https://github.com/ehsanmok/json/actions/workflows/ci.yml)
[![Docs](https://github.com/ehsanmok/json/actions/workflows/docs.yaml/badge.svg)](https://ehsanmok.github.io/json/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-performance JSON library for Mojo with GPU acceleration.

- **Python-like API:** `loads`, `dumps`, `load`, `dump`
- **Reflection serde:** zero-boilerplate struct serialization via compile-time reflection
- **GPU accelerated:** 2-4x faster than [cuJSON](https://github.com/AutomataLab/cuJSON) on large files (NVIDIA, AMD)
- **Tape-backed Value:** v0.2 stores parsed JSON as a packed tape inside a `Document`; `Value` is a view; nested mutation propagates correctly
- **Two-pass CPU parser:** stage 1 builds a structural index (scalar oracle + SIMD), stage 2 walks it; ~1.4 GB/s, zero FFI
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

```bash
# Download large dataset first (required for meaningful GPU benchmarks).
# `download-*` lives in the dev feature because it needs gdown, so pass -e dev.
pixi run -e dev download-twitter-large

# Run GPU benchmark (only use large files)
pixi run bench-gpu benchmark/datasets/twitter_large_record.json
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

Tasks that need extra tooling (docs, formatters, dataset downloaders)
live in the `dev` feature. Pass `-e dev` or enter a dev shell:

```bash
pixi run -e dev format       # mojo format + pre-commit hook install
pixi run -e dev format-check # used by CI
pixi run -e dev docs         # mojodoc + open in browser
pixi run -e dev docs-build   # build into target/doc (used by Pages workflow)
pixi run -e dev download-twitter-large  # gdown cuJSON benchmark datasets
```

`pixi run <task>` without `-e` auto-routes dev-only tasks to the dev env
when the task is not defined in the default env. End users consuming the
library as a package never need the `dev` feature.

Further documentation:

- [Architecture](./docs/architecture.md): CPU/GPU backend design
- [Performance](./docs/performance.md): optimization deep dive
- [Benchmarks](./benchmark/README.md): reproducible benchmarks

## License

[MIT](./LICENSE)
