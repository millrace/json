"""High-performance JSON library for Mojo.

- **Python-like API:** `loads`, `dumps`, `load`, `dump`
- **Reflection serde:** zero-boilerplate struct ser/de via compile-time reflection
- **GPU accelerated:** 2-4x faster than cuJSON on large files (NVIDIA, AMD)
- **CPU fast path:** pure-Mojo two-pass parser, ~1.4 GB/s, zero FFI
- **JSONPath, Schema, Patch:** RFC 6901, RFC 6902, RFC 7396

See [docs/api.md](docs/api.md) for the full API reference and usage
examples. The block comment below is the short orientation; for
anything beyond five lines of code please go to the API doc.

```mojo
from json import loads, dumps, load, dump

var data = loads('{"name": "Alice"}')         # default fast Mojo CPU
var fast = loads[target="cpu-simdjson"](s)    # simdjson FFI
var big  = load[target="gpu"]("huge.json")    # GPU (non-Apple)

print(dumps(data, indent="  "))
```

v0.2 changes worth knowing:
- `Value` is a view over a tape-backed `Document`; mutation is
  copy-on-write, so nested `set` / `append` propagates correctly.
- The default CPU parser is the two-pass stage 1 + stage 2 walker
  (`json.cpu.parse_cpu_native_tape`); `parse_mojo` / `parse_simd` and
  the `MojoJSONParser` / `FastParser` types are gone.
- `loads[target='gpu']` on Apple Silicon raises by default; recompile
  with `-D JSON_GPU_ALLOW_APPLE_FALLBACK=1` to keep the v0.1 silent
  CPU fallback.
- Typed NDJSON file load: `load[format='ndjson'](path) -> List[Value]`.
- Configured parsing / serialisation use the two-arg overloads:
  `loads(s, config)` and `dumps(v, config)`.
"""

# Core API.
from .parser import loads, load
from .serialize import dumps, dump
from .config import ParserConfig, SerializerConfig

# Value type.
from .value import Value, Null

# Manual ser/de traits.
from .serialize import to_json_value, to_json_string, Serializable, serialize
from .deserialize import (
    get_string,
    get_int,
    get_bool,
    get_float,
    Deserializable,
    deserialize,
)

# Reflection serde (zero boilerplate).
from .reflection import (
    serialize_json,
    serialize_value,
    deserialize_json,
    deserialize_value,
    try_deserialize_json,
    JsonSerializable,
    JsonDeserializable,
)

# RFCs and queries.
from .patch import apply_patch, merge_patch, create_merge_patch
from .jsonpath import jsonpath_query, jsonpath_one
from .schema import validate, is_valid, ValidationResult, ValidationError

# Streaming/lazy parsing (CPU only).
from .lazy import LazyValue
from .streaming import StreamingParser, ArrayStreamingParser
