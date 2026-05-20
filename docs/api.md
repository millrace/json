# json API reference

The full long-form API guide for the `json` library. The package
`__init__.mojo` keeps a short orientation; this file is the canonical
reference.

## Quick Start

```mojo
from json import loads, dumps, load, dump

var data = loads('{"name": "Alice", "scores": [95, 87, 92]}')
print(data["name"].string_value())  # Alice
print(data["scores"][0].int_value())  # 95
print(dumps(data, indent="  "))  # Pretty print

var config = load("config.json")
var logs = load[format="ndjson"]("events.ndjson")  # List[Value]

var big = load[target="gpu"]("large.json")
```

## `json.prelude` -- one-line import for everyday code

`json.prelude` re-exports the names a typical module reaches for, so
the standard import block becomes a single line:

```mojo
from json.prelude import *

var data = loads('{"name": "Alice"}')
print(dumps(data, indent="  "))
```

The exact set of re-exports is intentionally small:

| Symbol                                                                            | From                |
|-----------------------------------------------------------------------------------|---------------------|
| `loads`, `load`                                                                   | `json.parser`       |
| `dumps`, `dump`                                                                   | `json.serialize`    |
| `ParserConfig`, `SerializerConfig`                                                | `json.config`       |
| `Value`, `Null`                                                                   | `json.value`        |
| `serialize_json`, `serialize_value`, `deserialize_json`, `deserialize_value`,     |                     |
| `try_deserialize_json`, `JsonSerializable`, `JsonDeserializable`                  | `json.reflection`   |

Domain-specific surfaces are deliberately **not** re-exported -- the
prelude stays small enough that the import block of a real module
still documents which features it actually uses. Import directly:

| Surface                                                                              | Import from           |
|--------------------------------------------------------------------------------------|-----------------------|
| `jsonpath_query`, `jsonpath_one`                                                     | `json.jsonpath`       |
| `apply_patch`, `merge_patch`, `create_merge_patch`                                   | `json.patch`          |
| `validate`, `is_valid`, `ValidationResult`, `ValidationError`                        | `json.schema`         |
| `LazyValue`                                                                          | `json.lazy`           |
| `StreamingParser`, `ArrayStreamingParser`                                            | `json.streaming`      |
| `Serializable`, `to_json_value`, `to_json_string`                                    | `json.serialize`      |
| `Deserializable`, `get_int`, `get_string`, `get_bool`, `get_float`                   | `json.deserialize`    |
| `SimdjsonFFI`                                                                        | `json.cpu.simdjson_ffi` |

## loads() - Parse Strings

```mojo
from json import loads, ParserConfig

# Default: pure-Mojo two-pass parser.
var data = loads('{"name": "Alice", "age": 30}')

# simdjson FFI backend.
var data = loads[target="cpu-simdjson"]('{"name": "Alice"}')

# GPU acceleration (large files only).
var data = loads[target="gpu"](large_json_string)

# Parser configuration.
var config = ParserConfig(allow_comments=True, allow_trailing_comma=True)
var data = loads('{"a": 1,} // comment', config)

# NDJSON -> List[Value]
var values = loads[format="ndjson"]('{"a":1}\n{"a":2}\n{"a":3}')

# Lazy parsing (parse on demand).
var lazy = loads[lazy=True](huge_json_string)
var name = lazy.get("/users/0/name")
```

## dumps() - Serialize Strings

```mojo
from json import dumps, SerializerConfig

var json = dumps(data)
var pretty = dumps(data, indent="  ")

var config = SerializerConfig(escape_unicode=True, escape_forward_slash=True)
var json = dumps(data, config)

var ndjson = dumps[format="ndjson"](list_of_values)
```

## load() - Parse Files

```mojo
from json import load

var data = load("config.json")

# v0.2: typed NDJSON load -> List[Value]
var events = load[format="ndjson"]("events.ndjson")

# Backward-compatible auto-detection: returns an array Value.
var events_value = load("events.ndjson")

var big = load[target="gpu"]("large.json")

# Streaming for files larger than memory.
var parser = load[streaming=True]("huge.ndjson")
while parser.has_next():
    var item = parser.next()
    process(item)
parser.close()
```

## dump() - Write Files

```mojo
from json import dump

var f = open("output.json", "w")
dump(data, f)
f.close()

var f = open("output.json", "w")
dump(data, f, indent="  ")
f.close()
```

## Feature Matrix

| Feature                  | CPU      | GPU            | Notes                |
| ------------------------ | -------- | -------------- | -------------------- |
| `loads(s)`               | default  | `target="gpu"` |                      |
| `load(path)`             | default  | `target="gpu"` | Auto-detects .ndjson |
| `loads[format="ndjson"]` | default  | `target="gpu"` | Returns List[Value]  |
| `load[format="ndjson"]`  | default  | `target="gpu"` | Returns List[Value]  |
| `loads[lazy=True]`       | yes      | no             | CPU only             |
| `load[streaming=True]`   | yes      | no             | CPU only             |
| `dumps` / `dump`         | yes      | no             | CPU only             |

## Value Type

The `Value` struct is the in-memory representation of any JSON value.

### Type Checking

```mojo
v.is_null()    # true if null
v.is_bool()    # true if boolean
v.is_int()     # true if integer
v.is_float()   # true if float
v.is_string()  # true if string
v.is_array()   # true if array
v.is_object()  # true if object
v.is_number()  # true if int or float
```

### Value Extraction

```mojo
v.bool_value()    # -> Bool
v.int_value()     # -> Int64
v.float_value()   # -> Float64
v.string_value()  # -> String
v.raw_json()      # -> String (for arrays/objects)
```

### Access & Iteration

```mojo
var name = obj["name"]              # -> Value
var items = obj.object_items()      # -> List[Tuple[String, Value]]
var keys = obj.object_keys()        # -> List[String]

var first = arr[0]                  # -> Value
var items = arr.array_items()       # -> List[Value]
var count = arr.array_count()       # -> Int

var nested = data.at("/users/0/name")  # JSON Pointer (RFC 6901)
```

### Mutation

In v0.2 mutations route through `OwnedValue` and propagate through
nested access (so `doc["a"]["b"].set(...)` is now observed by the
parent), matching Python's `json.loads` + dict semantics.

```mojo
obj.set("key", Value("value"))
obj.set("count", Value(42))

arr.set(0, Value("new first"))
arr.append(Value("new item"))

# Nested mutation via JSON Pointer.
doc.set_at("/users/0/name", Value("Updated"))
```

### Creating Values

```mojo
from json import Value, Null

var null_val = Value(Null())
var bool_val = Value(True)
var int_val = Value(42)
var float_val = Value(3.14)
var str_val = Value("hello")
```

## Reflection-Based Serde

```mojo
from json import serialize_json, deserialize_json

@fieldwise_init
struct Person(Defaultable, Movable):
    var name: String
    var age: Int
    def __init__(out self):
        self.name = ""
        self.age = 0

var json = serialize_json(Person(name="Alice", age=30))
var person = deserialize_json[Person](json)

print(serialize_json[pretty=True](person))

var fast = deserialize_json[Person, target="gpu"](json)

from json import try_deserialize_json
var maybe = try_deserialize_json[Person]('bad json')  # None
```

Supported field types: `Int`, `Int64`, `Bool`, `Float64`, `Float32`,
`String`, `List[T]`, `Optional[T]`, nested structs, `Value` (raw JSON
pass-through), and the v0.2 additions: `Dict[String, T]`,
`List[Optional[T]]`, `Optional[List[T]]`, `List[List[T]]`.

### Custom Serialization Traits

```mojo
from json import JsonSerializable, JsonDeserializable

struct Color(JsonSerializable, Defaultable, Movable):
    var r: Int
    var g: Int
    var b: Int
    def to_json_value(self) raises -> Value:
        ...

struct RGBArray(JsonDeserializable, Defaultable, Movable):
    var r: Int
    var g: Int
    var b: Int
    @staticmethod
    def from_json_value(json: Value) raises -> Self:
        ...
```

### Reflection Serde API

| Function                                | Description                  |
| --------------------------------------- | ---------------------------- |
| `serialize_json(value)`                 | Struct -> JSON string        |
| `serialize_json[pretty=True](value)`    | Struct -> pretty JSON string |
| `serialize_value(value)`                | Struct -> Value object       |
| `deserialize_json[T](json_str)`         | JSON string -> struct T      |
| `deserialize_json[T, target="gpu"](s)`  | GPU parse -> struct T        |
| `deserialize_value[T](value)`           | Value object -> struct T     |
| `try_deserialize_json[T](json_str)`     | Non-raising, Optional[T]     |

## Manual Serializable / Deserializable

For full control over format:

```mojo
from json import Serializable, serialize, to_json_value

struct Person(Serializable):
    var name: String
    var age: Int

    def to_json(self) -> String:
        return '{"name":' + to_json_value(self.name) + ',"age":' + to_json_value(self.age) + '}'

var json = serialize(Person("Alice", 30))
```

```mojo
from json import Deserializable, deserialize, get_string, get_int

struct Person(Deserializable):
    var name: String
    var age: Int

    @staticmethod
    def from_json(json: Value) raises -> Self:
        return Self(
            name=get_string(json, "name"),
            age=get_int(json, "age"),
        )

var person = deserialize[Person]('{"name":"Alice","age":30}')
```

### Helper Functions

| Function                  | Description                       |
| ------------------------- | --------------------------------- |
| `to_json_value(s: String)`| Escape and quote string for JSON  |
| `to_json_value(i: Int)`   | Convert int to JSON               |
| `to_json_value(f: Float64)`| Convert float to JSON            |
| `to_json_value(b: Bool)`  | Convert bool to JSON              |
| `get_string(v, key)`      | Extract string field              |
| `get_int(v, key)`         | Extract int field                 |
| `get_float(v, key)`       | Extract float field               |
| `get_bool(v, key)`        | Extract bool field                |

## Error Handling

```mojo
try:
    var data = loads('{"invalid": }')
except e:
    print(e)
    # JSON parse error at line 1, column 13: ...
```

In v0.2, `loads[target='gpu']` raises an explicit error on Apple
Silicon (Metal backend lacks raw-pointer kernel support); recompile
with `-D JSON_GPU_ALLOW_APPLE_FALLBACK=1` to opt into the legacy
silent CPU fallback.

## GPU Parsing

Recommended for files >100MB. Works on NVIDIA (CUDA 7.0+), AMD
(ROCm 6+); Apple Silicon currently raises (see above).

```mojo
var data = loads[target="gpu"](large_json)
```

## NDJSON

```mojo
from json import loads, load, dumps

# Strings: format="ndjson" -> List[Value]
var values = loads[format="ndjson"]('{"a":1}\n{"a":2}\n{"a":3}')

# Files: typed overload -> List[Value]
var events = load[format="ndjson"]("events.ndjson")

# Serialize List[Value] -> NDJSON string
var ndjson = dumps[format="ndjson"](values)
```

## Lazy / On-Demand Parsing

```mojo
from json import loads

var lazy = loads[lazy=True](huge_json_string)
var name = lazy.get("/users/0/name")
var age = lazy.get_int("/users/0/age")
```

## Streaming

```mojo
from json import load

var parser = load[streaming=True]("logs.ndjson")
while parser.has_next():
    var entry = parser.next()
    process(entry)
parser.close()
```

## Parser Configuration

```mojo
from json import loads, ParserConfig

var config = ParserConfig(
    allow_comments=True,
    allow_trailing_comma=True,
    max_depth=100,
)
var data = loads('{"a": 1,} // comment', config)
```

## Serializer Configuration

```mojo
from json import dumps, SerializerConfig

var config = SerializerConfig(
    indent="  ",
    escape_unicode=True,
    escape_forward_slash=True,
)
var json = dumps(value, config)
```

## JSON Patch (RFC 6902)

```mojo
from json import apply_patch, loads

var doc = loads('{"name":"Alice","age":30}')
var patch = loads('[{"op":"replace","path":"/name","value":"Bob"}]')
var result = apply_patch(doc, patch)
# {"name":"Bob","age":30}
```

Operations: `add`, `remove`, `replace`, `move`, `copy`, `test`.

## JSON Merge Patch (RFC 7396)

```mojo
from json import merge_patch, create_merge_patch, loads

var target = loads('{"a":1,"b":2}')
var patch = loads('{"b":null,"c":3}')  # null removes keys
var result = merge_patch(target, patch)
# {"a":1,"c":3}
```

## JSONPath Queries

```mojo
from json import jsonpath_query, jsonpath_one, loads

var doc = loads('{"users":[{"name":"Alice"},{"name":"Bob"}]}')
var names = jsonpath_query(doc, "$.users[*].name")
# [Value("Alice"), Value("Bob")]
```

Supported syntax: `$`, `.key`, `[n]`, `[*]`, `..`, `[start:end]`, `[?expr]`.

## JSON Schema Validation

```mojo
from json import validate, is_valid, loads

var schema = loads('{"type":"object","required":["name"]}')
var doc = loads('{"name":"Alice"}')
if is_valid(doc, schema):
    print("Valid!")
```

Supported keywords: `type`, `enum`, `const`, `minimum/maximum`,
`minLength/maxLength`, `minItems/maxItems`, `items`, `required`,
`properties`, `additionalProperties`, `allOf`, `anyOf`, `oneOf`, `not`.
