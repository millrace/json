# json - Core Value type.
#
# `Value` is the public JSON value type. In v0.1 it stored arrays and
# objects as raw JSON substrings and re-parsed them on every access --
# a design the v0.2 redesign is incrementally walking back. Phase A
# moves the type and its helpers into the `value/` package and adds
# the `Document`/`Tape` storage that Phase B (copy-on-write) and
# Phase C (tape-backed reads) will depend on, while keeping every
# v0.1 method signature so callers (patch, jsonpath, schema,
# serialize, reflection) build unchanged.
#
# Helpers split:
#   - Pure string operations live in `raw_ops.mojo` (no Value dep).
#   - Value-dependent helpers (`_value_to_json`,
#     `_parse_json_value_to_value`, `_navigate_pointer`) live here so
#     the import graph stays acyclic.

from std.collections import List
from std.memory import ArcPointer

from .raw_ops import (
    _extract_field_value,
    _extract_array_element,
    _count_array_elements,
    _extract_object_keys,
    _parse_json_pointer,
)
from .owned import (
    OwnedValue,
    _materialize_for_write,
    _serialize_into_value,
    _set_at_pointer,
    _value_to_owned,
)
from ..document import (
    Document,
    TAPE_TAG_NULL,
    TAPE_TAG_BOOL,
    TAPE_TAG_INT,
    TAPE_TAG_FLOAT,
    TAPE_TAG_STRING,
    TAPE_TAG_STRING_OWNED,
    TAPE_TAG_ARRAY,
    TAPE_TAG_OBJECT,
    TAPE_TAG_KEY,
)
from ..unicode import unescape_json_string


struct Null(Writable):
    """Represents JSON null."""

    def __init__(out self):
        pass

    def __str__(self) -> String:
        return "null"

    def write_to[W: Writer](self, mut writer: W):
        writer.write("null")


struct Value(Copyable, Movable, Writable):
    """A JSON value that can hold null, bool, int, float, string, array, or object.

    A `Value` lives in one of two storage modes:

    * Legacy / owned: `_is_view == False`. The fields `_type`, `_bool`,
      `_int`, `_float`, `_string`, `_raw`, `_keys`, `_count` carry the
      data; arrays / objects are reconstituted from `_raw` on demand.
      This is the v0.1 representation.
    * Tape-backed view: `_is_view == True`. The data lives in
      `_doc[].tape[_tape_idx]` and the side pools of `_doc`. Read
      accessors below dispatch on `_is_view` and read directly from
      the tape; mutations (`set` / `append`) materialise the view into
      a legacy tree first.

    The view path is what `parse_into_document` (Phase 2a) feeds into
    when `loads()` is wired to the tape pipeline (Phase 2c).
    """

    var _type: Int  # 0=null, 1=bool, 2=int, 3=float, 4=string, 5=array, 6=object
    var _bool: Bool
    var _int: Int64
    var _float: Float64
    var _string: String
    var _raw: String  # Raw JSON for arrays/objects
    var _keys: List[String]  # Object keys
    var _count: Int  # Array/object element count
    # Phase 2b: tape-backed view fields. `_doc` is shared via
    # ArcPointer so child views (e.g. those returned by
    # `array_items()`) bump a refcount instead of cloning the document.
    var _is_view: Bool
    var _doc: ArcPointer[Document]
    var _tape_idx: Int

    def __init__(out self, null: Null):
        self._type = 0
        self._bool = False
        self._int = 0
        self._float = 0.0
        self._string = String()
        self._raw = String()
        self._keys = List[String]()
        self._count = 0
        self._is_view = False
        self._doc = ArcPointer(Document())
        self._tape_idx = -1

    def __init__(out self, none: NoneType):
        self._type = 0
        self._bool = False
        self._int = 0
        self._float = 0.0
        self._string = String()
        self._raw = String()
        self._keys = List[String]()
        self._count = 0
        self._is_view = False
        self._doc = ArcPointer(Document())
        self._tape_idx = -1

    def __init__(out self, b: Bool):
        self._type = 1
        self._bool = b
        self._int = 0
        self._float = 0.0
        self._string = String()
        self._raw = String()
        self._keys = List[String]()
        self._count = 0
        self._is_view = False
        self._doc = ArcPointer(Document())
        self._tape_idx = -1

    def __init__(out self, i: Int):
        self._type = 2
        self._bool = False
        self._int = Int64(i)
        self._float = 0.0
        self._string = String()
        self._raw = String()
        self._keys = List[String]()
        self._count = 0
        self._is_view = False
        self._doc = ArcPointer(Document())
        self._tape_idx = -1

    def __init__(out self, i: Int64):
        self._type = 2
        self._bool = False
        self._int = i
        self._float = 0.0
        self._string = String()
        self._raw = String()
        self._keys = List[String]()
        self._count = 0
        self._is_view = False
        self._doc = ArcPointer(Document())
        self._tape_idx = -1

    def __init__(out self, f: Float64):
        self._type = 3
        self._bool = False
        self._int = 0
        self._float = f
        self._string = String()
        self._raw = String()
        self._keys = List[String]()
        self._count = 0
        self._is_view = False
        self._doc = ArcPointer(Document())
        self._tape_idx = -1

    def __init__(out self, s: String):
        self._type = 4
        self._bool = False
        self._int = 0
        self._float = 0.0
        self._string = s
        self._raw = String()
        self._keys = List[String]()
        self._count = 0
        self._is_view = False
        self._doc = ArcPointer(Document())
        self._tape_idx = -1

    def copy(self) -> Self:
        """Create a copy of this Value.

        For legacy values this is a deep copy of the in-memory tree.
        For tape-backed views the copy is cheap: the document lives
        behind an ArcPointer, so we bump a refcount and clone the
        scalar fields.

        Returns:
            A new Value with the same content.
        """
        var v = Value(Null())
        v._type = self._type
        v._bool = self._bool
        v._int = self._int
        v._float = self._float
        v._string = self._string
        v._raw = self._raw
        v._keys = self._keys.copy()
        v._count = self._count
        v._is_view = self._is_view
        v._doc = self._doc.copy()
        v._tape_idx = self._tape_idx
        return v^

    def clone(self) -> Self:
        """Alias for copy(). Creates a deep copy of this Value.

        Returns:
            A new Value with the same content.
        """
        return self.copy()

    # ------------------------------------------------------------------
    # View-mode helpers
    # ------------------------------------------------------------------

    @always_inline
    def _view_tag(self) -> UInt8:
        """Return the tape tag for the entry this view points at."""
        return self._doc[].get_tag(self._tape_idx)

    # Type checking
    def is_null(self) -> Bool:
        if self._is_view:
            return self._view_tag() == TAPE_TAG_NULL
        return self._type == 0

    def is_bool(self) -> Bool:
        if self._is_view:
            return self._view_tag() == TAPE_TAG_BOOL
        return self._type == 1

    def is_int(self) -> Bool:
        if self._is_view:
            return self._view_tag() == TAPE_TAG_INT
        return self._type == 2

    def is_float(self) -> Bool:
        if self._is_view:
            return self._view_tag() == TAPE_TAG_FLOAT
        return self._type == 3

    def is_string(self) -> Bool:
        if self._is_view:
            var t = self._view_tag()
            return t == TAPE_TAG_STRING or t == TAPE_TAG_STRING_OWNED
        return self._type == 4

    def is_array(self) -> Bool:
        if self._is_view:
            return self._view_tag() == TAPE_TAG_ARRAY
        return self._type == 5

    def is_object(self) -> Bool:
        if self._is_view:
            return self._view_tag() == TAPE_TAG_OBJECT
        return self._type == 6

    def is_number(self) -> Bool:
        if self._is_view:
            var t = self._view_tag()
            return t == TAPE_TAG_INT or t == TAPE_TAG_FLOAT
        return self._type == 2 or self._type == 3

    # Value extraction
    def bool_value(self) -> Bool:
        if self._is_view:
            return self._doc[].get_bool(self._tape_idx)
        return self._bool

    def int_value(self) -> Int64:
        if self._is_view:
            return self._doc[].get_int(self._tape_idx)
        return self._int

    def float_value(self) -> Float64:
        if self._is_view:
            return self._doc[].get_float(self._tape_idx)
        return self._float

    def string_value(self) -> String:
        if self._is_view:
            return self._doc[].get_string(self._tape_idx)
        return self._string

    def raw_json(self) -> String:
        if self._is_view:
            return _view_to_json(self)
        return self._raw

    def array_count(self) -> Int:
        if self._is_view:
            return self._doc[].get_count(self._tape_idx)
        return self._count

    def object_keys(self) -> List[String]:
        if self._is_view:
            ref doc = self._doc[]
            var pair_count = doc.get_count(self._tape_idx)
            var child_start = doc.get_child_start(self._tape_idx)
            var keys = List[String](capacity=pair_count)
            for i in range(pair_count):
                keys.append(doc.get_key(child_start + 2 * i))
            return keys^
        return self._keys.copy()

    def object_count(self) -> Int:
        if self._is_view:
            return self._doc[].get_count(self._tape_idx)
        return self._count

    # Stringable
    def __str__(self) -> String:
        if self._is_view:
            return _view_to_json(self)
        if self._type == 0:
            return "null"
        elif self._type == 1:
            return "true" if self._bool else "false"
        elif self._type == 2:
            return String(self._int)
        elif self._type == 3:
            return String(self._float)
        elif self._type == 4:
            return '"' + self._string + '"'
        elif self._type == 5 or self._type == 6:
            return self._raw
        return "unknown"

    def write_to[W: Writer](self, mut writer: W):
        writer.write(self.__str__())

    # Equality. Equality between view and legacy Values compares
    # their materialized JSON forms, which is the only definition
    # consistent with what user code can observe via the public API.
    def __eq__(self, other: Value) -> Bool:
        if self._is_view or other._is_view:
            return _view_eq(self, other)
        if self._type != other._type:
            return False
        if self._type == 0:
            return True
        elif self._type == 1:
            return self._bool == other._bool
        elif self._type == 2:
            return self._int == other._int
        elif self._type == 3:
            return self._float == other._float
        elif self._type == 4:
            return self._string == other._string
        elif self._type == 5 or self._type == 6:
            return self._raw == other._raw
        return False

    def __ne__(self, other: Value) -> Bool:
        return not self.__eq__(other)

    def get(self, key: String) raises -> String:
        """Get a field value from a JSON object as a string.

        This is a helper for deserialization. For objects, it parses
        the raw JSON to extract the field value.

        Args:
            key: The field name to extract.

        Returns:
            The raw JSON value as a string.

        Raises:
            Error if not an object or key not found.
        """
        if not self.is_object():
            raise Error("get() can only be called on JSON objects")

        if self._is_view:
            ref doc = self._doc[]
            var pair_count = doc.get_count(self._tape_idx)
            var child_start = doc.get_child_start(self._tape_idx)
            for i in range(pair_count):
                if doc.get_key(child_start + 2 * i) == key:
                    var v = _make_view_child(self._doc, child_start + 2 * i + 1)
                    return _view_to_json(v)
            raise Error("Key '" + key + "' not found in JSON object")

        var found = False
        for i in range(len(self._keys)):
            if self._keys[i] == key:
                found = True
                break

        if not found:
            raise Error("Key '" + key + "' not found in JSON object")

        return _extract_field_value(self._raw, key)

    def array_items(self) raises -> List[Value]:
        """Get all items in a JSON array as a list of Values.

        Returns:
            List of Value objects representing array elements.

        Raises:
            Error if not an array.

        Example:
            var data = loads('[1, "hello", true]')
            for item in data.array_items():
                print(item).
        """
        if not self.is_array():
            raise Error("array_items() can only be called on JSON arrays")

        if self._is_view:
            ref doc = self._doc[]
            var count = doc.get_count(self._tape_idx)
            var child_start = doc.get_child_start(self._tape_idx)
            var result = List[Value](capacity=count)
            for i in range(count):
                result.append(_make_view_child(self._doc, child_start + i))
            return result^

        var result = List[Value]()
        var raw = self._raw

        if self._count == 0:
            return result^

        for i in range(self._count):
            var elem_str = _extract_array_element(raw, i)
            var elem = _parse_json_value_to_value(elem_str)
            result.append(elem^)

        return result^

    def object_items(self) raises -> List[Tuple[String, Value]]:
        """Get all key-value pairs in a JSON object.

        Returns:
            List of (key, value) tuples.

        Raises:
            Error if not an object.

        Example:
            var data = loads('{"a": 1, "b": 2}')
            for pair in data.object_items():
                var key = pair[0]
                var value = pair[1]
                print(key, value).
        """
        if not self.is_object():
            raise Error("object_items() can only be called on JSON objects")

        if self._is_view:
            ref doc = self._doc[]
            var pair_count = doc.get_count(self._tape_idx)
            var child_start = doc.get_child_start(self._tape_idx)
            var result = List[Tuple[String, Value]](capacity=pair_count)
            for i in range(pair_count):
                var k = doc.get_key(child_start + 2 * i)
                var v = _make_view_child(self._doc, child_start + 2 * i + 1)
                result.append((k, v^))
            return result^

        var result = List[Tuple[String, Value]]()
        var raw = self._raw

        for i in range(len(self._keys)):
            var key = self._keys[i]
            var value_str = _extract_field_value(raw, key)
            var value = _parse_json_value_to_value(value_str)
            result.append((key, value^))

        return result^

    def __getitem__(self, index: Int) raises -> Value:
        """Get array element by index.

        Args:
            index: Zero-based array index.

        Returns:
            The Value at the given index.

        Example:
            var arr = loads('[1, 2, 3]')
            print(arr[0])  # Prints 1.
        """
        if not self.is_array():
            raise Error("Index access requires a JSON array")

        if self._is_view:
            var count = self._doc[].get_count(self._tape_idx)
            if index < 0 or index >= count:
                raise Error("Array index out of bounds: " + String(index))
            var child_start = self._doc[].get_child_start(self._tape_idx)
            return _make_view_child(self._doc, child_start + index)

        if index < 0 or index >= self._count:
            raise Error("Array index out of bounds: " + String(index))
        var elem_str = _extract_array_element(self._raw, index)
        return _parse_json_value_to_value(elem_str)

    def __getitem__(self, key: String) raises -> Value:
        """Get object value by key.

        Args:
            key: Object key.

        Returns:
            The Value for the given key.

        Example:
            var obj = loads('{"name": "Alice"}')
            print(obj["name"])  # Prints "Alice".
        """
        if not self.is_object():
            raise Error("Key access requires a JSON object")

        if self._is_view:
            ref doc = self._doc[]
            var pair_count = doc.get_count(self._tape_idx)
            var child_start = doc.get_child_start(self._tape_idx)
            for i in range(pair_count):
                if doc.get_key(child_start + 2 * i) == key:
                    return _make_view_child(self._doc, child_start + 2 * i + 1)
            raise Error("Key not found: " + key)

        var found = False
        for i in range(len(self._keys)):
            if self._keys[i] == key:
                found = True
                break

        if not found:
            raise Error("Key not found: " + key)

        var value_str = _extract_field_value(self._raw, key)
        return _parse_json_value_to_value(value_str)

    def set(mut self, key: String, value: Value) raises:
        """Set or update a value in a JSON object.

        Routes through `OwnedValue` so the in-memory tree stays the source of
        truth for the mutation. The view is then re-serialized so subsequent
        reads via `raw_json()` / `__str__()` observe the new state.

        Args:
            key: Object key.
            value: New value to set.

        Example:
            var obj = loads('{"name": "Alice"}')
            obj.set("age", Value(30))
            obj.set("name", Value("Bob"))  # Update existing.
        """
        if not self.is_object():
            raise Error("set() can only be called on JSON objects")

        var owned = _materialize_for_write(self)
        var owned_value = _value_to_owned(value)

        var key_pos = -1
        for i in range(len(owned.object_keys)):
            if owned.object_keys[i] == key:
                key_pos = i
                break

        if key_pos >= 0:
            owned.object_values[key_pos] = owned_value^
        else:
            owned.object_keys.append(key)
            owned.object_values.append(owned_value^)

        var rebuilt = _serialize_into_value(owned)
        # Mutation always lands in the legacy representation: drop the
        # tape-backed view (if any) and replace the legacy fields.
        self._is_view = False
        self._tape_idx = -1
        self._type = 6
        self._raw = rebuilt._raw
        self._keys = rebuilt._keys.copy()
        self._count = rebuilt._count

    def set(mut self, index: Int, value: Value) raises:
        """Set a value at an array index.

        Args:
            index: Array index (must be valid).
            value: New value to set.

        Example:
            var arr = loads('[1, 2, 3]')
            arr.set(1, Value(20))  # Result is `[1, 20, 3]`.
        """
        if not self.is_array():
            raise Error("set(index) can only be called on JSON arrays")
        var current_count = self.array_count()
        if index < 0 or index >= current_count:
            raise Error("Array index out of bounds: " + String(index))

        var owned = _materialize_for_write(self)
        var owned_value = _value_to_owned(value)
        owned.array_val[index] = owned_value^

        var rebuilt = _serialize_into_value(owned)
        self._is_view = False
        self._tape_idx = -1
        self._type = 5
        self._raw = rebuilt._raw
        self._count = rebuilt._count

    def append(mut self, value: Value) raises:
        """Append a value to a JSON array.

        Args:
            value: Value to append.

        Example:
            var arr = loads('[1, 2]')
            arr.append(Value(3))  # Result is `[1, 2, 3]`.
        """
        if not self.is_array():
            raise Error("append() can only be called on JSON arrays")

        var owned = _materialize_for_write(self)
        var owned_value = _value_to_owned(value)
        owned.array_val.append(owned_value^)

        var rebuilt = _serialize_into_value(owned)
        self._is_view = False
        self._tape_idx = -1
        self._type = 5
        self._raw = rebuilt._raw
        self._count = rebuilt._count

    def set_at(mut self, pointer: String, value: Value) raises:
        """Set a nested value via JSON Pointer (RFC 6901).

        Unlike chained `__getitem__`, this propagates the mutation through the
        full parent chain. Intermediate objects/arrays are created or updated
        as required by the pointer; missing scalar parents raise.

        Args:
            pointer: JSON Pointer string (`""` for root, `"/a/b"` for nested).
            value: New value to install at `pointer`.

        Example:
            var doc = loads('{"a":{"b":1}}')
            doc.set_at("/a/b", Value(42))  # Result is `{"a":{"b":42}}`.
        """
        if pointer == "":
            # Whole-document replacement.
            self._type = value._type
            self._bool = value._bool
            self._int = value._int
            self._float = value._float
            self._string = value._string
            self._raw = value._raw
            self._keys = value._keys.copy()
            self._count = value._count
            self._is_view = value._is_view
            self._doc = value._doc.copy()
            self._tape_idx = value._tape_idx
            return

        var tokens = _parse_json_pointer(pointer)
        var tree = _materialize_for_write(self)
        var new_val = _value_to_owned(value)
        _set_at_pointer(tree, tokens, 0, new_val^)

        var rebuilt = _serialize_into_value(tree)
        # Mutation always lands in the legacy representation.
        self._is_view = False
        self._tape_idx = -1
        self._type = rebuilt._type
        self._bool = rebuilt._bool
        self._int = rebuilt._int
        self._float = rebuilt._float
        self._string = rebuilt._string
        self._raw = rebuilt._raw
        self._keys = rebuilt._keys.copy()
        self._count = rebuilt._count

    def at(self, pointer: String) raises -> Value:
        """Navigate to a value using JSON Pointer (RFC 6901).

        JSON Pointer syntax:
            "" (empty) = the whole document.
            "/foo" = member "foo" of object.
            "/foo/0" = first element of array "foo".
            "/a~1b" = member "a/b" (/ escaped as ~1).
            "/m~0n" = member "m~n" (~ escaped as ~0).

        Args:
            pointer: JSON Pointer string (e.g., "/users/0/name").

        Returns:
            The Value at the pointer location.

        Raises:
            Error if pointer is invalid or path doesn't exist.

        Example:
            var data = loads('{"users":[{"name":"Alice"}]}')
            var name = data.at("/users/0/name")  # `Value("Alice")`.
        """
        if pointer == "":
            return self.copy()

        if not pointer.startswith("/"):
            raise Error("JSON Pointer must start with '/' or be empty")

        var tokens = _parse_json_pointer(pointer)
        return _navigate_pointer(self, tokens)


def make_array_value(raw: String, count: Int) -> Value:
    """Create an array Value from raw JSON."""
    var v = Value(Null())
    v._type = 5
    v._raw = raw
    v._count = count
    return v^


def make_object_value(raw: String, var keys: List[String]) -> Value:
    """Create an object Value from raw JSON and keys."""
    var v = Value(Null())
    v._type = 6
    v._raw = raw
    v._count = len(keys)
    v._keys = keys^
    return v^


# ---------------------------------------------------------------------------
# Value-dependent helpers
# ---------------------------------------------------------------------------


def _value_to_json(v: Value) -> String:
    """Convert a Value to its JSON string representation.

    Used by `Value.set` / `Value.append` to render the new value into
    JSON text before splicing it into the raw object/array string.
    Phase B replaces this with a structured COW path. The full-featured
    serializer (indent, escape options, etc.) is `serialize.dumps`.
    """
    if v.is_null():
        return "null"
    elif v.is_bool():
        return "true" if v.bool_value() else "false"
    elif v.is_int():
        return String(v.int_value())
    elif v.is_float():
        return String(v.float_value())
    elif v.is_string():
        var result = String('"')
        var s = v.string_value()
        var s_bytes = s.as_bytes()
        for i in range(len(s_bytes)):
            var c = s_bytes[i]
            if c == UInt8(ord('"')):
                result += '\\"'
            elif c == UInt8(ord("\\")):
                result += "\\\\"
            elif c == UInt8(ord("\n")):
                result += "\\n"
            elif c == UInt8(ord("\r")):
                result += "\\r"
            elif c == UInt8(ord("\t")):
                result += "\\t"
            else:
                result += chr(Int(c))
        result += '"'
        return result^
    elif v.is_array() or v.is_object():
        return v.raw_json()
    return "null"


def _parse_json_value_to_value(json_str: String) raises -> Value:
    """Parse a raw JSON value string into a Value.

    Used by `Value.__getitem__`, `Value.array_items`, `Value.object_items`,
    `LazyValue.get`, and `patch.apply_patch` to materialize child values
    on demand. Phase C replaces this scalar walk with a tape lookup.
    """
    var s = json_str
    var s_bytes = s.as_bytes()
    var n = len(s_bytes)

    if n == 0:
        raise Error("Empty JSON value")

    var i = 0
    while i < n and (
        s_bytes[i] == UInt8(ord(" "))
        or s_bytes[i] == UInt8(ord("\t"))
        or s_bytes[i] == UInt8(ord("\n"))
        or s_bytes[i] == UInt8(ord("\r"))
    ):
        i += 1

    if i >= n:
        raise Error("Empty JSON value")

    var first_char = s_bytes[i]

    if first_char == UInt8(ord("n")):
        return Value(Null())

    if first_char == UInt8(ord("t")):
        return Value(True)

    if first_char == UInt8(ord("f")):
        return Value(False)

    if first_char == UInt8(ord('"')):
        var start_idx = i + 1
        var end_idx = start_idx
        var has_escapes = False
        while end_idx < n:
            var c = s_bytes[end_idx]
            if c == UInt8(ord("\\")):
                has_escapes = True
                end_idx += 2
                continue
            if c == UInt8(ord('"')):
                break
            end_idx += 1

        if not has_escapes:
            return Value(
                String(String(unsafe_from_utf8=s.as_bytes()[start_idx:end_idx]))
            )

        var bytes_list = List[UInt8](capacity=n)
        for j in range(n):
            bytes_list.append(s_bytes[j])
        var unescaped = unescape_json_string(bytes_list, start_idx, end_idx)
        return Value(String(unsafe_from_utf8=unescaped^))

    if first_char == UInt8(ord("-")) or (
        first_char >= UInt8(ord("0")) and first_char <= UInt8(ord("9"))
    ):
        var num_str = String()
        var is_float = False
        while i < n:
            var c = s_bytes[i]
            if (
                c == UInt8(ord("-"))
                or c == UInt8(ord("+"))
                or (c >= UInt8(ord("0")) and c <= UInt8(ord("9")))
            ):
                num_str += chr(Int(c))
            elif (
                c == UInt8(ord("."))
                or c == UInt8(ord("e"))
                or c == UInt8(ord("E"))
            ):
                num_str += chr(Int(c))
                is_float = True
            else:
                break
            i += 1
        if is_float:
            return Value(atof(num_str))
        else:
            return Value(atol(num_str))

    if first_char == UInt8(ord("[")):
        var count = _count_array_elements(s)
        return make_array_value(s, count)

    if first_char == UInt8(ord("{")):
        var keys = _extract_object_keys(s)
        return make_object_value(s, keys^)

    raise Error("Invalid JSON value: " + s)


def _navigate_pointer(v: Value, tokens: List[String]) raises -> Value:
    """Navigate through a Value using parsed pointer tokens."""
    if len(tokens) == 0:
        return v.copy()

    var current_raw = v.raw_json() if v.is_array() or v.is_object() else ""
    var token = tokens[0]

    if v.is_object():
        var value_str = _extract_field_value(current_raw, token)
        var child = _parse_json_value_to_value(value_str)

        if len(tokens) == 1:
            return child^

        var remaining = List[String]()
        for i in range(1, len(tokens)):
            remaining.append(tokens[i])
        return _navigate_pointer(child, remaining^)

    elif v.is_array():
        var index: Int
        try:
            index = atol(token)
        except:
            raise Error("Array index must be a number: " + token)

        if index < 0:
            raise Error("Array index cannot be negative: " + token)

        var value_str = _extract_array_element(current_raw, index)
        var child = _parse_json_value_to_value(value_str)

        if len(tokens) == 1:
            return child^

        var remaining = List[String]()
        for i in range(1, len(tokens)):
            remaining.append(tokens[i])
        return _navigate_pointer(child, remaining^)

    else:
        raise Error(
            "Cannot navigate into primitive value with pointer: /" + token
        )


# ---------------------------------------------------------------------------
# Tape-backed view helpers (Phase 2b)
# ---------------------------------------------------------------------------


def make_view_value(doc: ArcPointer[Document], tape_idx: Int) -> Value:
    """Build a tape-backed `Value` view over `doc[].tape[tape_idx]`.

    The view shares ownership of `doc` -- copying the resulting Value
    (including child views from `array_items` / `object_items`) only
    bumps an atomic refcount. Phase 2c routes `loads(target='cpu')`
    through this factory.
    """
    var v = Value(Null())
    v._is_view = True
    v._doc = doc
    v._tape_idx = tape_idx
    # Reflect the tape tag in `_type` so any v0.1 code that still
    # peeks at `_type` directly stays honest. View-mode dispatch
    # always wins, but accessors that don't dispatch (e.g. raw
    # struct field reads in user code) still see the right kind.
    var tag = doc[].get_tag(tape_idx)
    if tag == TAPE_TAG_NULL:
        v._type = 0
    elif tag == TAPE_TAG_BOOL:
        v._type = 1
    elif tag == TAPE_TAG_INT:
        v._type = 2
    elif tag == TAPE_TAG_FLOAT:
        v._type = 3
    elif tag == TAPE_TAG_STRING or tag == TAPE_TAG_STRING_OWNED:
        v._type = 4
    elif tag == TAPE_TAG_ARRAY:
        v._type = 5
    elif tag == TAPE_TAG_OBJECT:
        v._type = 6
    return v^


def _make_view_child(doc: ArcPointer[Document], tape_idx: Int) -> Value:
    """Build a child view by sharing the parent's `doc`.

    Identical to `make_view_value` but kept private so the factory
    function can stay the only public entry point for constructing a
    view.
    """
    return make_view_value(doc, tape_idx)


def _view_to_json(v: Value) -> String:
    """Serialize a tape-backed view into a JSON string.

    Used by `Value.raw_json()` / `Value.__str__()` / equality
    comparison when the value is in view mode. Walks the tape
    recursively. Non-raising: on a corrupt tape the function returns
    a sentinel "<bad-tape>" string, which is a strictly better
    failure mode than panicking inside `__str__`.
    """
    if not v._is_view:
        return v.raw_json()
    return _emit_view_json(v._doc, v._tape_idx)


def _emit_view_json(doc: ArcPointer[Document], tape_idx: Int) -> String:
    ref d = doc[]
    var tag = d.get_tag(tape_idx)
    if tag == TAPE_TAG_NULL:
        return "null"
    if tag == TAPE_TAG_BOOL:
        return "true" if d.get_bool(tape_idx) else "false"
    if tag == TAPE_TAG_INT:
        return String(d.get_int(tape_idx))
    if tag == TAPE_TAG_FLOAT:
        return String(d.get_float(tape_idx))
    if tag == TAPE_TAG_STRING or tag == TAPE_TAG_STRING_OWNED:
        var s = d.get_string(tape_idx)
        return _escape_json_string(s)
    if tag == TAPE_TAG_ARRAY:
        var count = d.get_count(tape_idx)
        var child_start = d.get_child_start(tape_idx)
        var out = String("[")
        for i in range(count):
            if i > 0:
                out += ","
            out += _emit_view_json(doc, child_start + i)
        out += "]"
        return out^
    if tag == TAPE_TAG_OBJECT:
        var pair_count = d.get_count(tape_idx)
        var child_start = d.get_child_start(tape_idx)
        var out = String("{")
        for i in range(pair_count):
            if i > 0:
                out += ","
            var key = d.get_key(child_start + 2 * i)
            out += _escape_json_string(key)
            out += ":"
            out += _emit_view_json(doc, child_start + 2 * i + 1)
        out += "}"
        return out^
    return String("<bad-tape>")


def _escape_json_string(s: String) -> String:
    """Wrap a string in quotes and escape the JSON-mandatory bytes
    (`\"`, `\\`, control chars 0..31). Used by `_emit_view_json`.
    """
    var out = String('"')
    var b = s.as_bytes()
    for i in range(len(b)):
        var c = b[i]
        if c == UInt8(ord('"')):
            out += '\\"'
        elif c == UInt8(ord("\\")):
            out += "\\\\"
        elif c == UInt8(ord("\n")):
            out += "\\n"
        elif c == UInt8(ord("\r")):
            out += "\\r"
        elif c == UInt8(ord("\t")):
            out += "\\t"
        elif c == UInt8(ord("\b")):
            out += "\\b"
        elif c == UInt8(ord("\f")):
            out += "\\f"
        elif c < UInt8(0x20):
            out += "\\u00"
            var hi = Int(c) >> 4
            var lo = Int(c) & 0xF
            out += chr(ord("0") + hi) if hi < 10 else chr(ord("a") + hi - 10)
            out += chr(ord("0") + lo) if lo < 10 else chr(ord("a") + lo - 10)
        else:
            out += chr(Int(c))
    out += '"'
    return out^


def _view_eq(a: Value, b: Value) -> Bool:
    """Equality comparison when at least one side is a view.

    Compares by serialized JSON, which is the only definition that
    survives mixing view and legacy values without round-tripping
    everything through OwnedValue.
    """
    var sa = a.raw_json() if (a.is_array() or a.is_object()) else String()
    var sb = b.raw_json() if (b.is_array() or b.is_object()) else String()
    if a.is_array() or a.is_object() or b.is_array() or b.is_object():
        if a.is_array() != b.is_array():
            return False
        if a.is_object() != b.is_object():
            return False
        return sa == sb
    if a.is_null() != b.is_null():
        return False
    if a.is_null():
        return True
    if a.is_bool() != b.is_bool():
        return False
    if a.is_bool():
        return a.bool_value() == b.bool_value()
    if a.is_int() != b.is_int():
        return False
    if a.is_int():
        return a.int_value() == b.int_value()
    if a.is_float() != b.is_float():
        return False
    if a.is_float():
        return a.float_value() == b.float_value()
    if a.is_string() != b.is_string():
        return False
    if a.is_string():
        return a.string_value() == b.string_value()
    return False
