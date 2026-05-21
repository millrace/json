# Stage 2: walk a `StructuralIndex` and produce a `Value`.
#
# Stage 1 -- either `stage1_scalar.parse_structural_scalar` or
# `stage1.parse_structural_simd` -- produces an ordered list of byte
# offsets into the input, one per structural character (plus quote
# boundaries). Stage 2 walks that list left-to-right and reconstructs a
# `Value` tree without ever re-scanning the byte stream for structure.
#
# Compared with the v0.1 `parse_simd` pipeline (single-pass byte scan)
# the stage 2 walker is structurally simpler and keeps the parser logic
# decoupled from the structural-scanning logic. That decoupling is the
# real win in v0.2: future SIMD or GPU stage 1 implementations can be
# swapped in without touching the value construction logic.
#
# Output representation
# ---------------------
# Values for arrays and objects use the v0.1 `make_array_value` /
# `make_object_value` factories so the rest of the library (LazyValue,
# JSONPath, JSON Patch, schema validation) needs no changes. A future
# patch can switch stage 2 to emit a tape-backed `Document` once the
# read-side `Value` view supports it.

from std.collections import List
from std.memory import memcpy

from ..value import (
    Value,
    Null,
    make_array_value,
    make_object_value,
)
from ..unicode import unescape_json_string, unescape_json_string_span
from ..document import (
    Document,
    pack_tape_entry,
    pack_pair,
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
from .stage1_scalar import StructuralIndex


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_with_index(input: String, index: StructuralIndex) raises -> Value:
    """Build a `Value` by walking the structural index over `input`.

    Args:
        input: Original JSON bytes.
        index: Output of stage 1 (`parse_structural_scalar` or
               `parse_structural_simd`).

    Returns:
        Parsed `Value`.
    """
    var positions = index.positions.copy()
    var pos_idx = 0
    var n = input.byte_length()
    var bytes = input.as_bytes()

    # Skip leading whitespace so primitives at the document root start
    # at a known byte offset.
    var doc_start = 0
    while doc_start < n and _is_ws(bytes[doc_start]):
        doc_start += 1

    var value = _parse_value(input, positions, pos_idx, doc_start, n)

    # Compute byte offset of the first byte AFTER the parsed value.
    # Containers and strings advance `pos_idx` past their final
    # structural (close-bracket / close-quote); primitives leave it
    # untouched, so we walk bytes until we hit whitespace or EOF.
    var consumed_end: Int
    if pos_idx > 0:
        consumed_end = Int(positions[pos_idx - 1]) + 1
    else:
        consumed_end = _primitive_end(bytes, doc_start, n)

    # Trailing-content check: only whitespace may follow the value.
    while consumed_end < n:
        if not _is_ws(bytes[consumed_end]):
            raise Error(
                "Stage 2: trailing content after top-level JSON value at"
                " offset "
                + String(consumed_end)
            )
        consumed_end += 1

    return value^


def _primitive_end(bytes: Span[UInt8, _], start: Int, n: Int) -> Int:
    """Find the first byte after a top-level primitive (number / null /
    true / false). Skips any leading whitespace (so callers can pass
    the byte right after `:` or `,`), then scans until whitespace,
    EOF, or a structural byte."""
    var i = start
    while i < n and _is_ws(bytes[i]):
        i += 1
    while i < n:
        var c = bytes[i]
        if (
            _is_ws(c)
            or c == UInt8(ord(","))
            or c == UInt8(ord("}"))
            or c == UInt8(ord("]"))
        ):
            break
        i += 1
    return i


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


@always_inline
def _is_ws(b: UInt8) -> Bool:
    return (
        b == UInt8(ord(" "))
        or b == UInt8(ord("\t"))
        or b == UInt8(ord("\n"))
        or b == UInt8(ord("\r"))
    )


def _skip_ws_input(input: String, start: Int, end: Int) -> Int:
    var bytes = input.as_bytes()
    var i = start
    while i < end and _is_ws(bytes[i]):
        i += 1
    return i


def _parse_value(
    input: String,
    mut positions: List[UInt32],
    mut pos_idx: Int,
    start: Int,
    end: Int,
) raises -> Value:
    var i = _skip_ws_input(input, start, end)
    if i >= end:
        raise Error("Stage 2: empty value")

    var bytes = input.as_bytes()
    var c = bytes[i]

    if c == UInt8(ord("{")):
        return _parse_object(input, positions, pos_idx, i, end)
    if c == UInt8(ord("[")):
        return _parse_array(input, positions, pos_idx, i, end)
    if c == UInt8(ord('"')):
        return _parse_string(input, positions, pos_idx, i)
    if c == UInt8(ord("n")):
        return Value(Null())
    if c == UInt8(ord("t")):
        return Value(True)
    if c == UInt8(ord("f")):
        return Value(False)
    if c == UInt8(ord("-")) or (c >= UInt8(ord("0")) and c <= UInt8(ord("9"))):
        return _parse_number(input, i, end)

    raise Error("Stage 2: unexpected character at offset " + String(i))


def _parse_string(
    input: String,
    mut positions: List[UInt32],
    mut pos_idx: Int,
    open_quote: Int,
) raises -> Value:
    """The opening quote was emitted by stage 1, so the cursor is sitting
    on it. The next position is the closing quote.
    """
    if pos_idx >= len(positions) or Int(positions[pos_idx]) != open_quote:
        raise Error(
            "Stage 2: cursor desync at string open offset " + String(open_quote)
        )
    pos_idx += 1
    if pos_idx >= len(positions):
        raise Error(
            "Stage 2: unterminated string at offset " + String(open_quote)
        )
    var close_quote = Int(positions[pos_idx])
    pos_idx += 1

    var bytes = input.as_bytes()
    var start_idx = open_quote + 1
    var end_idx = close_quote

    var has_escape = False
    for j in range(start_idx, end_idx):
        if bytes[j] == UInt8(ord("\\")):
            has_escape = True
            break

    if not has_escape:
        return Value(String(unsafe_from_utf8=bytes[start_idx:end_idx]))

    # Validate every escape sequence before handing off to the
    # unescaper: per JSON spec (RFC 8259 §7) only "\\\"\\/bfnrtu" are
    # permitted after a backslash. The unescaper itself is non-raising
    # and would silently keep the backslash for unknown escapes.
    _validate_escapes(bytes, start_idx, end_idx)

    # Span-based unescape: we only need to read between the quotes,
    # so binding `bytes` (a `Span[UInt8, _]` over the whole input)
    # avoids copying the input into a `List[UInt8]` per string.
    var unescaped = unescape_json_string_span(bytes, start_idx, end_idx)
    return Value(String(unsafe_from_utf8=unescaped^))


def _validate_escapes(bytes: Span[UInt8, _], start: Int, end: Int) raises:
    var j = start
    while j < end:
        if bytes[j] != UInt8(ord("\\")):
            j += 1
            continue
        if j + 1 >= end:
            raise Error("Stage 2: trailing backslash in string")
        var esc = bytes[j + 1]
        if (
            esc != UInt8(ord('"'))
            and esc != UInt8(ord("\\"))
            and esc != UInt8(ord("/"))
            and esc != UInt8(ord("b"))
            and esc != UInt8(ord("f"))
            and esc != UInt8(ord("n"))
            and esc != UInt8(ord("r"))
            and esc != UInt8(ord("t"))
            and esc != UInt8(ord("u"))
        ):
            raise Error(
                "Stage 2: invalid escape sequence '\\"
                + chr(Int(esc))
                + "' at offset "
                + String(j)
            )
        if esc == UInt8(ord("u")):
            j += 6
        else:
            j += 2


def _parse_number(input: String, start: Int, end: Int) raises -> Value:
    var bytes = input.as_bytes()
    var i = start
    var is_float = False
    if bytes[i] == UInt8(ord("-")):
        i += 1

    # Per JSON spec (RFC 8259) numbers are `0` or `[1-9][0-9]*`. Reject
    # leading zeros like `007` and `-00.5`.
    if (
        i < end
        and bytes[i] == UInt8(ord("0"))
        and i + 1 < end
        and bytes[i + 1] >= UInt8(ord("0"))
        and bytes[i + 1] <= UInt8(ord("9"))
    ):
        raise Error(
            "Stage 2: leading zeros are not allowed in JSON numbers (offset "
            + String(start)
            + ")"
        )

    while i < end:
        var c = bytes[i]
        if c >= UInt8(ord("0")) and c <= UInt8(ord("9")):
            i += 1
            continue
        if c == UInt8(ord(".")) or c == UInt8(ord("e")) or c == UInt8(ord("E")):
            is_float = True
            i += 1
            continue
        if c == UInt8(ord("+")) or c == UInt8(ord("-")):
            i += 1
            continue
        break

    if is_float:
        # Float still needs a String for atof.
        var num_str = String(unsafe_from_utf8=bytes[start:i])
        return Value(atof(num_str))
    # Integer fast path: SWAR-friendly inline parse, no String alloc.
    return Value(_parse_int_inline(bytes, start, i))


struct _CloseInfo(Copyable, Movable):
    """Result of `_find_matching_close`: index into the structural-position
    list AND the byte offset of the matching closing bracket/brace."""

    var close_idx: Int
    var close_offset: Int

    def __init__(out self, close_idx: Int, close_offset: Int):
        self.close_idx = close_idx
        self.close_offset = close_offset


def _find_matching_close(
    input: String,
    positions: List[UInt32],
    open_idx: Int,
    open_byte: UInt8,
    close_byte: UInt8,
) raises -> _CloseInfo:
    var bytes = input.as_bytes()
    var depth = 1
    var k = open_idx + 1
    while k < len(positions):
        var off = Int(positions[k])
        var b = bytes[off]
        if b == open_byte:
            depth += 1
        elif b == close_byte:
            depth -= 1
            if depth == 0:
                return _CloseInfo(k, off)
        k += 1
    raise Error("Stage 2: unterminated container")


def _parse_array(
    input: String,
    mut positions: List[UInt32],
    mut pos_idx: Int,
    open_offset: Int,
    end: Int,
) raises -> Value:
    if pos_idx >= len(positions) or Int(positions[pos_idx]) != open_offset:
        raise Error("Stage 2: cursor desync at array open")
    var open_idx = pos_idx
    pos_idx += 1

    var close = _find_matching_close(
        input, positions, open_idx, UInt8(ord("[")), UInt8(ord("]"))
    )
    var close_idx = close.close_idx
    var close_offset = close.close_offset

    # Count elements: top-level (depth 0) commas between the brackets,
    # plus 1 if the array body has any non-whitespace content.
    var bytes = input.as_bytes()
    var count = 0
    var depth = 0
    var last_top_comma_byte = -1
    var prev_top_comma_byte: Int
    var k = open_idx + 1
    while k < close_idx:
        var off = Int(positions[k])
        var b = bytes[off]
        if b == UInt8(ord("[")) or b == UInt8(ord("{")):
            depth += 1
        elif b == UInt8(ord("]")) or b == UInt8(ord("}")):
            depth -= 1
        elif b == UInt8(ord(",")) and depth == 0:
            count += 1
            prev_top_comma_byte = last_top_comma_byte
            last_top_comma_byte = off
            # Double-comma check: between two consecutive top-level
            # commas there must be a non-ws value byte.
            if prev_top_comma_byte >= 0:
                var has_between = False
                for j in range(prev_top_comma_byte + 1, off):
                    if not _is_ws(bytes[j]):
                        has_between = True
                        break
                if not has_between:
                    raise Error(
                        "Stage 2: empty element between commas in array"
                        " at offset "
                        + String(off)
                    )
            else:
                # Between `[` and the first comma must also be non-ws.
                var has_first = False
                for j in range(open_offset + 1, off):
                    if not _is_ws(bytes[j]):
                        has_first = True
                        break
                if not has_first:
                    raise Error(
                        "Stage 2: leading comma in array at offset "
                        + String(off)
                    )
        elif b == UInt8(ord('"')):
            # Skip the matching closing quote that stage 1 emitted.
            k += 1
        k += 1

    # `count` so far is the number of top-level commas. An array's
    # element count is `commas + 1` if there is any non-whitespace
    # content between the brackets, else 0 (`[]` / `[ ]`).
    var has_content = False
    for j in range(open_offset + 1, close_offset):
        if not _is_ws(bytes[j]):
            has_content = True
            break
    if has_content:
        count += 1

    # Trailing-comma check: if the last top-level comma is followed only
    # by whitespace before `]`, the array is malformed.
    if last_top_comma_byte >= 0:
        var has_value_after_comma = False
        for j in range(last_top_comma_byte + 1, close_offset):
            if not _is_ws(bytes[j]):
                has_value_after_comma = True
                break
        if not has_value_after_comma:
            raise Error(
                "Stage 2: trailing comma in array at offset "
                + String(last_top_comma_byte)
            )

    pos_idx = close_idx + 1
    var raw = String(unsafe_from_utf8=bytes[open_offset : close_offset + 1])
    return make_array_value(raw, count)


def _parse_object(
    input: String,
    mut positions: List[UInt32],
    mut pos_idx: Int,
    open_offset: Int,
    end: Int,
) raises -> Value:
    if pos_idx >= len(positions) or Int(positions[pos_idx]) != open_offset:
        raise Error("Stage 2: cursor desync at object open")
    var open_idx = pos_idx
    pos_idx += 1

    var close = _find_matching_close(
        input, positions, open_idx, UInt8(ord("{")), UInt8(ord("}"))
    )
    var close_idx = close.close_idx
    var close_offset = close.close_offset

    var bytes = input.as_bytes()
    var keys = List[String]()
    var depth = 0
    var expect_key = True
    var last_top_colon_byte = -1
    var last_top_comma_byte = -1
    var top_comma_count = 0
    # Tracks whether we have seen a `:` at depth 0 since the last key
    # was consumed. Reset on every top-level comma; cleared when a key
    # is consumed; set when `:` is encountered. Used to detect
    # `{"key" "value"}` (missing colon between key and value).
    var saw_colon_after_key = True
    var k = open_idx + 1
    while k < close_idx:
        var off = Int(positions[k])
        var b = bytes[off]

        if b == UInt8(ord("{")) or b == UInt8(ord("[")):
            depth += 1
            k += 1
            continue
        if b == UInt8(ord("}")) or b == UInt8(ord("]")):
            depth -= 1
            k += 1
            continue
        if b == UInt8(ord(",")) and depth == 0:
            expect_key = True
            saw_colon_after_key = True
            last_top_comma_byte = off
            top_comma_count += 1
            k += 1
            continue
        if b == UInt8(ord(":")) and depth == 0:
            last_top_colon_byte = off
            saw_colon_after_key = True
            k += 1
            continue
        if b == UInt8(ord('"')):
            if depth == 0 and expect_key:
                if k + 1 >= close_idx:
                    raise Error("Stage 2: malformed object key")
                var close_quote = Int(positions[k + 1])
                var key_start = off + 1
                var key_len = close_quote - key_start
                var key_bytes = List[UInt8](capacity=key_len)
                key_bytes.resize(key_len, 0)
                memcpy(
                    dest=key_bytes.unsafe_ptr(),
                    src=bytes.unsafe_ptr() + key_start,
                    count=key_len,
                )
                keys.append(String(unsafe_from_utf8=key_bytes^))
                expect_key = False
                saw_colon_after_key = False
                k += 2
                continue
            # Depth-0 string with `expect_key == False`. Must be the
            # value half of a key:value pair, so a colon must have
            # been seen since the last key.
            if depth == 0 and not saw_colon_after_key:
                raise Error(
                    "Stage 2: missing ':' between key and value at offset "
                    + String(off)
                )
            k += 2
            continue

        k += 1

    # Unquoted-key check: every member contributes one key, so the
    # number of collected keys must equal `top_comma_count + 1` for a
    # non-empty object. A missing key indicates an unquoted/missing
    # identifier (e.g., `{key: 1}`).
    var has_content = False
    for j in range(open_offset + 1, close_offset):
        if not _is_ws(bytes[j]):
            has_content = True
            break
    if has_content:
        var expected_keys = top_comma_count + 1
        if len(keys) != expected_keys:
            raise Error(
                "Stage 2: expected "
                + String(expected_keys)
                + " keys, got "
                + String(len(keys))
                + " (likely unquoted or missing key in object)"
            )

    # Trailing-comma check: if the last top-level comma is followed only
    # by whitespace before `}`, the object is malformed.
    if last_top_comma_byte >= 0:
        var has_kv_after_comma = False
        for j in range(last_top_comma_byte + 1, close_offset):
            if not _is_ws(bytes[j]):
                has_kv_after_comma = True
                break
        if not has_kv_after_comma:
            raise Error(
                "Stage 2: trailing comma in object at offset "
                + String(last_top_comma_byte)
            )

    # Missing-value check: a `:` at depth 0 must be followed by a non-
    # whitespace value before the next `,` or the closing `}`.
    if last_top_colon_byte >= 0:
        var has_value_after_colon = False
        var stop = close_offset
        if last_top_comma_byte > last_top_colon_byte:
            stop = last_top_comma_byte
        for j in range(last_top_colon_byte + 1, stop):
            if not _is_ws(bytes[j]):
                has_value_after_colon = True
                break
        if not has_value_after_colon:
            raise Error(
                "Stage 2: missing value after ':' at offset "
                + String(last_top_colon_byte)
            )

    pos_idx = close_idx + 1
    var raw = String(unsafe_from_utf8=bytes[open_offset : close_offset + 1])
    return make_object_value(raw, keys^)


# ---------------------------------------------------------------------------
# Convenience: full parse from a raw input string.
# ---------------------------------------------------------------------------


def parse_two_pass[force_scalar: Bool = False](input: String) raises -> Value:
    """End-to-end stage 1 + stage 2 parse.

    Parameters:
        force_scalar: When False (default), use the SIMD stage 1
            implementation (`stage1.parse_structural_simd`); on the
            benchmark corpora SIMD is 1.5x to 2.2x faster than the
            scalar walker. When True, use the scalar oracle -- useful
            for differential testing and for inputs small enough that
            the SIMD chunk loop never runs (n < 32). Both produce
            identical output (enforced by
            `tests/test_stage1_equivalence.mojo`); this is purely a
            performance switch.
    """
    from .stage1_scalar import parse_structural_scalar
    from .stage1 import parse_structural_simd

    comptime if force_scalar:
        var index = parse_structural_scalar(input)
        return parse_with_index(input, index)
    else:
        var index = parse_structural_simd(input)
        return parse_with_index(input, index)


def parse_two_pass_tape[
    force_scalar: Bool = False
](var input: String) raises -> Document:
    """End-to-end stage 1 + stage 2 parse that emits a `Document`.

    Parameters:
        force_scalar: When False (default), use SIMD stage 1; same
            switch as `parse_two_pass`.

    Args:
        input: JSON input. The returned `Document` owns this string,
            so its bytes can back zero-copy string slices on the tape.

    Returns:
        Owned `Document` with the root at `Document.root()`.
    """
    from .stage1_scalar import parse_structural_scalar
    from .stage1 import parse_structural_simd

    comptime if force_scalar:
        var index = parse_structural_scalar(input)
        return parse_into_document(input^, index)
    else:
        var index = parse_structural_simd(input)
        return parse_into_document(input^, index)


# ---------------------------------------------------------------------------
# Tape-emitting walker (Phase 2a).
#
# `parse_into_document` walks the same `StructuralIndex` as
# `parse_with_index` but emits entries into a `Document.tape` instead
# of building a `Value` tree with raw-JSON substrings.
#
# Layout follows the rules pinned in `json/document.mojo`:
#   - Children are written before parents so the parent's
#     `child_start_idx` payload points backwards into a contiguous run
#     of header entries.
#   - For an OBJECT, that contiguous run alternates KEY, VALUE, KEY,
#     VALUE, ... (so `count` pairs occupy `2 * count` slots).
#   - The root is the last entry, which is what `Document.root()`
#     assumes.
#
# Validation logic mirrors `_parse_array` / `_parse_object` exactly:
# trailing commas, double commas, leading commas, missing colons,
# missing values after colons, and unquoted keys all raise the same
# errors as the v0.1 path. The two paths must accept and reject
# byte-identical inputs; that's enforced by tests/test_stage2_tape.mojo.
# ---------------------------------------------------------------------------


def parse_into_document(
    var input: String, index: StructuralIndex
) raises -> Document:
    """Build a `Document` (tape + side pools) by walking the structural
    index over `input`. The returned document owns `input`.

    Args:
        input: Original JSON bytes.
        index: Output of stage 1.

    Returns:
        Owned `Document` whose root entry is the last tape slot.
    """
    var positions = index.positions.copy()
    var pos_idx = 0
    var n = input.byte_length()
    var bytes = input.as_bytes()

    var doc_start = 0
    while doc_start < n and _is_ws(bytes[doc_start]):
        doc_start += 1

    var doc = Document(input^)

    var root_entry = _emit_value_to_doc(doc, positions, pos_idx, doc_start, n)

    var consumed_end: Int
    if pos_idx > 0:
        consumed_end = Int(positions[pos_idx - 1]) + 1
    else:
        consumed_end = _primitive_end(doc.input.as_bytes(), doc_start, n)

    var bytes2 = doc.input.as_bytes()
    while consumed_end < n:
        if not _is_ws(bytes2[consumed_end]):
            raise Error(
                "Stage 2: trailing content after top-level JSON value at"
                " offset "
                + String(consumed_end)
            )
        consumed_end += 1

    # The root is the last appended entry.
    doc.tape.append(root_entry)
    return doc^


def _emit_value_to_doc(
    mut doc: Document,
    mut positions: List[UInt32],
    mut pos_idx: Int,
    start: Int,
    end: Int,
) raises -> UInt64:
    """Emit a single value's tape `header` entry. May write descendants
    of this value into `doc.tape` as a side effect, but does NOT write
    this value's own header (the caller does that)."""
    var bytes = doc.input.as_bytes()
    var i = _skip_ws_input(doc.input, start, end)
    if i >= end:
        raise Error("Stage 2: empty value")

    var c = bytes[i]
    if c == UInt8(ord("{")):
        return _emit_object_to_doc(doc, positions, pos_idx, i, end)
    if c == UInt8(ord("[")):
        return _emit_array_to_doc(doc, positions, pos_idx, i, end)
    if c == UInt8(ord('"')):
        return _emit_string_to_doc(doc, positions, pos_idx, i)
    if c == UInt8(ord("n")):
        return pack_tape_entry(TAPE_TAG_NULL, 0)
    if c == UInt8(ord("t")):
        return pack_tape_entry(TAPE_TAG_BOOL, 1)
    if c == UInt8(ord("f")):
        return pack_tape_entry(TAPE_TAG_BOOL, 0)
    if c == UInt8(ord("-")) or (c >= UInt8(ord("0")) and c <= UInt8(ord("9"))):
        return _emit_number_to_doc(doc, i, end)

    raise Error("Stage 2: unexpected character at offset " + String(i))


def _emit_string_to_doc(
    mut doc: Document,
    mut positions: List[UInt32],
    mut pos_idx: Int,
    open_quote: Int,
) raises -> UInt64:
    """Same string parsing logic as `_parse_string`, but emits a
    STRING (clean, zero-copy slice into input) or STRING_OWNED
    (post-unescape copy in `string_pool`) tape header."""
    if pos_idx >= len(positions) or Int(positions[pos_idx]) != open_quote:
        raise Error(
            "Stage 2: cursor desync at string open offset " + String(open_quote)
        )
    pos_idx += 1
    if pos_idx >= len(positions):
        raise Error(
            "Stage 2: unterminated string at offset " + String(open_quote)
        )
    var close_quote = Int(positions[pos_idx])
    pos_idx += 1

    var bytes = doc.input.as_bytes()
    var start_idx = open_quote + 1
    var end_idx = close_quote

    var has_escape = False
    for j in range(start_idx, end_idx):
        if bytes[j] == UInt8(ord("\\")):
            has_escape = True
            break

    if not has_escape:
        # Zero-copy: payload is (offset, length) into doc.input.
        return pack_tape_entry(
            TAPE_TAG_STRING,
            pack_pair(UInt64(start_idx), UInt64(end_idx - start_idx)),
        )

    _validate_escapes(bytes, start_idx, end_idx)

    # Span-based unescape so we don't copy the entire input for every
    # string with escapes; the unescape walker only needs the bytes
    # between the quotes plus the unchanged context for surrogate
    # pair lookahead.
    var unescaped = unescape_json_string_span(bytes, start_idx, end_idx)
    var s = String(unsafe_from_utf8=unescaped^)
    var pool_idx = len(doc.string_pool)
    doc.string_pool.append(s^)
    return pack_tape_entry(TAPE_TAG_STRING_OWNED, UInt64(pool_idx))


def _emit_number_to_doc(
    mut doc: Document, start: Int, end: Int
) raises -> UInt64:
    """Same number parsing logic as `_parse_number`. Inlines small ints
    in the 60-bit payload; large ints and floats spill to side pools.

    Integers are parsed inline by `_parse_int_inline`, which walks the
    byte span and accumulates a UInt64 without allocating a String.
    Floats still need a String for `atof` (we don't ship a Lemire
    implementation yet), but the integer fast path avoids that cost
    on the most common case.
    """
    var bytes = doc.input.as_bytes()
    var i = start
    var is_float = False
    if bytes[i] == UInt8(ord("-")):
        i += 1

    if (
        i < end
        and bytes[i] == UInt8(ord("0"))
        and i + 1 < end
        and bytes[i + 1] >= UInt8(ord("0"))
        and bytes[i + 1] <= UInt8(ord("9"))
    ):
        raise Error(
            "Stage 2: leading zeros are not allowed in JSON numbers (offset "
            + String(start)
            + ")"
        )

    while i < end:
        var c = bytes[i]
        if c >= UInt8(ord("0")) and c <= UInt8(ord("9")):
            i += 1
            continue
        if c == UInt8(ord(".")) or c == UInt8(ord("e")) or c == UInt8(ord("E")):
            is_float = True
            i += 1
            continue
        if c == UInt8(ord("+")) or c == UInt8(ord("-")):
            i += 1
            continue
        break

    if is_float:
        # Float fast path still needs a String for atof, but the
        # span is the just the number, not the whole input.
        var num_str = String(unsafe_from_utf8=bytes[start:i])
        var pool_idx = len(doc.float_pool)
        doc.float_pool.append(atof(num_str))
        return pack_tape_entry(TAPE_TAG_FLOAT, UInt64(pool_idx))
    # Integer: parse inline (no String allocation, no atol call).
    var v = _parse_int_inline(bytes, start, i)
    var payload = UInt64(v) & ((UInt64(1) << 60) - 1)
    return pack_tape_entry(TAPE_TAG_INT, payload)


@always_inline
def _parse_int_inline(bytes: Span[UInt8, _], start: Int, end: Int) -> Int64:
    """SWAR-friendly signed integer parser.

    Walks `bytes[start:end]` byte-by-byte and accumulates the value in
    a `UInt64`. Caller has already validated that the substring is
    `-?[0-9]+`, so we don't re-check digit ranges. The Mojo compiler
    optimises this tight loop into something close to an explicit
    SWAR sequence for short numbers (1-9 digits), which is the common
    case in JSON; integers spilling above 60 bits are rejected by the
    payload mask in the caller.
    """
    var i = start
    var negative = False
    if i < end and bytes[i] == UInt8(ord("-")):
        negative = True
        i += 1
    var result: UInt64 = 0
    while i < end:
        result = result * 10 + UInt64(bytes[i]) - UInt64(ord("0"))
        i += 1
    if negative:
        return -Int64(result)
    return Int64(result)


def _emit_array_to_doc(
    mut doc: Document,
    mut positions: List[UInt32],
    mut pos_idx: Int,
    open_offset: Int,
    end: Int,
) raises -> UInt64:
    """Same validation logic as `_parse_array`, but recursively emits
    child headers into a contiguous block in `doc.tape` and returns the
    parent ARRAY entry packed with (count, child_start_idx)."""
    var input = doc.input
    if pos_idx >= len(positions) or Int(positions[pos_idx]) != open_offset:
        raise Error("Stage 2: cursor desync at array open")
    var open_idx = pos_idx
    pos_idx += 1

    var close = _find_matching_close(
        input, positions, open_idx, UInt8(ord("[")), UInt8(ord("]"))
    )
    var close_idx = close.close_idx
    var close_offset = close.close_offset

    var bytes = input.as_bytes()
    var depth = 0
    var last_top_comma_byte = -1
    var prev_top_comma_byte: Int

    # Validation pass: same checks as _parse_array. We only count and
    # validate; child emission happens in the second walk below so that
    # we can collect child byte ranges and recurse.
    var count = 0
    var k = open_idx + 1
    while k < close_idx:
        var off = Int(positions[k])
        var b = bytes[off]
        if b == UInt8(ord("[")) or b == UInt8(ord("{")):
            depth += 1
        elif b == UInt8(ord("]")) or b == UInt8(ord("}")):
            depth -= 1
        elif b == UInt8(ord(",")) and depth == 0:
            count += 1
            prev_top_comma_byte = last_top_comma_byte
            last_top_comma_byte = off
            if prev_top_comma_byte >= 0:
                var has_between = False
                for j in range(prev_top_comma_byte + 1, off):
                    if not _is_ws(bytes[j]):
                        has_between = True
                        break
                if not has_between:
                    raise Error(
                        "Stage 2: empty element between commas in array"
                        " at offset "
                        + String(off)
                    )
            else:
                var has_first = False
                for j in range(open_offset + 1, off):
                    if not _is_ws(bytes[j]):
                        has_first = True
                        break
                if not has_first:
                    raise Error(
                        "Stage 2: leading comma in array at offset "
                        + String(off)
                    )
        elif b == UInt8(ord('"')):
            k += 1
        k += 1

    var has_content = False
    for j in range(open_offset + 1, close_offset):
        if not _is_ws(bytes[j]):
            has_content = True
            break
    if has_content:
        count += 1

    if last_top_comma_byte >= 0:
        var has_value_after_comma = False
        for j in range(last_top_comma_byte + 1, close_offset):
            if not _is_ws(bytes[j]):
                has_value_after_comma = True
                break
        if not has_value_after_comma:
            raise Error(
                "Stage 2: trailing comma in array at offset "
                + String(last_top_comma_byte)
            )

    # Emit each child by recursing on its byte range. We re-walk the
    # structural index to find each top-level child boundary; on the
    # way, `_emit_value_to_doc` itself advances pos_idx for
    # strings / arrays / objects but leaves it alone for primitives
    # (null / true / false / numbers). We track pos_idx before/after
    # each call to choose between "use the last consumed structural"
    # and "scan bytes for the primitive's end".
    var headers = List[UInt64](capacity=count)
    if count > 0:
        var child_byte = open_offset + 1
        while child_byte < close_offset and _is_ws(bytes[child_byte]):
            child_byte += 1
        var i = 0
        while i < count:
            var pos_before = pos_idx
            var h = _emit_value_to_doc(
                doc, positions, pos_idx, child_byte, close_offset
            )
            headers.append(h)
            var consumed_end: Int
            if pos_idx > pos_before:
                consumed_end = Int(positions[pos_idx - 1]) + 1
            else:
                consumed_end = _primitive_end(bytes, child_byte, close_offset)
            # Skip ws + comma + ws to land on the next child.
            var j = consumed_end
            while j < close_offset and _is_ws(bytes[j]):
                j += 1
            if j < close_offset and bytes[j] == UInt8(ord(",")):
                # Keep pos_idx aligned with the structural index by
                # advancing past this comma if it's the next entry.
                if pos_idx < len(positions) and Int(positions[pos_idx]) == j:
                    pos_idx += 1
                j += 1
            while j < close_offset and _is_ws(bytes[j]):
                j += 1
            child_byte = j
            i += 1

    var child_start = len(doc.tape)
    for j in range(len(headers)):
        doc.tape.append(headers[j])

    pos_idx = close_idx + 1
    return pack_tape_entry(
        TAPE_TAG_ARRAY,
        pack_pair(UInt64(count), UInt64(child_start)),
    )


def _emit_object_to_doc(
    mut doc: Document,
    mut positions: List[UInt32],
    mut pos_idx: Int,
    open_offset: Int,
    end: Int,
) raises -> UInt64:
    """Same validation logic as `_parse_object`, but emits a
    contiguous block of (KEY, VALUE) header entries and returns the
    parent OBJECT entry packed with (pair_count, child_start_idx)."""
    var input = doc.input
    if pos_idx >= len(positions) or Int(positions[pos_idx]) != open_offset:
        raise Error("Stage 2: cursor desync at object open")
    var open_idx = pos_idx
    pos_idx += 1

    var close = _find_matching_close(
        input, positions, open_idx, UInt8(ord("{")), UInt8(ord("}"))
    )
    var close_idx = close.close_idx
    var close_offset = close.close_offset

    var bytes = input.as_bytes()

    # Reuse the v0.1 _parse_object validation walk to collect keys
    # AND surface every malformed-object error before we touch the
    # tape. We then re-walk to recurse into each value.
    var keys = List[String]()
    var depth = 0
    var expect_key = True
    var last_top_colon_byte = -1
    var last_top_comma_byte = -1
    var top_comma_count = 0
    var saw_colon_after_key = True
    var k = open_idx + 1
    while k < close_idx:
        var off = Int(positions[k])
        var b = bytes[off]

        if b == UInt8(ord("{")) or b == UInt8(ord("[")):
            depth += 1
            k += 1
            continue
        if b == UInt8(ord("}")) or b == UInt8(ord("]")):
            depth -= 1
            k += 1
            continue
        if b == UInt8(ord(",")) and depth == 0:
            expect_key = True
            saw_colon_after_key = True
            last_top_comma_byte = off
            top_comma_count += 1
            k += 1
            continue
        if b == UInt8(ord(":")) and depth == 0:
            last_top_colon_byte = off
            saw_colon_after_key = True
            k += 1
            continue
        if b == UInt8(ord('"')):
            if depth == 0 and expect_key:
                if k + 1 >= close_idx:
                    raise Error("Stage 2: malformed object key")
                var close_quote = Int(positions[k + 1])
                var key_start = off + 1
                var key_len = close_quote - key_start
                var key_bytes = List[UInt8](capacity=key_len)
                key_bytes.resize(key_len, 0)
                memcpy(
                    dest=key_bytes.unsafe_ptr(),
                    src=bytes.unsafe_ptr() + key_start,
                    count=key_len,
                )
                keys.append(String(unsafe_from_utf8=key_bytes^))
                expect_key = False
                saw_colon_after_key = False
                k += 2
                continue
            if depth == 0 and not saw_colon_after_key:
                raise Error(
                    "Stage 2: missing ':' between key and value at offset "
                    + String(off)
                )
            k += 2
            continue

        k += 1

    var has_content = False
    for j in range(open_offset + 1, close_offset):
        if not _is_ws(bytes[j]):
            has_content = True
            break
    if has_content:
        var expected_keys = top_comma_count + 1
        if len(keys) != expected_keys:
            raise Error(
                "Stage 2: expected "
                + String(expected_keys)
                + " keys, got "
                + String(len(keys))
                + " (likely unquoted or missing key in object)"
            )

    if last_top_comma_byte >= 0:
        var has_kv_after_comma = False
        for j in range(last_top_comma_byte + 1, close_offset):
            if not _is_ws(bytes[j]):
                has_kv_after_comma = True
                break
        if not has_kv_after_comma:
            raise Error(
                "Stage 2: trailing comma in object at offset "
                + String(last_top_comma_byte)
            )

    if last_top_colon_byte >= 0:
        var has_value_after_colon = False
        var stop = close_offset
        if last_top_comma_byte > last_top_colon_byte:
            stop = last_top_comma_byte
        for j in range(last_top_colon_byte + 1, stop):
            if not _is_ws(bytes[j]):
                has_value_after_colon = True
                break
        if not has_value_after_colon:
            raise Error(
                "Stage 2: missing value after ':' at offset "
                + String(last_top_colon_byte)
            )

    # Second walk: recurse into each value and collect (key_header,
    # value_header) pairs, then write them contiguously.
    var pair_count = len(keys)
    var headers = List[UInt64](capacity=2 * pair_count)
    if pair_count > 0:
        var i = 0
        var cursor = open_offset + 1
        while i < pair_count:
            while cursor < close_offset and _is_ws(bytes[cursor]):
                cursor += 1
            # The key is a quoted string starting at `cursor`. Stage 1
            # emitted both quote positions; advance pos_idx past them.
            if (
                pos_idx + 1 >= len(positions)
                or Int(positions[pos_idx]) != cursor
            ):
                raise Error("Stage 2: cursor desync at object key")
            var key_close = Int(positions[pos_idx + 1])
            pos_idx += 2
            # Intern the key into key_pool.
            var key_pool_idx = len(doc.key_pool)
            doc.key_pool.append(keys[i])
            var key_header = pack_tape_entry(TAPE_TAG_KEY, UInt64(key_pool_idx))
            # The next structural is the `:` at byte c2. Advance
            # pos_idx past it before recursing into the value.
            var c2 = key_close + 1
            while c2 < close_offset and bytes[c2] != UInt8(ord(":")):
                c2 += 1
            if pos_idx < len(positions) and Int(positions[pos_idx]) == c2:
                pos_idx += 1
            var value_start = c2 + 1
            var pos_before = pos_idx
            var value_header = _emit_value_to_doc(
                doc, positions, pos_idx, value_start, close_offset
            )
            headers.append(key_header)
            headers.append(value_header)
            # Compute where this value ends in bytes -- pos_idx may or
            # may not have advanced (advances for string / array /
            # object children; not for primitives).
            var value_end: Int
            if pos_idx > pos_before:
                value_end = Int(positions[pos_idx - 1]) + 1
            else:
                value_end = _primitive_end(bytes, value_start, close_offset)
            # Skip ws + comma + ws to the next key.
            var j = value_end
            while j < close_offset and _is_ws(bytes[j]):
                j += 1
            if j < close_offset and bytes[j] == UInt8(ord(",")):
                if pos_idx < len(positions) and Int(positions[pos_idx]) == j:
                    pos_idx += 1
                j += 1
            cursor = j
            i += 1

    var child_start = len(doc.tape)
    for j in range(len(headers)):
        doc.tape.append(headers[j])

    pos_idx = close_idx + 1
    return pack_tape_entry(
        TAPE_TAG_OBJECT,
        pack_pair(UInt64(pair_count), UInt64(child_start)),
    )
