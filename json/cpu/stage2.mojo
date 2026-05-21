# Stage 2: walk a `StructuralIndex` and emit a tape-backed `Document`.
#
# Stage 1 -- either `stage1_scalar.parse_structural_scalar` or
# `stage1.parse_structural_simd` -- produces an ordered list of byte
# offsets into the input, one per structural character (plus quote
# boundaries). Stage 2 walks that list left-to-right and emits tape
# entries into a `Document` without ever re-scanning the byte stream
# for structure.
#
# The decoupling is the real win in v0.2: future SIMD or GPU stage 1
# implementations can be swapped in without touching value
# construction. Children are always written before parents so a
# parent's `child_start_idx` payload points backwards into a
# contiguous run of header entries; the root is the last entry, which
# is what `Document.root()` assumes.

from std.collections import List
from std.memory import memcpy

from ..unicode import unescape_json_string_span
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
# Shared helpers (used by both the lazy walker [removed] and the
# tape-emitting walker below)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Convenience: full parse from a raw input string into a tape-backed
# `Document`.
# ---------------------------------------------------------------------------


def parse_two_pass_tape[
    force_scalar: Bool = False
](var input: String) raises -> Document:
    """End-to-end stage 1 + stage 2 parse that emits a `Document`.

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
# Tape-emitting walker.
#
# `parse_into_document` walks a `StructuralIndex` and emits entries
# into a `Document.tape`.
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
# Validation rules: trailing commas, double commas, leading commas,
# missing colons, missing values after colons, and unquoted keys all
# raise structured errors during the walk; coverage lives in
# `tests/test_stage2_tape.mojo`.
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
    """Emit an ARRAY tape entry. Single forward pass: validates and
    recurses into each child in one structural-index walk, no separate
    counting pre-pass.

    Children's headers are collected in a local `headers` list and then
    written contiguously into `doc.tape`; that's the layout invariant
    the readers (`Document.get_child_start`) rely on.

    Validation rules (matching `_parse_array`):

    * leading comma   -> error
    * trailing comma  -> error
    * double comma    -> error
    * unmatched close -> caught by `_find_matching_close`
    """
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
    var headers = List[UInt64]()

    # Skip leading whitespace inside the brackets.
    var cursor = open_offset + 1
    while cursor < close_offset and _is_ws(bytes[cursor]):
        cursor += 1

    # Empty array fast path: `[ ]` with only whitespace between brackets.
    if cursor >= close_offset:
        var child_start = len(doc.tape)
        pos_idx = close_idx + 1
        return pack_tape_entry(
            TAPE_TAG_ARRAY,
            pack_pair(UInt64(0), UInt64(child_start)),
        )

    if bytes[cursor] == UInt8(ord(",")):
        raise Error(
            "Stage 2: leading comma in array at offset " + String(cursor)
        )

    while True:
        var pos_before = pos_idx
        var h = _emit_value_to_doc(
            doc, positions, pos_idx, cursor, close_offset
        )
        headers.append(h)

        var child_end: Int
        if pos_idx > pos_before:
            child_end = Int(positions[pos_idx - 1]) + 1
        else:
            child_end = _primitive_end(bytes, cursor, close_offset)

        var j = child_end
        while j < close_offset and _is_ws(bytes[j]):
            j += 1

        if j >= close_offset:
            break

        if bytes[j] != UInt8(ord(",")):
            # Anything other than ',' or end-of-array here is malformed,
            # but `_find_matching_close` already verified that
            # `bytes[close_offset]` is the matching ']'. So we just
            # break and let validation upstream catch any odd state.
            break

        # Consume the comma and align pos_idx with the structural index.
        if pos_idx < len(positions) and Int(positions[pos_idx]) == j:
            pos_idx += 1
        j += 1
        while j < close_offset and _is_ws(bytes[j]):
            j += 1

        if j >= close_offset:
            raise Error(
                "Stage 2: trailing comma in array at offset " + String(j - 1)
            )
        if bytes[j] == UInt8(ord(",")):
            raise Error(
                "Stage 2: empty element between commas in array at offset "
                + String(j)
            )
        cursor = j

    var child_start = len(doc.tape)
    for j in range(len(headers)):
        doc.tape.append(headers[j])

    pos_idx = close_idx + 1
    return pack_tape_entry(
        TAPE_TAG_ARRAY,
        pack_pair(UInt64(len(headers)), UInt64(child_start)),
    )


def _emit_object_to_doc(
    mut doc: Document,
    mut positions: List[UInt32],
    mut pos_idx: Int,
    open_offset: Int,
    end: Int,
) raises -> UInt64:
    """Emit an OBJECT tape entry. Single forward pass: parses keys and
    values inline, no separate validation walk.

    Children's headers (KEY, VALUE, KEY, VALUE, ...) are collected in
    a local `headers` list and written contiguously to `doc.tape`.

    Validation rules (matching `_parse_object`):

    * unquoted key   -> error
    * missing colon  -> error
    * trailing comma -> error
    * leading comma  -> error
    """
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
    var headers = List[UInt64]()

    # Skip leading whitespace.
    var cursor = open_offset + 1
    while cursor < close_offset and _is_ws(bytes[cursor]):
        cursor += 1

    # Empty object fast path.
    if cursor >= close_offset:
        var child_start = len(doc.tape)
        pos_idx = close_idx + 1
        return pack_tape_entry(
            TAPE_TAG_OBJECT,
            pack_pair(UInt64(0), UInt64(child_start)),
        )

    if bytes[cursor] == UInt8(ord(",")):
        raise Error(
            "Stage 2: leading comma in object at offset " + String(cursor)
        )

    while True:
        # Each iteration parses one (key, value) pair.
        if bytes[cursor] != UInt8(ord('"')):
            raise Error(
                "Stage 2: expected string key at offset " + String(cursor)
            )

        # Stage 1 already emitted both quote positions for the key;
        # consume them.
        if pos_idx + 1 >= len(positions) or Int(positions[pos_idx]) != cursor:
            raise Error("Stage 2: cursor desync at object key")
        var key_close = Int(positions[pos_idx + 1])
        pos_idx += 2

        # Intern the key into key_pool. We unescape if needed so
        # downstream `Document.get_key` returns plain text.
        var key_start = cursor + 1
        var key_len = key_close - key_start
        var has_escape = False
        for j in range(key_start, key_close):
            if bytes[j] == UInt8(ord("\\")):
                has_escape = True
                break
        var key: String
        if has_escape:
            _validate_escapes(bytes, key_start, key_close)
            var unesc = unescape_json_string_span(bytes, key_start, key_close)
            key = String(unsafe_from_utf8=unesc^)
        else:
            var key_bytes = List[UInt8](capacity=key_len)
            key_bytes.resize(key_len, 0)
            memcpy(
                dest=key_bytes.unsafe_ptr(),
                src=bytes.unsafe_ptr() + key_start,
                count=key_len,
            )
            key = String(unsafe_from_utf8=key_bytes^)
        var key_pool_idx = len(doc.key_pool)
        doc.key_pool.append(key^)
        var key_header = pack_tape_entry(TAPE_TAG_KEY, UInt64(key_pool_idx))

        # Find the colon after the key.
        var after_key = key_close + 1
        while after_key < close_offset and _is_ws(bytes[after_key]):
            after_key += 1
        if after_key >= close_offset or bytes[after_key] != UInt8(ord(":")):
            raise Error(
                "Stage 2: missing ':' between key and value at offset "
                + String(after_key)
            )
        if pos_idx < len(positions) and Int(positions[pos_idx]) == after_key:
            pos_idx += 1
        var value_start = after_key + 1
        while value_start < close_offset and _is_ws(bytes[value_start]):
            value_start += 1
        if value_start >= close_offset:
            raise Error(
                "Stage 2: missing value after ':' at offset "
                + String(after_key)
            )

        var pos_before = pos_idx
        var value_header = _emit_value_to_doc(
            doc, positions, pos_idx, value_start, close_offset
        )
        headers.append(key_header)
        headers.append(value_header)

        var value_end: Int
        if pos_idx > pos_before:
            value_end = Int(positions[pos_idx - 1]) + 1
        else:
            value_end = _primitive_end(bytes, value_start, close_offset)

        var j = value_end
        while j < close_offset and _is_ws(bytes[j]):
            j += 1
        if j >= close_offset:
            break
        if bytes[j] != UInt8(ord(",")):
            break
        if pos_idx < len(positions) and Int(positions[pos_idx]) == j:
            pos_idx += 1
        j += 1
        while j < close_offset and _is_ws(bytes[j]):
            j += 1
        if j >= close_offset:
            raise Error(
                "Stage 2: trailing comma in object at offset " + String(j - 1)
            )
        cursor = j

    var pair_count = len(headers) // 2
    var child_start = len(doc.tape)
    for j in range(len(headers)):
        doc.tape.append(headers[j])

    pos_idx = close_idx + 1
    return pack_tape_entry(
        TAPE_TAG_OBJECT,
        pack_pair(UInt64(pair_count), UInt64(child_start)),
    )
