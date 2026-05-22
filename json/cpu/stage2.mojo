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
#
# Phase 1 perf rewrite (v0.2-F)
# -----------------------------
# The previous walker had three hot-path costs that scaled poorly on
# real-world JSON (citm_catalog, twitter):
#
#   1. `positions.copy()` at entry. On citm that's a 117k-entry
#      List[UInt32] copy on every parse. Now we read `index.positions`
#      in place via an immutable borrow.
#   2. `_find_matching_close` per container -- a forward bracket-counting
#      scan over `positions`. For deeply-nested or wide objects this is
#      O(structural_count * container_count). Replaced with a forward
#      walk that discovers the close by inspecting the next position's
#      byte (`positions` is monotonically increasing, so the next
#      `]` / `}` for the current container *is* the next position whose
#      byte is `]` / `}` after we've fully consumed the children).
#   3. Per-container `headers: List[UInt64]`. Each container allocated
#      its own header scratch and copied into `doc.tape` at close. We
#      now share a single `headers_scratch: List[UInt64]` across all
#      recursive calls; each container records `headers_lo = len(scratch)`
#      on entry, appends its children's headers as they're parsed, then
#      at close copies `scratch[headers_lo:]` to `doc.tape` and shrinks
#      the scratch back to `headers_lo`. One allocation amortized over
#      the whole document instead of one per container.
#
# Validation rules (trailing commas, double commas, leading commas,
# missing colons, missing values after colons, unquoted keys) all still
# raise structured errors during the walk; coverage in
# `tests/test_stage2_tape.mojo`.

from std.bit import count_trailing_zeros
from std.collections import List
from std.memory import memcpy
from std.memory.unsafe import pack_bits

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
    TAPE_TAG_KEY_INLINE,
)
from .stage1_scalar import StructuralIndex
from .number_parse import parse_int_swar


# ---------------------------------------------------------------------------
# Whitespace + primitive end helpers
# ---------------------------------------------------------------------------


@always_inline
def _string_has_escape(bytes: Span[UInt8, _], start: Int, end: Int) -> Bool:
    """SIMD-accelerated scan for backslash in [start, end).

    The dominant case on real JSON corpora is "no escape", so we scan
    32 bytes at a time using `SIMD.eq` + `reduce_or` and only fall
    back to a scalar loop for the tail (< 32 bytes) and to confirm
    position when we *do* find one.
    """
    var n = end - start
    if n <= 0:
        return False
    var ptr = bytes.unsafe_ptr()
    var i = start
    var stop = end - 32
    while i <= stop:
        var chunk = ptr.load[width=32](i)
        if chunk.eq(UInt8(ord("\\"))).reduce_or():
            return True
        i += 32
    while i < end:
        if ptr[i] == UInt8(ord("\\")):
            return True
        i += 1
    return False


@always_inline
def _is_ws(b: UInt8) -> Bool:
    return (
        b == UInt8(ord(" "))
        or b == UInt8(ord("\t"))
        or b == UInt8(ord("\n"))
        or b == UInt8(ord("\r"))
    )


@always_inline
def _skip_ws(bytes: Span[UInt8, _], start: Int, end: Int) -> Int:
    """Skip ASCII whitespace.

    Branch-prediction-friendly hybrid:
      * The first ~4 bytes are checked scalar (the dense-JSON case
        where there is 0-1 whitespace bytes between fields - no SIMD
        setup cost).
      * Only if we're still in whitespace after that do we fall into
        the SIMD 32-byte loop, which uses pack_bits +
        count_trailing_zeros to find the first non-whitespace byte
        without per-byte iteration.

    Pretty-printed corpora (citm: ~9 ws bytes/pair) hit the SIMD path;
    compact corpora pay only a single scalar byte test.
    """
    var i = start
    var ptr = bytes.unsafe_ptr()

    # Scalar prelude (up to 4 bytes). Common case for compact / dense
    # JSON: returns after the first byte test.
    var scalar_stop = min(end, start + 4)
    while i < scalar_stop and _is_ws(ptr[i]):
        i += 1
    if i < scalar_stop:
        return i

    # SIMD body for long whitespace runs.
    var stop = end - 32
    while i <= stop:
        var chunk = ptr.load[width=32](i)
        var is_ws_mask = (
            chunk.eq(UInt8(ord(" ")))
            | chunk.eq(UInt8(ord("\t")))
            | chunk.eq(UInt8(ord("\n")))
            | chunk.eq(UInt8(ord("\r")))
        )
        var ws_bits = pack_bits[dtype=DType.uint32](is_ws_mask)
        if ws_bits != UInt32(0xFFFF_FFFF):
            var first_non_ws = Int(count_trailing_zeros(~ws_bits))
            return i + first_non_ws
        i += 32

    while i < end and _is_ws(ptr[i]):
        i += 1
    return i


def _primitive_end(bytes: Span[UInt8, _], start: Int, end: Int) -> Int:
    """First byte after a top-level primitive (number / null / true /
    false). Skips leading whitespace then scans until whitespace, EOF,
    or a structural byte."""
    var i = _skip_ws(bytes, start, end)
    while i < end:
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
            benchmark corpora SIMD is ~1.2x faster than the scalar
            walker. When True, use the scalar oracle -- useful for
            differential testing and for inputs small enough that the
            SIMD chunk loop never runs (n < 32). Both produce identical
            output (enforced by `tests/test_stage1_equivalence.mojo`);
            this is purely a performance switch.

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
# Tape-emitting walker
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
# ---------------------------------------------------------------------------


def parse_into_document(
    var input: String, index: StructuralIndex
) raises -> Document:
    """Build a `Document` (tape + side pools) by walking the structural
    index over `input`. The returned document owns `input`.

    Args:
        input: Original JSON bytes.
        index: Output of stage 1. Borrowed immutably -- `positions` is
            read in place, no copy on entry.

    Returns:
        Owned `Document` whose root entry is the last tape slot.
    """
    var n = input.byte_length()
    ref positions = index.positions

    var doc = Document(input^)

    # Pre-size: each structural position contributes ~0.5 - 1 tape
    # entries (open/close/comma/colon are bookkeeping; quote pairs and
    # the primitives between commas are the entries). On twitter and
    # citm `len(tape) / len(positions)` is around 0.55, so reserving
    # `len(positions)` is a slight over-estimate that keeps the tape
    # vector from reallocating mid-walk. The +8 covers the root
    # primitive case (positions can be empty for `42`).
    doc.tape.reserve(len(positions) + 8)
    # Side-pool guesses: cheap to reserve, expensive to grow under
    # contention with the tape append. ~10% of positions for floats /
    # owned strings / keys is a generous overestimate on real corpora.
    var side_hint = len(positions) // 8 + 4
    doc.float_pool.reserve(side_hint)
    doc.string_pool.reserve(side_hint)
    doc.key_pool.reserve(side_hint)

    # Shared scratch for collecting children's headers across all
    # recursive container frames. Each frame uses a contiguous slice
    # `[lo, len(scratch))` and shrinks back to `lo` at close.
    var headers_scratch = List[UInt64]()
    headers_scratch.reserve(64)

    var doc_start = _skip_ws(doc.input.as_bytes(), 0, n)
    var pos_idx = 0
    var value_end = doc_start

    var root_entry = _emit_value(
        doc,
        positions,
        pos_idx,
        headers_scratch,
        value_end,
        doc_start,
        n,
    )

    # Trailing-content check: nothing but whitespace allowed after the
    # top-level value.
    var bytes2 = doc.input.as_bytes()
    var t = value_end
    while t < n:
        if not _is_ws(bytes2[t]):
            raise Error(
                "Stage 2: trailing content after top-level JSON value at"
                " offset "
                + String(t)
            )
        t += 1

    # The root is the last appended entry.
    doc.tape.append(root_entry)
    return doc^


# ---------------------------------------------------------------------------
# Recursive emitters.
#
# Each `_emit_*` function:
#
#   - reads positions[pos_idx..] and bytes starting at `start`,
#   - appends 0 or more descendant headers to `doc.tape` (and side
#     pools) as it parses,
#   - returns this value's own header (NOT yet appended to doc.tape;
#     the caller is responsible for either appending it or stashing
#     it in the shared `headers_scratch`),
#   - advances `pos_idx` and `value_end` so the caller knows how far
#     into the input it consumed without re-scanning.
# ---------------------------------------------------------------------------


def _emit_value(
    mut doc: Document,
    positions: List[UInt32],
    mut pos_idx: Int,
    mut headers_scratch: List[UInt64],
    mut value_end: Int,
    start: Int,
    n: Int,
) raises -> UInt64:
    """Emit a single value's header. May write descendants into
    `doc.tape` as a side effect. Sets `value_end` to the byte offset
    just past the parsed value (or just past its closing
    bracket / quote)."""
    var bytes = doc.input.as_bytes()
    var i = _skip_ws(bytes, start, n)
    if i >= n:
        raise Error("Stage 2: empty value")

    var c = bytes[i]
    if c == UInt8(ord("{")):
        return _emit_object(
            doc, positions, pos_idx, headers_scratch, value_end, i, n
        )
    if c == UInt8(ord("[")):
        return _emit_array(
            doc, positions, pos_idx, headers_scratch, value_end, i, n
        )
    if c == UInt8(ord('"')):
        return _emit_string(doc, positions, pos_idx, value_end, i)
    if c == UInt8(ord("n")):
        # null literal -- 4 bytes, no positions consumed.
        if (
            i + 4 > n
            or bytes[i + 1] != UInt8(ord("u"))
            or bytes[i + 2] != UInt8(ord("l"))
            or bytes[i + 3] != UInt8(ord("l"))
        ):
            raise Error(
                "Stage 2: expected 'null' literal at offset " + String(i)
            )
        value_end = i + 4
        return pack_tape_entry(TAPE_TAG_NULL, 0)
    if c == UInt8(ord("t")):
        if (
            i + 4 > n
            or bytes[i + 1] != UInt8(ord("r"))
            or bytes[i + 2] != UInt8(ord("u"))
            or bytes[i + 3] != UInt8(ord("e"))
        ):
            raise Error(
                "Stage 2: expected 'true' literal at offset " + String(i)
            )
        value_end = i + 4
        return pack_tape_entry(TAPE_TAG_BOOL, 1)
    if c == UInt8(ord("f")):
        if (
            i + 5 > n
            or bytes[i + 1] != UInt8(ord("a"))
            or bytes[i + 2] != UInt8(ord("l"))
            or bytes[i + 3] != UInt8(ord("s"))
            or bytes[i + 4] != UInt8(ord("e"))
        ):
            raise Error(
                "Stage 2: expected 'false' literal at offset " + String(i)
            )
        value_end = i + 5
        return pack_tape_entry(TAPE_TAG_BOOL, 0)
    if c == UInt8(ord("-")) or (c >= UInt8(ord("0")) and c <= UInt8(ord("9"))):
        return _emit_number(doc, value_end, i, n)

    raise Error("Stage 2: unexpected character at offset " + String(i))


def _emit_string(
    mut doc: Document,
    positions: List[UInt32],
    mut pos_idx: Int,
    mut value_end: Int,
    open_quote: Int,
) raises -> UInt64:
    """Same string-parsing logic as before but reads positions in
    place. Emits a STRING (clean, zero-copy slice into input) or
    STRING_OWNED (post-unescape copy in `string_pool`) header."""
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
    value_end = close_quote + 1

    var start_idx = open_quote + 1
    var end_idx = close_quote

    var bytes = doc.input.as_bytes()
    var has_escape = _string_has_escape(bytes, start_idx, end_idx)

    if not has_escape:
        return pack_tape_entry(
            TAPE_TAG_STRING,
            pack_pair(UInt64(start_idx), UInt64(end_idx - start_idx)),
        )

    _validate_escapes(bytes, start_idx, end_idx)

    var unescaped = unescape_json_string_span(bytes, start_idx, end_idx)
    var s = String(unsafe_from_utf8=unescaped^)
    var pool_idx = len(doc.string_pool)
    doc.string_pool.append(s^)
    return pack_tape_entry(TAPE_TAG_STRING_OWNED, UInt64(pool_idx))


def _emit_number(
    mut doc: Document,
    mut value_end: Int,
    start: Int,
    n: Int,
) raises -> UInt64:
    """Same number-parsing logic as `_parse_number`. Inlines small
    ints in the 60-bit payload; large ints and floats spill to side
    pools."""
    var bytes = doc.input.as_bytes()
    var i = start
    var is_float = False
    if bytes[i] == UInt8(ord("-")):
        i += 1

    if (
        i < n
        and bytes[i] == UInt8(ord("0"))
        and i + 1 < n
        and bytes[i + 1] >= UInt8(ord("0"))
        and bytes[i + 1] <= UInt8(ord("9"))
    ):
        raise Error(
            "Stage 2: leading zeros are not allowed in JSON numbers (offset "
            + String(start)
            + ")"
        )

    while i < n:
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

    value_end = i

    if is_float:
        var num_str = String(unsafe_from_utf8=bytes[start:i])
        var pool_idx = len(doc.float_pool)
        doc.float_pool.append(atof(num_str))
        return pack_tape_entry(TAPE_TAG_FLOAT, UInt64(pool_idx))
    var v = parse_int_swar(bytes, start, i)
    var payload = UInt64(v) & ((UInt64(1) << 60) - 1)
    return pack_tape_entry(TAPE_TAG_INT, payload)


def _emit_array(
    mut doc: Document,
    positions: List[UInt32],
    mut pos_idx: Int,
    mut headers_scratch: List[UInt64],
    mut value_end: Int,
    open_offset: Int,
    n: Int,
) raises -> UInt64:
    """Emit an ARRAY header by walking the structural index forward.

    No bracket-counting pre-pass: we recurse into each child via
    `_emit_value`, and after the child returns we inspect the next
    structural position. The next position whose byte is `]` is
    necessarily the close for *this* array (because any `]` belonging
    to a nested array has already been consumed by the recursive
    `_emit_array` for that nested container).

    Children's headers go through `headers_scratch` -- a single shared
    list across all recursive frames. We record `headers_lo` on entry,
    each child appends its header to `headers_scratch`, and at close
    we copy `headers_scratch[headers_lo:]` into `doc.tape` and shrink
    the scratch back to `headers_lo`.

    Validation rules: leading comma, trailing comma, double comma,
    unmatched / missing close all raise structured errors.
    """
    if pos_idx >= len(positions) or Int(positions[pos_idx]) != open_offset:
        raise Error("Stage 2: cursor desync at array open")
    pos_idx += 1

    var headers_lo = len(headers_scratch)

    var cursor = _skip_ws(doc.input.as_bytes(), open_offset + 1, n)

    # Empty array fast path.
    var bytes_ref = doc.input.as_bytes()
    if (
        pos_idx < len(positions)
        and Int(positions[pos_idx]) == cursor
        and bytes_ref[cursor] == UInt8(ord("]"))
    ):
        pos_idx += 1
        value_end = cursor + 1
        var child_start = len(doc.tape)
        return pack_tape_entry(
            TAPE_TAG_ARRAY,
            pack_pair(UInt64(0), UInt64(child_start)),
        )

    if cursor < n and bytes_ref[cursor] == UInt8(ord(",")):
        raise Error(
            "Stage 2: leading comma in array at offset " + String(cursor)
        )
    if cursor >= n:
        raise Error("Stage 2: unterminated array")

    while True:
        var child_end = cursor
        var h = _emit_value(
            doc,
            positions,
            pos_idx,
            headers_scratch,
            child_end,
            cursor,
            n,
        )
        headers_scratch.append(h)

        var bytes = doc.input.as_bytes()
        var j = _skip_ws(bytes, child_end, n)
        if j >= n:
            raise Error("Stage 2: unterminated array")

        var b = bytes[j]
        if b == UInt8(ord("]")):
            # Sanity check: this ']' must be the next structural pos.
            if pos_idx >= len(positions) or Int(positions[pos_idx]) != j:
                raise Error("Stage 2: cursor desync at array close")
            pos_idx += 1
            value_end = j + 1
            break

        if b != UInt8(ord(",")):
            raise Error(
                "Stage 2: expected ',' or ']' in array at offset " + String(j)
            )
        # Consume the comma.
        if pos_idx >= len(positions) or Int(positions[pos_idx]) != j:
            raise Error("Stage 2: cursor desync at array comma")
        pos_idx += 1

        var next_cursor = _skip_ws(bytes, j + 1, n)
        if next_cursor >= n:
            raise Error(
                "Stage 2: trailing comma in array at offset " + String(j)
            )
        var nb = bytes[next_cursor]
        if nb == UInt8(ord("]")):
            raise Error(
                "Stage 2: trailing comma in array at offset " + String(j)
            )
        if nb == UInt8(ord(",")):
            raise Error(
                "Stage 2: empty element between commas in array at offset "
                + String(next_cursor)
            )
        cursor = next_cursor

    # Flush this frame's children to doc.tape via memcpy and shrink
    # the scratch.
    var count = len(headers_scratch) - headers_lo
    var child_start = len(doc.tape)
    if count > 0:
        doc.tape.resize(child_start + count, 0)
        memcpy(
            dest=doc.tape.unsafe_ptr() + child_start,
            src=headers_scratch.unsafe_ptr() + headers_lo,
            count=count,
        )
    headers_scratch.shrink(headers_lo)

    return pack_tape_entry(
        TAPE_TAG_ARRAY,
        pack_pair(UInt64(count), UInt64(child_start)),
    )


def _emit_object(
    mut doc: Document,
    positions: List[UInt32],
    mut pos_idx: Int,
    mut headers_scratch: List[UInt64],
    mut value_end: Int,
    open_offset: Int,
    n: Int,
) raises -> UInt64:
    """Emit an OBJECT header. Same single-pass-forward design as
    `_emit_array`, but each iteration parses a (KEY, VALUE) pair.
    KEY is interned into `doc.key_pool`."""
    if pos_idx >= len(positions) or Int(positions[pos_idx]) != open_offset:
        raise Error("Stage 2: cursor desync at object open")
    pos_idx += 1

    var headers_lo = len(headers_scratch)

    var cursor = _skip_ws(doc.input.as_bytes(), open_offset + 1, n)

    # Empty object fast path.
    var bytes_ref = doc.input.as_bytes()
    if (
        pos_idx < len(positions)
        and Int(positions[pos_idx]) == cursor
        and bytes_ref[cursor] == UInt8(ord("}"))
    ):
        pos_idx += 1
        value_end = cursor + 1
        var child_start = len(doc.tape)
        return pack_tape_entry(
            TAPE_TAG_OBJECT,
            pack_pair(UInt64(0), UInt64(child_start)),
        )

    if cursor < n and bytes_ref[cursor] == UInt8(ord(",")):
        raise Error(
            "Stage 2: leading comma in object at offset " + String(cursor)
        )
    if cursor >= n:
        raise Error("Stage 2: unterminated object")

    while True:
        var after_key: Int
        var value_start: Int
        var key_header: UInt64

        # Local scope for `bytes` borrow so we can mutate doc later.
        var bytes = doc.input.as_bytes()
        if bytes[cursor] != UInt8(ord('"')):
            raise Error(
                "Stage 2: expected string key at offset " + String(cursor)
            )

        # Stage 1 emits both quote positions for the key.
        if pos_idx + 1 >= len(positions) or Int(positions[pos_idx]) != cursor:
            raise Error("Stage 2: cursor desync at object key")
        var key_close = Int(positions[pos_idx + 1])
        pos_idx += 2

        var key_start = cursor + 1
        var key_len = key_close - key_start
        var has_escape = _string_has_escape(bytes, key_start, key_close)

        if has_escape:
            # Slow path: keys with escapes still need allocation +
            # interning. They're rare on real JSON corpora.
            _validate_escapes(bytes, key_start, key_close)
            var unesc = unescape_json_string_span(bytes, key_start, key_close)
            var key = String(unsafe_from_utf8=unesc^)
            var key_pool_idx = len(doc.key_pool)
            doc.key_pool.append(key^)
            key_header = pack_tape_entry(TAPE_TAG_KEY, UInt64(key_pool_idx))
        else:
            # Fast path: zero-copy KEY_INLINE -- store (offset, length)
            # into input. No allocation, no memcpy, no key_pool append.
            key_header = pack_tape_entry(
                TAPE_TAG_KEY_INLINE,
                pack_pair(UInt64(key_start), UInt64(key_len)),
            )

        # Find the colon while we still have the bytes borrow.
        after_key = _skip_ws(bytes, key_close + 1, n)
        if after_key >= n or bytes[after_key] != UInt8(ord(":")):
            raise Error(
                "Stage 2: missing ':' between key and value at offset "
                + String(after_key)
            )
        value_start = _skip_ws(bytes, after_key + 1, n)
        if value_start >= n:
            raise Error(
                "Stage 2: missing value after ':' at offset "
                + String(after_key)
            )

        headers_scratch.append(key_header)

        if pos_idx >= len(positions) or Int(positions[pos_idx]) != after_key:
            raise Error("Stage 2: cursor desync at object colon")
        pos_idx += 1

        var v_end = value_start
        var value_header = _emit_value(
            doc,
            positions,
            pos_idx,
            headers_scratch,
            v_end,
            value_start,
            n,
        )
        headers_scratch.append(value_header)

        var bytes2 = doc.input.as_bytes()
        var j = _skip_ws(bytes2, v_end, n)
        if j >= n:
            raise Error("Stage 2: unterminated object")

        var b = bytes2[j]
        if b == UInt8(ord("}")):
            if pos_idx >= len(positions) or Int(positions[pos_idx]) != j:
                raise Error("Stage 2: cursor desync at object close")
            pos_idx += 1
            value_end = j + 1
            break

        if b != UInt8(ord(",")):
            raise Error(
                "Stage 2: expected ',' or '}' in object at offset " + String(j)
            )
        if pos_idx >= len(positions) or Int(positions[pos_idx]) != j:
            raise Error("Stage 2: cursor desync at object comma")
        pos_idx += 1

        var next_cursor = _skip_ws(bytes2, j + 1, n)
        if next_cursor >= n:
            raise Error(
                "Stage 2: trailing comma in object at offset " + String(j)
            )
        if bytes2[next_cursor] == UInt8(ord("}")):
            raise Error(
                "Stage 2: trailing comma in object at offset " + String(j)
            )
        cursor = next_cursor

    var total = len(headers_scratch) - headers_lo
    var pair_count = total // 2
    var child_start = len(doc.tape)
    if total > 0:
        doc.tape.resize(child_start + total, 0)
        memcpy(
            dest=doc.tape.unsafe_ptr() + child_start,
            src=headers_scratch.unsafe_ptr() + headers_lo,
            count=total,
        )
    headers_scratch.shrink(headers_lo)

    return pack_tape_entry(
        TAPE_TAG_OBJECT,
        pack_pair(UInt64(pair_count), UInt64(child_start)),
    )
