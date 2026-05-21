# SIMD stage 1: structural-index builder using `pack_bits`.
#
# This is the SIMD counterpart of `stage1_scalar.parse_structural_scalar`.
# It scans the input in 32-byte chunks, classifies each byte as
# structural / quote / backslash / other via SIMD comparisons, then uses
# `pack_bits` to convert the comparison masks into 32-bit bitmaps so we
# can iterate match positions with `count_trailing_zeros`.
#
# String/escape state -------------------------------------------------------
#
# JSON's "is this byte inside a string?" rule depends on a byte-by-byte
# escape-state machine: `\"` does not close a string, `\\"` does, etc.
# A true SIMD stage 1 uses a carry-less multiply (CLMUL) trick to turn
# this into a SIMD-friendly prefix XOR. Mojo does not yet expose CLMUL,
# so we fall back to a per-chunk scalar scan over the quote / backslash
# positions only -- still much cheaper than the byte-by-byte scalar
# version because we touch only those positions.
#
# Important subtlety: the marker walk only iterates positions that hit
# `{ } [ ] : , \ "`. An escape consumes the byte at `bslash + 1`, which
# may NOT be a marker -- it could be `n`, `t`, `u`, an arbitrary text
# byte, etc. Carrying `escaped` to the next *marker* (regardless of
# distance) is therefore wrong: a non-marker byte between the backslash
# and the next marker would already have consumed the escape silently.
# We track the absolute position of the backslash that set the escape
# and only honor the escape when the next visited position is exactly
# `bslash_pos + 1`. The same rule applies across chunk boundaries.
#
# Performance ---------------------------------------------------------------
#
# On the benchmark corpora this SIMD scan is the better default:
#
#   twitter.json  (616 KB) : scalar 0.60 GB/s, SIMD 1.24 GB/s (2.07x)
#   citm_catalog.json (1.7 MB) : scalar 0.64 GB/s, SIMD 1.38 GB/s (2.17x)
#   twitter_large_record.json (804 MB) : scalar 0.51 GB/s, SIMD 0.75 GB/s (1.46x)
#
# `parse_two_pass` therefore defaults to SIMD; the scalar oracle stays
# in `stage1_scalar.mojo` for differential testing and for inputs
# smaller than one SIMD chunk where the chunk loop never runs.

from std.collections import List
from std.bit import count_trailing_zeros
from std.memory.unsafe import pack_bits

from .stage1_scalar import StructuralIndex


# ---------------------------------------------------------------------------
# SIMD parameters
# ---------------------------------------------------------------------------


comptime SIMD_WIDTH: Int = 32
"""Bytes processed per SIMD iteration. Picked to match common AVX2 / NEON
register sizes; `pack_bits` produces a 32-bit mask we iterate with
`count_trailing_zeros`."""


# ---------------------------------------------------------------------------
# parse_structural_simd
# ---------------------------------------------------------------------------


def parse_structural_simd(input: String) -> StructuralIndex:
    """Build a structural index using a SIMD scan.

    Output is byte-for-byte identical to
    `stage1_scalar.parse_structural_scalar(input)` -- the equivalence is
    enforced by `tests/test_stage1_equivalence.mojo`, including a
    full-document run against the benchmark corpora.

    Faster than the scalar walker by 1.5x to 2.2x on the benchmark
    corpora; this is the default stage 1 used by `parse_two_pass`.
    """
    var bytes = input.as_bytes()
    var n = len(bytes)
    var index = StructuralIndex(capacity=n // 4)

    var in_string = False
    # Absolute position of the most recent backslash that has not yet
    # had its escape consumed. -1 means "no pending escape". The byte
    # actually escaped is bslash_pos + 1, regardless of whether it is
    # a marker or a non-marker.
    var bslash_pos: Int = -1
    var i = 0

    comptime W = SIMD[DType.uint8, SIMD_WIDTH]

    while i + SIMD_WIDTH <= n:
        var chunk = bytes.unsafe_ptr().load[width=SIMD_WIDTH](i)

        # Mask of bytes equal to a "structural-or-string" marker --
        # `{` `}` `[` `]` `:` `,` `\\` `"`. We need the backslashes here
        # so the per-chunk scalar resolver can advance escape state.
        var lbrace = chunk.eq(W(UInt8(ord("{"))))
        var rbrace = chunk.eq(W(UInt8(ord("}"))))
        var lbrack = chunk.eq(W(UInt8(ord("["))))
        var rbrack = chunk.eq(W(UInt8(ord("]"))))
        var colon = chunk.eq(W(UInt8(ord(":"))))
        var comma = chunk.eq(W(UInt8(ord(","))))
        var quote = chunk.eq(W(UInt8(ord('"'))))
        var bslash = chunk.eq(W(UInt8(ord("\\"))))

        var struct_no_q = lbrace | rbrace | lbrack | rbrack | colon | comma
        var any_marker = struct_no_q | quote | bslash

        var struct_mask = pack_bits[dtype=DType.uint32](struct_no_q)
        var quote_mask = pack_bits[dtype=DType.uint32](quote)
        var bslash_mask = pack_bits[dtype=DType.uint32](bslash)
        var any_mask = pack_bits[dtype=DType.uint32](any_marker)

        # Fast path: no markers in this chunk and we're not inside a
        # string. Any pending escape that pointed inside this chunk is
        # silently consumed by a non-marker byte; pending escapes that
        # point past the end of this chunk (only possible if the prior
        # chunk ended with `\\` at its last byte) survive into the next
        # chunk.
        if any_mask == 0 and not in_string:
            if bslash_pos >= 0 and bslash_pos < i + SIMD_WIDTH - 1:
                bslash_pos = -1
            i += SIMD_WIDTH
            continue

        # Walk just the marker positions inside this chunk; the rest of
        # the bytes are uninteresting for stage 1.
        var local = any_mask
        while local != 0:
            var bit = Int(count_trailing_zeros(local))
            var pos = i + bit
            local &= local - 1  # Clear the bit we just visited.

            # Resolve any pending escape against this exact position.
            # Three sub-cases:
            #   pos == bslash_pos + 1  -> this marker IS the escaped
            #                             byte; skip it and clear the
            #                             pending escape.
            #   pos >  bslash_pos + 1  -> the escape was consumed by a
            #                             non-marker byte that lives
            #                             between the backslash and
            #                             this marker; treat the
            #                             current marker normally.
            #   pos <  bslash_pos + 1  -> impossible: we iterate in
            #                             order and bslash_pos is set
            #                             from an earlier marker.
            if bslash_pos >= 0:
                if pos == bslash_pos + 1:
                    bslash_pos = -1
                    continue
                bslash_pos = -1

            var bit_mask = UInt32(1) << UInt32(bit)
            var is_quote = (quote_mask & bit_mask) != 0
            var is_bslash = (bslash_mask & bit_mask) != 0
            var is_struct = (struct_mask & bit_mask) != 0

            if in_string:
                if is_bslash:
                    bslash_pos = pos
                    continue
                if is_quote:
                    index.positions.append(UInt32(pos))
                    in_string = False
                continue

            # Outside a string.
            if is_quote:
                index.positions.append(UInt32(pos))
                in_string = True
                continue
            if is_struct:
                index.positions.append(UInt32(pos))

        # End of chunk. If a backslash was set during this chunk and
        # the byte it would escape is still inside this chunk, that
        # byte was a non-marker (otherwise we'd have visited it above
        # and cleared bslash_pos). Drop the pending escape so we don't
        # mis-fire it on the next chunk's first marker.
        if bslash_pos >= 0 and bslash_pos < i + SIMD_WIDTH - 1:
            bslash_pos = -1

        i += SIMD_WIDTH

    # Tail: handle bytes that didn't fit in the last 32-byte chunk
    # using the byte-by-byte algorithm (reuse the same logic as
    # `parse_structural_scalar`'s inner loop).
    var escaped = bslash_pos >= 0 and bslash_pos == i - 1
    while i < n:
        var c = bytes[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if in_string:
            if c == UInt8(ord("\\")):
                escaped = True
                i += 1
                continue
            if c == UInt8(ord('"')):
                index.positions.append(UInt32(i))
                in_string = False
                i += 1
                continue
            i += 1
            continue
        if c == UInt8(ord('"')):
            index.positions.append(UInt32(i))
            in_string = True
            i += 1
            continue
        if (
            c == UInt8(ord("{"))
            or c == UInt8(ord("}"))
            or c == UInt8(ord("["))
            or c == UInt8(ord("]"))
            or c == UInt8(ord(":"))
            or c == UInt8(ord(","))
        ):
            index.positions.append(UInt32(i))
        i += 1

    return index^
