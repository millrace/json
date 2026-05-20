# json - Tape-backed Document representation.
#
# This module introduces the v0.2 storage layout that arrays and objects will
# eventually be read from in O(1) per step. In Phase A the type is added and
# tested independently; later phases (B copy-on-write, C SIMD parser, D GPU
# adapter) progressively wire reads, mutations, and parsers through it.
#
# Tape entry layout (64 bits)
# ---------------------------
# Bits 60-63: 4-bit type tag.
# Bits 0-59 : 60-bit payload (interpretation depends on tag).
#
# Tag layouts:
#   NULL    (0): payload unused.
#   BOOL    (1): payload low bit = 0/1.
#   INT     (2): payload = sign-extended 60-bit signed integer
#                (callers must ensure |value| < 2^59; values that overflow
#                are spilled into Document.int_pool in later phases).
#   FLOAT   (3): payload = 30-bit offset into Document.float_pool.
#   STRING  (4): payload = (30-bit input offset || 30-bit length).
#   ARRAY   (5): payload = (30-bit element_count || 30-bit child_start_idx).
#   OBJECT  (6): payload = (30-bit pair_count || 30-bit child_start_idx).
#
# For an OBJECT, the children at child_start_idx are stored as alternating
# (key, value) entries: the key entry is a STRING whose payload is an offset
# into Document.key_pool (rather than into Document.input), so unescaped
# keys can be cached. The value entry sits at child_start_idx + 1, and the
# next pair starts at child_start_idx + 2, etc.
#
# For an ARRAY, the children at child_start_idx are stored as a contiguous
# run of entries, one per element.

from std.collections import List


# ---------------------------------------------------------------------------
# Tag constants
# ---------------------------------------------------------------------------

comptime TAPE_TAG_NULL: UInt8 = 0
comptime TAPE_TAG_BOOL: UInt8 = 1
comptime TAPE_TAG_INT: UInt8 = 2
comptime TAPE_TAG_FLOAT: UInt8 = 3
comptime TAPE_TAG_STRING: UInt8 = 4
comptime TAPE_TAG_ARRAY: UInt8 = 5
comptime TAPE_TAG_OBJECT: UInt8 = 6
# Tag 7 is reserved for object KEY entries that point into key_pool instead
# of input. Distinguishing keys at the tag level is convenient when walking
# a tape that mixes keys and values (e.g., when iterating object_items).
comptime TAPE_TAG_KEY: UInt8 = 7


# ---------------------------------------------------------------------------
# Bit packing
# ---------------------------------------------------------------------------

comptime _PAYLOAD_MASK: UInt64 = (UInt64(1) << 60) - 1
comptime _OFFSET_MASK: UInt64 = (UInt64(1) << 30) - 1


@always_inline
def pack_tape_entry(tag: UInt8, payload: UInt64) -> UInt64:
    """Pack a 4-bit tag with a 60-bit payload into one tape entry."""
    return (UInt64(tag) << 60) | (payload & _PAYLOAD_MASK)


@always_inline
def tape_tag(entry: UInt64) -> UInt8:
    """Extract the 4-bit type tag from a tape entry."""
    return UInt8((entry >> 60) & 0xF)


@always_inline
def tape_payload(entry: UInt64) -> UInt64:
    """Extract the 60-bit payload from a tape entry."""
    return entry & _PAYLOAD_MASK


@always_inline
def pack_pair(hi30: UInt64, lo30: UInt64) -> UInt64:
    """Pack two 30-bit unsigned values into a 60-bit payload."""
    return ((hi30 & _OFFSET_MASK) << 30) | (lo30 & _OFFSET_MASK)


@always_inline
def payload_hi30(payload: UInt64) -> Int:
    """Return the high 30 bits of a payload as an Int."""
    return Int((payload >> 30) & _OFFSET_MASK)


@always_inline
def payload_lo30(payload: UInt64) -> Int:
    """Return the low 30 bits of a payload as an Int."""
    return Int(payload & _OFFSET_MASK)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


struct Document(Copyable, Movable):
    """Owns the JSON input bytes and a packed tape of entries.

    A `Document` is the storage unit produced by parsers. A `Value` view
    references a tape entry inside a `Document` by index; when a `Value`
    is mutated, the document is materialized into an owned tree (Phase B).

    Side pools:
      - `key_pool` stores unescaped object keys. Object KEY entries store a
        pool offset rather than an offset into `input`, so unescaped keys
        survive even when the original bytes are released.
      - `float_pool` stores `Float64` values, since 64-bit IEEE 754 doesn't
        fit in 60 bits.
      - `int_pool` is reserved for spilled large integers; in Phase A all
        integers are inlined as 60-bit signed payloads.
    """

    var input: String
    var tape: List[UInt64]
    var key_pool: List[String]
    var float_pool: List[Float64]
    var int_pool: List[Int64]

    def __init__(out self):
        self.input = String()
        self.tape = List[UInt64]()
        self.key_pool = List[String]()
        self.float_pool = List[Float64]()
        self.int_pool = List[Int64]()

    def __init__(out self, var input: String):
        self.input = input^
        self.tape = List[UInt64]()
        self.key_pool = List[String]()
        self.float_pool = List[Float64]()
        self.int_pool = List[Int64]()

    def copy(self) -> Self:
        var d = Self()
        d.input = self.input
        d.tape = self.tape.copy()
        d.key_pool = self.key_pool.copy()
        d.float_pool = self.float_pool.copy()
        d.int_pool = self.int_pool.copy()
        return d^

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def append_null(mut self) -> Int:
        var idx = len(self.tape)
        self.tape.append(pack_tape_entry(TAPE_TAG_NULL, 0))
        return idx

    def append_bool(mut self, b: Bool) -> Int:
        var idx = len(self.tape)
        var payload: UInt64 = 1 if b else 0
        self.tape.append(pack_tape_entry(TAPE_TAG_BOOL, payload))
        return idx

    def append_int(mut self, value: Int64) -> Int:
        var idx = len(self.tape)
        # Two's-complement encoding into the low 60 bits.
        var payload = UInt64(value) & _PAYLOAD_MASK
        self.tape.append(pack_tape_entry(TAPE_TAG_INT, payload))
        return idx

    def append_float(mut self, value: Float64) -> Int:
        var pool_idx = len(self.float_pool)
        self.float_pool.append(value)
        var idx = len(self.tape)
        self.tape.append(pack_tape_entry(TAPE_TAG_FLOAT, UInt64(pool_idx)))
        return idx

    def append_string(mut self, offset: Int, length: Int) -> Int:
        """Append a STRING entry referencing bytes [offset:offset+length] of input.
        """
        var idx = len(self.tape)
        self.tape.append(
            pack_tape_entry(
                TAPE_TAG_STRING,
                pack_pair(UInt64(offset), UInt64(length)),
            )
        )
        return idx

    def append_key(mut self, var key: String) -> Int:
        """Intern a key into key_pool and append a KEY entry referencing it."""
        var pool_idx = len(self.key_pool)
        self.key_pool.append(key^)
        var idx = len(self.tape)
        self.tape.append(pack_tape_entry(TAPE_TAG_KEY, UInt64(pool_idx)))
        return idx

    def append_array(mut self, count: Int, child_start_idx: Int) -> Int:
        var idx = len(self.tape)
        self.tape.append(
            pack_tape_entry(
                TAPE_TAG_ARRAY,
                pack_pair(UInt64(count), UInt64(child_start_idx)),
            )
        )
        return idx

    def append_object(mut self, pair_count: Int, child_start_idx: Int) -> Int:
        var idx = len(self.tape)
        self.tape.append(
            pack_tape_entry(
                TAPE_TAG_OBJECT,
                pack_pair(UInt64(pair_count), UInt64(child_start_idx)),
            )
        )
        return idx

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    @always_inline
    def get_tag(self, tape_idx: Int) -> UInt8:
        return tape_tag(self.tape[tape_idx])

    @always_inline
    def get_payload(self, tape_idx: Int) -> UInt64:
        return tape_payload(self.tape[tape_idx])

    def get_bool(self, tape_idx: Int) -> Bool:
        return self.get_payload(tape_idx) == 1

    def get_int(self, tape_idx: Int) -> Int64:
        var payload = self.get_payload(tape_idx)
        # Sign-extend from 60 bits.
        var sign_bit = (payload >> 59) & 1
        if sign_bit == 1:
            return Int64(payload | (UInt64(0xF) << 60))
        return Int64(payload)

    def get_float(self, tape_idx: Int) -> Float64:
        var pool_idx = Int(self.get_payload(tape_idx))
        return self.float_pool[pool_idx]

    def get_string_offset(self, tape_idx: Int) -> Int:
        return payload_hi30(self.get_payload(tape_idx))

    def get_string_length(self, tape_idx: Int) -> Int:
        return payload_lo30(self.get_payload(tape_idx))

    def get_string(self, tape_idx: Int) -> String:
        """Materialize a STRING entry as a fresh `String`.

        The returned string is a fresh copy of the slice of `input` that the
        STRING entry references. This is safe even if `input` is later
        consumed.
        """
        var offset = self.get_string_offset(tape_idx)
        var length = self.get_string_length(tape_idx)
        return String(
            unsafe_from_utf8=self.input.as_bytes()[offset : offset + length]
        )

    def get_key(self, tape_idx: Int) -> String:
        """Return the unescaped key for a KEY entry."""
        var pool_idx = Int(self.get_payload(tape_idx))
        return self.key_pool[pool_idx]

    def get_count(self, tape_idx: Int) -> Int:
        """Element count for ARRAY, pair count for OBJECT."""
        return payload_hi30(self.get_payload(tape_idx))

    def get_child_start(self, tape_idx: Int) -> Int:
        """Tape index where this container's children start."""
        return payload_lo30(self.get_payload(tape_idx))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def root(self) -> Int:
        """Tape index of the root entry. Defined as the LAST entry, since
        builders write children before their parent so the parent can record
        a backwards-pointing child_start_idx."""
        return len(self.tape) - 1

    def size(self) -> Int:
        return len(self.tape)
