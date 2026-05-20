# json - Raw-string helpers for arrays/objects.
#
# These functions parse and surgically rewrite JSON arrays/objects stored
# as raw substrings on `Value`. They are PURE STRING operations -- they
# do NOT depend on the `Value` type, which keeps the import graph
# acyclic across the `value/` package.
#
# Phase B replaces the mutation surgery (`_update_*`, `_add_*`,
# `_append_*`) with copy-on-write through `OwnedValue`, leaving only
# the read-side `_extract_*` helpers used for `raw_json()` round-trip
# and for `LazyValue`.

from std.collections import List


# ---------------------------------------------------------------------------
# Read-side helpers (used by Value reads, LazyValue, JSON Pointer)
# ---------------------------------------------------------------------------


def _extract_field_value(raw: String, key: String) raises -> String:
    """Extract a field's value from raw JSON object string.

    Args:
        raw: Raw JSON object string (e.g., '{"a": 1, "b": "hello"}').
        key: Field name to extract.

    Returns:
        The raw JSON value as a string (e.g., '1' or '"hello"').
    """
    var raw_bytes = raw.as_bytes()
    var in_string = False
    var i = 0
    var n = len(raw_bytes)

    while i < n and (
        raw_bytes[i] == UInt8(ord("{"))
        or raw_bytes[i] == UInt8(ord(" "))
        or raw_bytes[i] == UInt8(ord("\t"))
        or raw_bytes[i] == UInt8(ord("\n"))
    ):
        i += 1

    while i < n:
        while i < n and (
            raw_bytes[i] == UInt8(ord(" "))
            or raw_bytes[i] == UInt8(ord("\t"))
            or raw_bytes[i] == UInt8(ord("\n"))
        ):
            i += 1

        if i >= n:
            break

        if raw_bytes[i] == UInt8(ord('"')) and not in_string:
            i += 1
            var key_start = i

            while i < n and raw_bytes[i] != UInt8(ord('"')):
                if raw_bytes[i] == UInt8(ord("\\")):
                    i += 2
                else:
                    i += 1

            var found_key = String(unsafe_from_utf8=raw.as_bytes()[key_start:i])
            i += 1

            while i < n and (
                raw_bytes[i] == UInt8(ord(" "))
                or raw_bytes[i] == UInt8(ord("\t"))
                or raw_bytes[i] == UInt8(ord("\n"))
                or raw_bytes[i] == UInt8(ord(":"))
            ):
                i += 1

            if found_key == key:
                return _extract_json_value(raw, i)
            else:
                _ = _extract_json_value(raw, i)
                while (
                    i < n
                    and raw_bytes[i] != UInt8(ord(","))
                    and raw_bytes[i] != UInt8(ord("}"))
                ):
                    i += 1
                if i < n and raw_bytes[i] == UInt8(ord(",")):
                    i += 1
        else:
            i += 1

    raise Error("Key not found in JSON object")


def _extract_json_value(raw: String, start: Int) raises -> String:
    """Extract a single JSON value starting at position start."""
    var raw_bytes = raw.as_bytes()
    var i = start
    var n = len(raw_bytes)

    while i < n and (
        raw_bytes[i] == UInt8(ord(" "))
        or raw_bytes[i] == UInt8(ord("\t"))
        or raw_bytes[i] == UInt8(ord("\n"))
    ):
        i += 1

    if i >= n:
        raise Error("Unexpected end of JSON")

    var first_char = raw_bytes[i]

    if first_char == UInt8(ord('"')):
        var value_start = i
        i += 1
        while i < n:
            if raw_bytes[i] == UInt8(ord("\\")):
                i += 2
                continue
            elif raw_bytes[i] == UInt8(ord('"')):
                return String(
                    String(unsafe_from_utf8=raw.as_bytes()[value_start : i + 1])
                )
            else:
                i += 1
        raise Error("Unterminated string")

    elif first_char == UInt8(ord("{")) or first_char == UInt8(ord("[")):
        var close_char = UInt8(ord("}")) if first_char == UInt8(
            ord("{")
        ) else UInt8(ord("]"))
        var depth = 1
        var value_start = i
        i += 1
        var in_string = False

        while i < n and depth > 0:
            if raw_bytes[i] == UInt8(ord("\\")) and in_string:
                i += 2
                continue
            elif raw_bytes[i] == UInt8(ord('"')):
                in_string = not in_string
            elif not in_string:
                if raw_bytes[i] == first_char:
                    depth += 1
                elif raw_bytes[i] == close_char:
                    depth -= 1
            i += 1

        return String(String(unsafe_from_utf8=raw.as_bytes()[value_start:i]))

    else:
        var value_start = i
        while (
            i < n
            and raw_bytes[i] != UInt8(ord(","))
            and raw_bytes[i] != UInt8(ord("}"))
            and raw_bytes[i] != UInt8(ord("]"))
            and raw_bytes[i] != UInt8(ord(" "))
            and raw_bytes[i] != UInt8(ord("\t"))
            and raw_bytes[i] != UInt8(ord("\n"))
        ):
            i += 1
        return String(String(unsafe_from_utf8=raw.as_bytes()[value_start:i]))


def _extract_array_element(raw: String, index: Int) raises -> String:
    """Extract an array element by index from raw JSON array string."""
    var raw_bytes = raw.as_bytes()
    var n = len(raw_bytes)
    var i = 0
    var current_index = 0
    var depth = 0

    while i < n and (
        raw_bytes[i] == UInt8(ord("["))
        or raw_bytes[i] == UInt8(ord(" "))
        or raw_bytes[i] == UInt8(ord("\t"))
        or raw_bytes[i] == UInt8(ord("\n"))
        or raw_bytes[i] == UInt8(ord("\r"))
    ):
        if raw_bytes[i] == UInt8(ord("[")):
            depth = 1
        i += 1

    if depth == 0:
        raise Error("Invalid JSON array")

    while i < n:
        while i < n and (
            raw_bytes[i] == UInt8(ord(" "))
            or raw_bytes[i] == UInt8(ord("\t"))
            or raw_bytes[i] == UInt8(ord("\n"))
            or raw_bytes[i] == UInt8(ord("\r"))
        ):
            i += 1

        if i >= n:
            break

        if raw_bytes[i] == UInt8(ord("]")):
            break

        if current_index == index:
            return _extract_json_value(raw, i)

        _ = _extract_json_value(raw, i)

        var element_depth = 0
        var in_string = False
        var escaped = False

        while i < n:
            var c = raw_bytes[i]
            if escaped:
                escaped = False
                i += 1
                continue
            if c == UInt8(ord("\\")) and in_string:
                escaped = True
                i += 1
                continue
            if c == UInt8(ord('"')):
                in_string = not in_string
                i += 1
                continue
            if in_string:
                i += 1
                continue
            if c == UInt8(ord("[")) or c == UInt8(ord("{")):
                element_depth += 1
            elif c == UInt8(ord("]")) or c == UInt8(ord("}")):
                if element_depth > 0:
                    element_depth -= 1
                else:
                    break
            elif c == UInt8(ord(",")) and element_depth == 0:
                i += 1
                current_index += 1
                break
            i += 1

        if i >= n or raw_bytes[i] == UInt8(ord("]")):
            break

    raise Error("Array index out of bounds: " + String(index))


def _count_array_elements(raw: String) -> Int:
    """Count elements in a JSON array."""
    var raw_bytes = raw.as_bytes()
    var n = len(raw_bytes)
    var count = 0
    var depth = 0
    var in_string = False
    var escaped = False

    for i in range(n):
        var c = raw_bytes[i]
        if escaped:
            escaped = False
            continue
        if c == UInt8(ord("\\")):
            escaped = True
            continue
        if c == UInt8(ord('"')):
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == UInt8(ord("[")) or c == UInt8(ord("{")):
            depth += 1
        elif c == UInt8(ord("]")) or c == UInt8(ord("}")):
            depth -= 1
        elif c == UInt8(ord(",")) and depth == 1:
            count += 1

    var has_content = False
    depth = 0
    in_string = False
    for i in range(n):
        var c = raw_bytes[i]
        if c == UInt8(ord("[")):
            depth += 1
        elif c == UInt8(ord("]")):
            depth -= 1
        elif c == UInt8(ord('"')):
            if depth == 1 and not in_string:
                has_content = True
            in_string = not in_string
        elif (
            depth == 1
            and not in_string
            and c != UInt8(ord(" "))
            and c != UInt8(ord("\t"))
            and c != UInt8(ord("\n"))
            and c != UInt8(ord("\r"))
        ):
            has_content = True

    if has_content:
        count += 1

    return count


def _extract_object_keys(raw: String) -> List[String]:
    """Extract all keys from a JSON object."""
    var keys = List[String]()
    var raw_bytes = raw.as_bytes()
    var n = len(raw_bytes)
    var depth = 0
    var in_string = False
    var escaped = False
    var key_start = -1
    var expect_key = True

    for i in range(n):
        var c = raw_bytes[i]
        if escaped:
            escaped = False
            continue
        if c == UInt8(ord("\\")):
            escaped = True
            continue
        if c == UInt8(ord('"')):
            if not in_string:
                in_string = True
                if depth == 1 and expect_key:
                    key_start = i + 1
            else:
                in_string = False
                if key_start >= 0 and depth == 1:
                    _ = i - key_start
                    keys.append(
                        String(
                            String(unsafe_from_utf8=raw.as_bytes()[key_start:i])
                        )
                    )
                    key_start = -1
            continue
        if in_string:
            continue
        if c == UInt8(ord("{")) or c == UInt8(ord("[")):
            depth += 1
        elif c == UInt8(ord("}")) or c == UInt8(ord("]")):
            depth -= 1
        elif c == UInt8(ord(":")) and depth == 1:
            expect_key = False
        elif c == UInt8(ord(",")) and depth == 1:
            expect_key = True

    return keys^


# ---------------------------------------------------------------------------
# Mutation-side helpers (raw-string surgery; replaced by COW in Phase B)
# ---------------------------------------------------------------------------


def _update_object_value(raw: String, key: String, new_value: String) -> String:
    """Update a value in a JSON object."""
    var raw_bytes = raw.as_bytes()
    var n = len(raw_bytes)
    var i = 0

    while i < n and raw_bytes[i] != UInt8(ord("{")):
        i += 1
    i += 1
    var depth = 1

    while i < n:
        while i < n and (
            raw_bytes[i] == UInt8(ord(" "))
            or raw_bytes[i] == UInt8(ord("\t"))
            or raw_bytes[i] == UInt8(ord("\n"))
        ):
            i += 1

        if i >= n:
            break

        if raw_bytes[i] == UInt8(ord('"')) and depth == 1:
            var key_start = i + 1
            i += 1
            while i < n and raw_bytes[i] != UInt8(ord('"')):
                if raw_bytes[i] == UInt8(ord("\\")):
                    i += 2
                else:
                    i += 1

            var found_key = String(unsafe_from_utf8=raw.as_bytes()[key_start:i])
            i += 1

            while i < n and raw_bytes[i] != UInt8(ord(":")):
                i += 1
            i += 1

            while i < n and (
                raw_bytes[i] == UInt8(ord(" "))
                or raw_bytes[i] == UInt8(ord("\t"))
                or raw_bytes[i] == UInt8(ord("\n"))
            ):
                i += 1

            if found_key == key:
                var value_start = i
                var value_end = _find_value_end_str(raw, i)
                return (
                    String(unsafe_from_utf8=raw.as_bytes()[:value_start])
                    + new_value
                    + String(unsafe_from_utf8=raw.as_bytes()[value_end:])
                )

        i += 1

    return raw


def _find_value_end_str(raw: String, start: Int) -> Int:
    """Find the end of a JSON value starting at start."""
    var raw_bytes = raw.as_bytes()
    var n = len(raw_bytes)
    var i = start
    var depth = 0
    var in_string = False
    var escaped = False

    while i < n:
        var c = raw_bytes[i]

        if escaped:
            escaped = False
            i += 1
            continue

        if c == UInt8(ord("\\")) and in_string:
            escaped = True
            i += 1
            continue

        if c == UInt8(ord('"')):
            in_string = not in_string
            i += 1
            continue

        if in_string:
            i += 1
            continue

        if c == UInt8(ord("{")) or c == UInt8(ord("[")):
            depth += 1
        elif c == UInt8(ord("}")) or c == UInt8(ord("]")):
            if depth > 0:
                depth -= 1
            else:
                return i
        elif c == UInt8(ord(",")) and depth == 0:
            return i

        i += 1

    return i


def _add_object_key(raw: String, key: String, value: String) -> String:
    """Add a new key-value pair to a JSON object."""
    var raw_bytes = raw.as_bytes()
    var n = len(raw_bytes)

    var close_pos = n - 1
    while close_pos >= 0 and raw_bytes[close_pos] != UInt8(ord("}")):
        close_pos -= 1

    if close_pos < 0:
        return raw

    var is_empty = True
    for i in range(1, close_pos):
        var c = raw_bytes[i]
        if (
            c != UInt8(ord(" "))
            and c != UInt8(ord("\t"))
            and c != UInt8(ord("\n"))
            and c != UInt8(ord("\r"))
        ):
            is_empty = False
            break

    if is_empty:
        return '{"' + key + '":' + value + "}"
    else:
        return (
            String(unsafe_from_utf8=raw.as_bytes()[:close_pos])
            + ',"'
            + key
            + '":'
            + value
            + "}"
        )


def _update_array_element(raw: String, index: Int, new_value: String) -> String:
    """Update an element in a JSON array."""
    var raw_bytes = raw.as_bytes()
    var n = len(raw_bytes)
    var current_index = 0
    var i = 0

    while i < n and raw_bytes[i] != UInt8(ord("[")):
        i += 1
    i += 1

    while i < n and (
        raw_bytes[i] == UInt8(ord(" "))
        or raw_bytes[i] == UInt8(ord("\t"))
        or raw_bytes[i] == UInt8(ord("\n"))
        or raw_bytes[i] == UInt8(ord("\r"))
    ):
        i += 1

    while i < n:
        if current_index == index:
            var value_start = i
            var value_end = _find_value_end_str(raw, i)
            return (
                String(unsafe_from_utf8=raw.as_bytes()[:value_start])
                + new_value
                + String(unsafe_from_utf8=raw.as_bytes()[value_end:])
            )

        var elem_end = _find_value_end_str(raw, i)
        i = elem_end

        while i < n and (
            raw_bytes[i] == UInt8(ord(","))
            or raw_bytes[i] == UInt8(ord(" "))
            or raw_bytes[i] == UInt8(ord("\t"))
            or raw_bytes[i] == UInt8(ord("\n"))
            or raw_bytes[i] == UInt8(ord("\r"))
        ):
            if raw_bytes[i] == UInt8(ord(",")):
                current_index += 1
            i += 1

        if i >= n or raw_bytes[i] == UInt8(ord("]")):
            break

    return raw


def _append_to_array(raw: String, value: String) -> String:
    """Append a value to a JSON array."""
    var raw_bytes = raw.as_bytes()
    var n = len(raw_bytes)

    var close_pos = n - 1
    while close_pos >= 0 and raw_bytes[close_pos] != UInt8(ord("]")):
        close_pos -= 1

    if close_pos < 0:
        return raw

    var is_empty = True
    for i in range(1, close_pos):
        var c = raw_bytes[i]
        if (
            c != UInt8(ord(" "))
            and c != UInt8(ord("\t"))
            and c != UInt8(ord("\n"))
            and c != UInt8(ord("\r"))
        ):
            is_empty = False
            break

    if is_empty:
        return "[" + value + "]"
    else:
        return (
            String(unsafe_from_utf8=raw.as_bytes()[:close_pos])
            + ","
            + value
            + "]"
        )


def _parse_json_pointer(pointer: String) raises -> List[String]:
    """Parse a JSON Pointer string into tokens.

    Handles RFC 6901 escape sequences:
        ~0 -> ~.
        ~1 -> /.

    Lives in `raw_ops.mojo` because it has no dependency on `Value` (it
    only inspects the pointer string and returns tokens).
    """
    var tokens = List[String]()
    var pointer_bytes = pointer.as_bytes()
    var n = len(pointer_bytes)
    var i = 1  # Skip leading /

    while i < n:
        var token = String()
        while i < n and pointer_bytes[i] != UInt8(ord("/")):
            if pointer_bytes[i] == UInt8(ord("~")):
                if i + 1 < n:
                    if pointer_bytes[i + 1] == UInt8(ord("0")):
                        token += "~"
                        i += 2
                        continue
                    elif pointer_bytes[i + 1] == UInt8(ord("1")):
                        token += "/"
                        i += 2
                        continue
                raise Error("Invalid escape sequence in JSON Pointer")
            token += chr(Int(pointer_bytes[i]))
            i += 1
        tokens.append(token^)
        i += 1  # Skip /

    return tokens^
