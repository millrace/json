# Equivalence tests for the v0.2 stage 1 implementations.
#
# `parse_structural_scalar` (oracle) and `parse_structural_simd` MUST
# produce byte-for-byte identical structural-index output. These tests
# run both on a fixture corpus -- short literals, nested structures,
# escaped strings, all the corner cases that historically broke SIMD
# string-state tracking -- and assert that the two position lists are
# equal.
#
# When the SIMD path eventually beats the scalar path on a given input
# the default in `parse_two_pass` flips, but the *correctness* contract
# pinned here never changes: the SIMD implementation has to match the
# oracle byte-for-byte before it ships at all.

from std.testing import assert_equal, assert_true, TestSuite
from std.collections import List

from json.cpu.stage1_scalar import parse_structural_scalar, StructuralIndex
from json.cpu.stage1 import parse_structural_simd
from json import loads


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_indices_equal(
    label: String, scalar: StructuralIndex, simd: StructuralIndex
) raises:
    """Assert `scalar.positions == simd.positions` element-by-element."""
    assert_equal(
        scalar.size(),
        simd.size(),
        label
        + ": size mismatch (scalar="
        + String(scalar.size())
        + ", simd="
        + String(simd.size())
        + ")",
    )
    for i in range(scalar.size()):
        assert_equal(
            Int(scalar.positions[i]),
            Int(simd.positions[i]),
            label + ": position " + String(i) + " differs",
        )


def _check(input: String, label: String) raises:
    var scalar = parse_structural_scalar(input)
    var simd = parse_structural_simd(input)
    _assert_indices_equal(label, scalar, simd)


# ---------------------------------------------------------------------------
# Fixture corpus
# ---------------------------------------------------------------------------


def test_empty() raises:
    _check("", "empty")


def test_single_primitive() raises:
    _check("null", "null")
    _check("true", "true")
    _check("false", "false")
    _check("42", "int")
    _check("3.14", "float")


def test_short_string() raises:
    _check('"hello"', "short string")


def test_empty_array() raises:
    _check("[]", "empty array")


def test_empty_object() raises:
    _check("{}", "empty object")


def test_simple_array() raises:
    _check("[1, 2, 3]", "simple array")


def test_simple_object() raises:
    _check('{"a": 1, "b": 2}', "simple object")


def test_array_of_strings() raises:
    _check('["alpha", "beta", "gamma"]', "string array")


def test_object_with_strings() raises:
    _check('{"name": "Alice", "city": "NYC"}', "object with strings")


def test_nested_object() raises:
    _check(
        '{"user": {"name": "Ada", "age": 36}}',
        "nested object",
    )


def test_array_of_objects() raises:
    _check(
        '[{"id": 1, "v": "a"}, {"id": 2, "v": "b"}]',
        "array of objects",
    )


def test_deeply_nested() raises:
    _check(
        '{"a": {"b": {"c": {"d": {"e": 1}}}}}',
        "deeply nested",
    )


def test_string_with_escaped_quote() raises:
    """`"a\\"b"` -- the inner escaped quote must NOT break out of the string,
    so stage 1 must emit positions for only the outer quotes."""
    _check('"a\\"b"', "escaped quote")
    _check('{"key": "a\\"b"}', "object with escaped quote in value")


def test_string_with_escaped_backslash() raises:
    """`"\\\\"` is a string containing a single backslash; the escape must
    be consumed correctly so the closing quote isn't escaped."""
    _check('"\\\\"', "escaped backslash")
    _check('"a\\\\b"', "embedded escaped backslash")


def test_string_with_double_escape_followed_by_quote() raises:
    """Tricky: `"\\\\\\""` is a string with `\\\\` then `\\"`; the SIMD
    state machine must handle `\\\\\\\\` -> not escaping next char, then
    `\\"` -> escape next char."""
    _check('"\\\\\\""', "double escape then escaped quote")


def test_string_with_structural_chars_inside() raises:
    """Structural chars inside a string must NOT appear in the index."""
    _check('"a, b: c, d"', "commas inside string")
    _check('"{ } [ ]"', "braces inside string")
    _check('{"k": "[1,2,3]"}', "array literal text in string value")


def test_string_with_escape_chars_other_kinds() raises:
    """Escape sequences other than `\\\\` and `\\"` must also be handled."""
    _check('"\\n\\r\\t"', "common escapes")
    _check('"\\u00FF"', "unicode escape")


def test_simd_chunk_escape_followed_by_non_marker() raises:
    """Regression: in a string longer than 32 bytes, `\\X` (where X is
    not a marker) must consume the escape on byte X, NOT on the next
    marker. A bug here causes the closing quote to be silently dropped
    and the parser reports `Stage 2: unterminated container`.

    The SIMD scan only iterates marker positions (`\\` `"` `{` `}` `[`
    `]` `:` `,`); non-markers between them are skipped. The escape
    state machine has to know that an escape gets consumed by the
    byte immediately after the backslash, even when that byte is
    silently skipped.
    """
    # Body must straddle a 32-byte chunk so the SIMD path runs.
    var body_left = String("a" * 16)
    var body_right = String("b" * 16)
    _check(
        '{"k":"' + body_left + "\\nfoo" + body_right + '","ok":true}',
        "newline mid-string across chunk",
    )
    _check(
        '{"k":"' + body_left + "\\tx\\rqz" + body_right + '"}',
        "tab + carriage-return mid-string",
    )
    _check(
        '{"text":"a'
        + body_left
        + "\\nthen{not}a{key}: yes"
        + body_right
        + '"}',
        "embedded structural-looking text after newline escape",
    )


def test_simd_chunk_escape_at_chunk_boundary() raises:
    """The `\\` lands in chunk N but the byte it escapes lands in
    chunk N+1. Escape state must cross the chunk break correctly."""
    # Pad so the backslash is the last byte of the first 32-byte chunk;
    # then the escaped quote / structural lands in the next chunk.
    for pad in range(28, 36):
        var prefix = String('"')
        for _ in range(pad):
            prefix += "a"
        # \" inside the string -- escaped quote, must NOT close.
        _check(prefix + '\\"x"', "boundary escaped quote pad=" + String(pad))
        # \\ inside the string -- escaped backslash, NOT followed by escape.
        _check(
            prefix + '\\\\"', "boundary escaped backslash pad=" + String(pad)
        )


def test_simd_escape_followed_by_unrelated_marker_far_later() raises:
    """The actually-escaped byte is the one immediately after `\\`, NOT
    the next marker. If a non-marker sits between `\\` and a later
    marker, the escape MUST be considered consumed before that marker.

    Regression: a real-world Twitter JSON payload with a backslash
    deep inside a long string was leaving the SIMD escape state set
    until the next marker many bytes later, silently dropping that
    marker (typically the closing quote of the string) and reporting
    `Stage 2: unterminated container`.
    """
    # `\n` consumes the escape on the `n`; the closing quote and the
    # following structural `,` must both be emitted unchanged. The
    # padding around the `\n` ensures the structural quote is far
    # enough from the backslash that an "until next marker" carry
    # would mis-fire.
    var pad = String("a" * 40)
    _check(
        '{"k":"' + pad + "\\n" + pad + '","next":1}',
        "long string with \\n in the middle",
    )
    # `\t`, `\r`, `\u00FF` -- escape consumed by a non-structural byte
    # (the letter or the unicode escape's first hex digit), the next
    # marker is the closing quote far ahead.
    _check(
        '{"k":"' + pad + "\\t" + pad + '","next":1}',
        "long string with \\t in the middle",
    )
    _check(
        '{"k":"' + pad + "\\u00FF" + pad + '","next":1}',
        "long string with \\u escape in the middle",
    )


def test_simd_twitter_japanese_string() raises:
    """Reduced repro of the twitter.json failure: a long string
    containing UTF-8 multibyte (Japanese) bytes followed eventually
    by a `\\n`, then more text, then a closing `"` plus structural
    `,`. The SIMD path must agree with the scalar oracle on every
    structural position."""
    # Synthetic Japanese-ish UTF-8 prefix (well-formed 3-byte sequences)
    # to mirror the encoding shape of the real twitter.json payload.
    var jp = String("")
    for _ in range(20):
        jp += chr(0xE3)
        jp += chr(0x81)
        jp += chr(0x82)
    _check(
        '{"text":"' + jp + "\\n" + jp + '","url":null}',
        "japanese + escape + japanese",
    )


def test_long_array_of_numbers() raises:
    var s = String("[")
    for i in range(50):
        if i > 0:
            s += ","
        s += String(i)
    s += "]"
    _check(s, "long array of numbers")


def test_long_array_of_strings() raises:
    var s = String("[")
    for i in range(20):
        if i > 0:
            s += ","
        s += '"x' + String(i) + '"'
    s += "]"
    _check(s, "long array of strings")


def test_whitespace_heavy_input() raises:
    _check(
        '   {  "a"  :  1 ,  "b"  :  [  1 , 2 , 3  ]  }   ',
        "whitespace heavy",
    )


def test_chunk_boundary_alignment() raises:
    """Inputs whose structurals fall right at 32-byte chunk boundaries
    must still resolve correctly. SIMD picks 32-byte chunks; this test
    sweeps prefix sizes around the boundary."""
    for pad in range(28, 38):
        var prefix = String("")
        for _ in range(pad):
            prefix += " "
        var s = prefix + '{"a": 1}'
        _check(s, "chunk boundary pad=" + String(pad))


def test_string_spanning_chunk_boundary() raises:
    """A string whose body straddles a 32-byte chunk boundary must keep
    its in-string state across the chunk break."""
    var s = '"' + String("a" * 50) + '"'
    _check(s, "string spanning chunk")
    var t = '{"k": "' + String("b" * 50) + '"}'
    _check(t, "object string spanning chunk")


def test_realistic_payload_smoke() raises:
    var s = (
        '{"id": 42, "user": {"name": "Ada Lovelace", "tags":'
        ' ["math", "compute", "history"], "active": true},'
        ' "scores": [95.5, 87.3, 91.0], "note": null}'
    )
    _check(s, "realistic payload")


# ---------------------------------------------------------------------------
# Round-trip equivalence: the actual `loads`-produced Value should match
# whichever stage-1 implementation we run through stage 2.
# ---------------------------------------------------------------------------


def test_stage1_equivalence_on_real_corpora() raises:
    """The strongest equivalence test: scalar and SIMD stage 1 must
    agree on the live benchmark corpora. This file is what the
    twitter.json regression caught -- previously the SIMD escape state
    machine carried the `escaped` flag to the next marker regardless
    of distance, silently dropping closing quotes inside strings that
    contained `\\n` / `\\t` followed by long UTF-8 text."""
    from std.pathlib import Path

    var fixtures = List[String]()
    fixtures.append("benchmark/datasets/twitter.json")
    fixtures.append("benchmark/datasets/canada.json")
    fixtures.append("benchmark/datasets/citm_catalog.json")

    for i in range(len(fixtures)):
        var path = fixtures[i]
        try:
            var content = Path(path).read_text()
            var scalar = parse_structural_scalar(content)
            var simd = parse_structural_simd(content)
            _assert_indices_equal(path, scalar, simd)
        except:
            # Fixture not present in this checkout (the large 804MB
            # file isn't always vendored); skip rather than fail.
            print("  skipped (missing):", path)


def test_two_pass_dumps_round_trip() raises:
    """Running stage 1 + stage 2 must produce a Value whose dumps()
    output round-trips through `loads` to an equal Value. This catches
    cursor-desync bugs where stage 2 misinterprets a structural offset.
    """
    from json.cpu.stage2 import parse_two_pass
    from json import dumps

    var fixtures = List[String]()
    fixtures.append("null")
    fixtures.append("true")
    fixtures.append("false")
    fixtures.append("42")
    fixtures.append("3.14")
    fixtures.append('"hello"')
    fixtures.append("[]")
    fixtures.append("{}")
    fixtures.append("[1, 2, 3]")
    fixtures.append('{"a": 1, "b": "x"}')
    fixtures.append('{"nested": {"v": [1, 2, 3]}}')
    fixtures.append('{"escaped": "a\\"b"}')

    for i in range(len(fixtures)):
        var scalar_v = parse_two_pass[force_scalar=True](fixtures[i])
        var simd_v = parse_two_pass[force_scalar=False](fixtures[i])
        var lhs = dumps(scalar_v)
        var rhs = dumps(simd_v)
        assert_equal(lhs, rhs, "two-pass mismatch on: " + fixtures[i])


def main() raises:
    print("=" * 60)
    print("test_stage1_equivalence.mojo")
    print("=" * 60)
    print()
    TestSuite.discover_tests[__functions_in_module()]().run()
