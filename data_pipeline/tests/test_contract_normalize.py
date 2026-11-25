"""Tests for the reference text normalization N."""

import pytest

from data_pipeline.contract.normalize import WHITE_SPACE, normalize_text

N = normalize_text

SPEC_WHITE_SPACE = [*range(0x09, 0x0E), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B)] + [
    0x2028,
    0x2029,
    0x202F,
    0x205F,
    0x3000,
]

CASES = [
    # step 1
    ("a\u0099b", "ab"),
    ("a\u200bb", "ab"),
    ("a\u00adb", "ab"),
    ("a\ufeffb", "ab"),
    ("a\u180eb", "ab"),
    ("a\x1cb", "ab"),
    ("a\x1db", "ab"),
    ("a\x1eb", "ab"),
    ("a\x1fb", "ab"),
    ("a\tb\nc", "a b c"),
    # step 2
    ("a\u2010b", "a-b"),
    ("a\u2011b", "a-b"),
    ("a\u2013b", "a-b"),
    ("a\u2014b", "a-b"),
    ("a\ufe58b", "a-b"),
    ("a\uff0db", "a-b"),
    ("a\u2e3ab", "a-b"),
    ("a\u2212b", "a-b"),
    ("\u2018x\u2019", "'x'"),
    ("\u201cx\u201d", '"x"'),
    ("\u2022 \u00ae \u2122 \u00e9 \u00a5 \u20ac \u00a7 \u03b1 \u0131", None),
    # step 3
    ("<<a>/b>", ""),
    ("</...>", ""),
    ("<p>x</p> y", "x y"),
    ("<.b>", ""),
    ("a<1>b", "a<1>b"),
    ("a < b > c", "a < b > c"),
    ("<a", "<a"),
    ("<->", "<->"),
    ("<\u200bp>x", "x"),
    ("<\u2013>", "<->"),
    ("a<b>\u00a0<i>c", "a c"),
    # step 4
    ("a\u3000\u3000b", "a b"),
    ("  a  ", "a"),
    ("\u00a0\u2003a b\u2028", "a b"),
    # step 5
    ("\u00b9", "\u00b9"),
    ("\u2122", "\u2122"),
    ("\ufb01", "\ufb01"),
    ("e\u0301", "\u00e9"),
    # nothing else
    ("ABC", "ABC"),
    ("What is it", "What is it"),
    ("a | b | c", "a | b | c"),
    ("", ""),
]


@pytest.mark.parametrize(("raw", "expected"), CASES)
def test_normalize_text_examples(raw, expected):
    assert N(raw) == (raw if expected is None else expected)


@pytest.mark.parametrize("code_point", sorted(set(SPEC_WHITE_SPACE) - {0x20}))
def test_each_white_space_code_point_collapses(code_point):
    assert N(f"a{chr(code_point)}b") == "a b"
    assert N(f"a{chr(code_point)}{chr(code_point)} b") == "a b"


def test_white_space_set_is_exactly_the_spec_list():
    assert set(WHITE_SPACE) == set(SPEC_WHITE_SPACE)
    assert len(WHITE_SPACE) == 25


def test_long_text_keeps_its_length():
    assert len(N("x" * 5000)) == 5000


@pytest.mark.parametrize("raw", [c[0] for c in CASES])
def test_normalize_text_is_idempotent(raw):
    assert N(N(raw)) == N(raw)
