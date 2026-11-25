"""Output checks V3 to V8 (docs/contracts.md, "Validator checks").

check_outputs compares a cleaner's output directory with the reference cleaner.
It reports problems as lines and never raises on malformed output.
"""

import re
import sys
import unicodedata
from functools import cache
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_pipeline.contract.normalize import TAG_PATTERN, normalize_text
from data_pipeline.contract.reference import (
    CLEANED_COLUMNS,
    CLEANED_FILENAME,
    DROP_LOG_COLUMNS,
    DROP_LOG_FILENAME,
    FLAG_COLUMNS,
    RAW_COLUMNS,
    TEXT_COLUMNS,
    clean,
    make_row_id,
)

MAX_LINES_PER_CHECK = 5
ROW_ID_PATTERN = re.compile(r"^\d{4}-[0-9a-f]{12}$")
_STRING_TYPES = (pa.string(), pa.large_string(), pa.string_view())
_DERIVED = ("company_name", "context_id", "qgroup_id", "is_table", "mentions_company")
_EXTRA_FORBIDDEN = (0x2212, 0x2018, 0x2019, 0x201C, 0x201D, *range(0x0400, 0x0530))


def cap(code: str, lines: list[str]) -> list[str]:
    """Keep the first MAX_LINES_PER_CHECK lines and add one "... and N more" line."""
    if len(lines) <= MAX_LINES_PER_CHECK:
        return lines
    rest = len(lines) - MAX_LINES_PER_CHECK
    return [*lines[:MAX_LINES_PER_CHECK], f"{code} ... and {rest} more"]


def _type_ok(column: str, data_type: pa.DataType) -> bool:
    if column == "raw_index":
        return data_type == pa.int32()
    if column in FLAG_COLUMNS:
        return data_type == pa.bool_()
    return data_type in _STRING_TYPES


def _load_cleaned(path: Path) -> tuple[dict[str, list] | None, list[str]]:
    """Read the cleaned parquet into column lists, or return V3 lines."""
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as error:
        return None, [
            f"V3 cleaned parquet {CLEANED_FILENAME} unreadable: {ascii(str(error))[:120]}"
        ]
    if tuple(table.column_names) != CLEANED_COLUMNS:
        return None, [
            f"V3 columns are {ascii(table.column_names)}, expected {list(CLEANED_COLUMNS)}"
        ]
    lines = [
        f"V3 column {name} has type {table.schema.field(name).type}"
        for name in CLEANED_COLUMNS
        if not _type_ok(name, table.schema.field(name).type)
    ]
    lines += [f"V3 column {n} has nulls" for n in CLEANED_COLUMNS if table.column(n).null_count]
    if lines:
        return None, lines
    return {n: table.column(n).to_pylist() for n in CLEANED_COLUMNS}, []


def _load_drop_log(path: Path) -> tuple[list[tuple[str, ...]] | None, list[str]]:
    """Read drop_log.csv as string tuples, or return V3 lines."""
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        return None, [f"V3 {DROP_LOG_FILENAME} unreadable: {ascii(str(error))[:120]}"]
    if tuple(frame.columns) != DROP_LOG_COLUMNS:
        return None, [f"V3 {DROP_LOG_FILENAME} columns are {ascii(list(frame.columns))}"]
    rows = list(zip(*(frame[c].tolist() for c in DROP_LOG_COLUMNS), strict=True))
    bad = [r for r in rows if not r[0].isascii() or not r[0].isdigit()]
    if bad:
        return None, [f"V3 {DROP_LOG_FILENAME} raw_index is not an integer: {ascii(bad[0][0])}"]
    return rows, []


def _check_shape(cleaned: dict[str, list]) -> list[str]:
    indexes, ids = cleaned["raw_index"], cleaned["row_id"]
    lines = [
        f"V3 row_id {ascii(i)} does not match the pattern"
        for i in ids
        if not ROW_ID_PATTERN.match(i)
    ]
    seen: set[str] = set()
    lines += [f"V3 row_id {i} is repeated" for i in ids if i in seen or seen.add(i)]
    lines += [
        f"V3 raw_index is not sorted at position {p}"
        for p, (a, b) in enumerate(zip(indexes, indexes[1:], strict=False))
        if a > b
    ]
    return cap("V3", lines)


def _check_provenance(cleaned: dict[str, list], raw: pd.DataFrame) -> list[str]:
    raw_cols = [raw[c].tolist() for c in RAW_COLUMNS]
    lines = []
    for index, row_id in zip(cleaned["raw_index"], cleaned["row_id"], strict=True):
        if not 0 <= index < len(raw):
            lines.append(f"V4 raw_index {index} is outside 0..{len(raw) - 1}")
        elif row_id != make_row_id(index, *(col[index] for col in raw_cols)):
            lines.append(f"V4 row_id {ascii(row_id)} does not match raw row {index}")
    return cap("V4", lines)


def _check_partition(
    cleaned: dict[str, list], drop_rows: list[tuple[str, ...]], raw_len: int, ref_drop: pd.DataFrame
) -> list[str]:
    kept = cleaned["raw_index"]
    dropped = [int(r[0]) for r in drop_rows]
    lines = []
    lines += [f"V5 kept raw_index {i} appears twice" for i in _repeats(kept)]
    lines += [f"V5 drop-log raw_index {i} appears twice" for i in _repeats(dropped)]
    lines += [
        f"V5 raw_index {i} is both kept and dropped" for i in sorted(set(kept) & set(dropped))
    ]
    lines += [
        f"V5 drop-log raw_index {i} is outside 0..{raw_len - 1}"
        for i in dropped
        if not 0 <= i < raw_len
    ]
    missing = sorted(set(range(raw_len)) - set(kept) - set(dropped))
    lines += [f"V5 raw_index {i} is neither kept nor in the drop log" for i in missing]
    lines += _compare_drop_log(drop_rows, ref_drop)
    return cap("V5", lines)


def _repeats(values: list[int]) -> list[int]:
    seen: set[int] = set()
    return [v for v in values if v in seen or seen.add(v)]


def _compare_drop_log(rows: list[tuple[str, ...]], ref_drop: pd.DataFrame) -> list[str]:
    expected = {
        str(r[0]): tuple(str(x) for x in r)
        for r in zip(*(ref_drop[c].tolist() for c in DROP_LOG_COLUMNS), strict=True)
    }
    actual = {r[0].lstrip("0") or "0": r for r in rows}
    lines = [f"V5 drop log lacks the row for raw_index {i}" for i in expected if i not in actual]
    lines += [
        f"V5 drop-log row {ascii(r)} is not expected"
        for k, r in actual.items()
        if k not in expected
    ]
    lines += [
        f"V5 drop-log row {ascii(actual[k])} differs from {ascii(v)}"
        for k, v in expected.items()
        if k in actual and actual[k] != v
    ]
    return lines


def _check_text(cleaned: dict[str, list], raw: pd.DataFrame) -> list[str]:
    lines = []
    for column in TEXT_COLUMNS:
        raw_col = raw[column].tolist()
        for index, value in zip(cleaned["raw_index"], cleaned[column], strict=True):
            if 0 <= index < len(raw_col) and value != normalize_text(raw_col[index]):
                lines.append(
                    f"V6 raw_index {index} {column} differs from N(raw): {ascii(value)[:80]}"
                )
    return cap("V6", lines)


def _check_derived(cleaned: dict[str, list], ref: pd.DataFrame) -> list[str]:
    position = {i: p for p, i in enumerate(ref["raw_index"].tolist())}
    lines = []
    for column in _DERIVED:
        expected = ref[column].tolist()
        for index, value in zip(cleaned["raw_index"], cleaned[column], strict=True):
            p = position.get(index)
            if p is not None and value != expected[p]:
                lines.append(
                    f"V7 raw_index {index} {column} is {ascii(value)}, "
                    f"expected {ascii(expected[p])}"
                )
    return cap("V7", lines)


@cache
def _forbidden() -> re.Pattern[str]:
    """Match any Cc/Cf code point, any Pd except "-", and the characters N rewrites."""
    chars = {
        c
        for c in range(sys.maxunicode + 1)
        if unicodedata.category(chr(c)) in ("Cc", "Cf", "Pd") and c != ord("-")
    } | set(_EXTRA_FORBIDDEN)
    return re.compile("[" + "".join(re.escape(chr(c)) for c in sorted(chars)) + "]")


def _text_problem(text: str) -> str | None:
    found = _forbidden().search(text)
    if found:
        return f"has forbidden character U+{ord(found.group()):04X}"
    if "  " in text:
        return "has a double space"
    if text.startswith(" ") or text.endswith(" "):
        return "has a leading or trailing space"
    if TAG_PATTERN.search(text):
        return "contains a tag"
    if not unicodedata.is_normalized("NFC", text):
        return "is not NFC"
    return None


def _check_properties(cleaned: dict[str, list]) -> list[str]:
    lines = []
    for column in TEXT_COLUMNS:
        for index, text in zip(cleaned["raw_index"], cleaned[column], strict=True):
            problem = _text_problem(text)
            if problem:
                lines.append(f"V8 raw_index {index} {column} {problem}")
    return cap("V8", lines)


def check_outputs(raw: pd.DataFrame, out_dir: Path, tickers: pd.DataFrame) -> list[str]:
    """Run V3 to V8 on out_dir against clean(raw, tickers); [] means every check passed."""
    cleaned, lines = _load_cleaned(out_dir / CLEANED_FILENAME)
    drop_rows, drop_lines = _load_drop_log(out_dir / DROP_LOG_FILENAME)
    lines += drop_lines
    if cleaned is not None:
        lines += _check_shape(cleaned)
    if lines or cleaned is None or drop_rows is None:
        return cap("V3", lines)
    try:
        ref_cleaned, ref_drop = clean(raw, tickers)
    except ValueError as error:
        return [f"V7 reference cleaning failed: {ascii(str(error))[:120]}"]
    return (
        _check_provenance(cleaned, raw)
        + _check_partition(cleaned, drop_rows, len(raw), ref_drop)
        + _check_text(cleaned, raw)
        + _check_derived(cleaned, ref_cleaned)
        + _check_properties(cleaned)
    )


def emit(label: str, lines: list[str]) -> bool:
    """Print one check's result; return True when it passed."""
    if lines:
        print("\n".join(lines))
    else:
        print(f"{label}: ok")
    return not lines


def report_outputs(lines: list[str]) -> bool:
    names = {
        "V3": "schema",
        "V4": "provenance",
        "V5": "partition",
        "V6": "text",
        "V7": "derived columns",
        "V8": "properties",
    }
    if any(line.startswith("V3 ") for line in lines):
        ok = emit("V3 schema", lines)
        print("V4-V8: skipped (V3 failed)")
        return ok
    results = [
        emit(f"{c} {n}", [x for x in lines if x.startswith(f"{c} ")]) for c, n in names.items()
    ]
    return all(results)
