"""Tests for the reference cleaner, its command line and tickers.csv."""

import copy
import csv
import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline.contract import build_reference, reference
from data_pipeline.contract.normalize import normalize_text as N

ROOT = Path(__file__).resolve().parents[2]
REAL_TICKERS = ROOT / "data_pipeline" / "reference" / "tickers.csv"
REAL_RAW = ROOT / "data" / "raw" / "financial_qa_10k.parquet"
FIELD_SEPARATOR = "\u001f"
RAW_FIELDS = ["question", "answer", "context", "ticker", "filing"]
REAL_TICKER_COUNT = 69
REAL_KEPT_ROWS = 6987
REAL_DUPLICATE_ROWS = 8


def raw_frame(*rows):
    """Build a raw frame; each row is a dict overriding a valid default row."""
    base = {
        "question": "What is revenue?",
        "answer": "It is money.",
        "context": "Revenue is money.",
        "ticker": "AAPL",
        "filing": "2023_10K",
    }
    return pd.DataFrame([{**base, **row} for row in rows], columns=RAW_FIELDS)


def expected_row_id(raw_index, question, answer, context, ticker, filing):
    joined = FIELD_SEPARATOR.join([question, answer, context, ticker, filing])
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
    return f"{raw_index:04d}-{digest}"


def row_id_of(frame, position):
    row = frame.iloc[position]
    return expected_row_id(position, *(row[f] for f in RAW_FIELDS))


def reasons(drop_log):
    return dict(zip(drop_log["raw_index"].tolist(), drop_log["reason"].tolist(), strict=True))


# ---------------------------------------------------------------- identifiers


def test_make_row_id_matches_the_spec_formula():
    args = ("Q?", "A.", "C.", "AAPL", "2023_10K")
    assert reference.make_row_id(12, *args) == expected_row_id(12, *args)


def test_make_row_id_pads_raw_index_to_four_digits():
    assert reference.make_row_id(7, "q", "a", "c", "AAPL", "2023_10K").startswith("0007-")


def test_make_row_id_depends_on_field_order():
    first = reference.make_row_id(1, "x", "y", "c", "AAPL", "2023_10K")
    swapped = reference.make_row_id(1, "y", "x", "c", "AAPL", "2023_10K")
    assert first != swapped


def test_make_row_id_uses_raw_text():
    plain = reference.make_row_id(1, "a b", "x", "c", "AAPL", "2023_10K")
    nbsp = reference.make_row_id(1, "a\u00a0b", "x", "c", "AAPL", "2023_10K")
    assert plain != nbsp


def test_make_context_id_is_sixteen_hex_of_sha256():
    expected = hashlib.sha256(b"Some context.").hexdigest()[:16]
    assert reference.make_context_id("Some context.") == expected


def test_clean_row_id_is_computed_from_raw_text():
    raw = raw_frame({"question": "a\u00a0b?"})
    cleaned, _ = reference.clean(raw)
    assert cleaned["row_id"].iloc[0] == row_id_of(raw, 0)
    assert cleaned["question"].iloc[0] == "a b?"


def test_clean_context_id_is_hash_of_cleaned_context():
    raw = raw_frame({"context": "x\u200b y"})
    cleaned, _ = reference.clean(raw)
    assert cleaned["context_id"].iloc[0] == hashlib.sha256(b"x y").hexdigest()[:16]


# ---------------------------------------------------------------- drop rules


def test_empty_after_normalization_is_empty_field():
    raw = raw_frame({"answer": "\u200b \u00a0"}, {"context": ""}, {"question": "<p>"})
    cleaned, drop_log = reference.clean(raw)
    assert cleaned.empty
    assert reasons(drop_log) == {0: "empty_field", 1: "empty_field", 2: "empty_field"}


def test_first_matching_rule_wins():
    raw = raw_frame({"answer": " ", "question": "\u0416"})
    _, drop_log = reference.clean(raw)
    assert reasons(drop_log) == {0: "empty_field"}


@pytest.mark.parametrize(
    ("char", "dropped"),
    [("\u0400", True), ("\u052f", True), ("\u03ff", False), ("\u0530", False)],
)
def test_cyrillic_range_edges(char, dropped):
    raw = raw_frame({"answer": f"x{char}y"})
    cleaned, drop_log = reference.clean(raw)
    assert (len(cleaned) == 0) is dropped
    if dropped:
        assert reasons(drop_log) == {0: "non_english_text"}


@pytest.mark.parametrize("field", ["ticker", "filing"])
def test_cyrillic_in_any_raw_field_counts(field):
    raw = raw_frame({field: "AAPL\u0416" if field == "ticker" else "2023\u0416"})
    # An unknown ticker must not raise before the row is dropped for its text.
    _, drop_log = reference.clean(raw)
    assert reasons(drop_log) == {0: "non_english_text"}


def test_exact_duplicate_compares_normalized_text():
    raw = raw_frame({"answer": "a b"}, {"answer": "a\u00a0b"})
    cleaned, drop_log = reference.clean(raw)
    assert cleaned["raw_index"].tolist() == [0]
    assert reasons(drop_log) == {1: "exact_duplicate"}
    assert drop_log["duplicate_of"].tolist() == [row_id_of(raw, 0)]
    assert drop_log["row_id"].tolist() == [row_id_of(raw, 1)]


def test_copy_of_a_dropped_row_keeps_the_dropped_rows_reason():
    raw = raw_frame({"answer": ""}, {"answer": ""}, {"answer": "\u0416"}, {"answer": "\u0416"})
    _, drop_log = reference.clean(raw)
    assert reasons(drop_log) == {
        0: "empty_field",
        1: "empty_field",
        2: "non_english_text",
        3: "non_english_text",
    }
    assert set(drop_log["duplicate_of"]) == {""}


def test_same_content_other_ticker_or_filing_is_kept():
    raw = raw_frame({}, {"ticker": "MSFT"}, {"filing": "2022_10K"})
    cleaned, drop_log = reference.clean(raw)
    assert cleaned["raw_index"].tolist() == [0, 1, 2]
    assert drop_log.empty


def test_paraphrase_with_another_answer_is_kept():
    raw = raw_frame({}, {"answer": "Money."})
    cleaned, drop_log = reference.clean(raw)
    assert len(cleaned) == 2
    assert drop_log.empty


# ---------------------------------------------------------------- derived columns


def test_qgroup_id_is_the_lowest_kept_row_with_casefolded_question():
    raw = raw_frame(
        {"question": "What is X?", "answer": ""},
        {"question": "What is X?"},
        {"question": "WHAT IS X?", "answer": "Other."},
        {"question": "What is X?", "ticker": "MSFT"},
    )
    cleaned, _ = reference.clean(raw)
    head = row_id_of(raw, 1)
    assert cleaned["raw_index"].tolist() == [1, 2, 3]
    assert cleaned["qgroup_id"].tolist() == [head, head, row_id_of(raw, 3)]


@pytest.mark.parametrize(("context", "expected"), [("a | b | c", True), ("a | b", False)])
def test_is_table_needs_two_pipes(context, expected):
    cleaned, _ = reference.clean(raw_frame({"context": context}))
    assert cleaned["is_table"].iloc[0] is expected or cleaned["is_table"].iloc[0] == expected


def test_is_table_counts_pipes_in_the_cleaned_context():
    cleaned, _ = reference.clean(raw_frame({"context": "<a|b> x | y"}))
    assert not cleaned["is_table"].iloc[0]


@pytest.mark.parametrize(
    ("question", "aliases", "expected"),
    [
        ("apple's revenue", ("Apple",), True),
        ("APPLE", ("Apple",), True),
        ("pineapple", ("Apple",), False),
        ("apples", ("Apple",), False),
        ("apple2", ("Apple",), False),
        ("2apple", ("Apple",), False),
        ("apple_inc", ("Apple",), True),
        ("apple\u00e9", ("Apple",), True),
        ("\u00e9apple", ("Apple",), True),
        ("at&t's network", ("AT&T",), True),
        ("nothing here", ("Apple",), False),
    ],
)
def test_mentions_company_boundaries(question, aliases, expected):
    assert reference.mentions_company(question, aliases) is expected


def test_one_letter_ticker_is_never_an_alias():
    raw = raw_frame(
        {"ticker": "F", "question": "Is f growing?"},
        {"ticker": "F", "question": "Is Ford growing?", "answer": "Yes."},
    )
    cleaned, _ = reference.clean(raw)
    assert cleaned["mentions_company"].tolist() == [False, True]


def test_aliases_by_ticker_reads_the_alias_column():
    aliases = reference.aliases_by_ticker(reference.load_tickers())
    assert aliases["F"] == ("Ford Motor", "Ford")
    assert aliases["T"] == ("AT&T",)


def test_company_name_comes_from_tickers_csv():
    raw = raw_frame({"ticker": "AAPL"}, {"ticker": "BRK-A"})
    cleaned, _ = reference.clean(raw)
    with REAL_TICKERS.open(encoding="utf-8", newline="") as handle:
        names = {r["ticker"]: r["company_name"] for r in csv.DictReader(handle)}
    assert cleaned["ticker"].tolist() == ["AAPL", "BRK-A"]
    assert cleaned["company_name"].tolist() == [names["AAPL"], names["BRK-A"]]


def test_unknown_ticker_raises_value_error():
    with pytest.raises(ValueError):
        reference.clean(raw_frame({"ticker": "ZZZZ"}))


# ---------------------------------------------------------------- frame shape


def test_clean_does_not_mutate_its_input():
    raw = raw_frame({"answer": "a\u00a0b"}, {"answer": ""})
    before = raw.copy(deep=True)
    reference.clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_clean_columns_order_and_drop_log_columns():
    raw = raw_frame({"question": "B?"}, {"question": "A?", "answer": ""}, {"question": "C?"})
    cleaned, drop_log = reference.clean(raw)
    assert tuple(cleaned.columns) == reference.CLEANED_COLUMNS
    assert tuple(drop_log.columns) == reference.DROP_LOG_COLUMNS
    assert cleaned["raw_index"].tolist() == sorted(cleaned["raw_index"].tolist())
    assert not cleaned.isna().any().any()


def test_constants_match_the_contract():
    assert reference.CLEANED_COLUMNS == (
        "row_id",
        "raw_index",
        "ticker",
        "company_name",
        "filing",
        "question",
        "answer",
        "context",
        "context_id",
        "qgroup_id",
        "is_table",
        "mentions_company",
    )
    assert reference.DROP_LOG_COLUMNS == ("raw_index", "row_id", "reason", "duplicate_of")


# ---------------------------------------------------------------- files


def test_write_outputs_schema_and_drop_log_format(tmp_path):
    raw = raw_frame({"question": "Q?"}, {"answer": ""}, {"answer": "\u0416"})
    cleaned, drop_log = reference.clean(raw)
    reference.write_outputs(cleaned, drop_log, tmp_path)

    schema = pq.read_schema(tmp_path / reference.CLEANED_FILENAME)
    assert tuple(schema.names) == reference.CLEANED_COLUMNS
    types = {f.name: f.type for f in schema}
    assert types["raw_index"] == pa.int32()
    assert types["is_table"] == pa.bool_()
    assert types["mentions_company"] == pa.bool_()
    assert all(
        types[n] == pa.string()
        for n in types
        if n not in {"raw_index", "is_table"} | {"mentions_company"}
    )

    data = (tmp_path / reference.DROP_LOG_FILENAME).read_bytes()
    assert b"\r" not in data
    assert data.decode("utf-8").split("\n")[0] == "raw_index,row_id,reason,duplicate_of"


def test_write_outputs_round_trips_cleaned_rows(tmp_path):
    cleaned, drop_log = reference.clean(raw_frame({}, {"ticker": "MSFT"}))
    reference.write_outputs(cleaned, drop_log, tmp_path)
    back = pq.read_table(tmp_path / reference.CLEANED_FILENAME).to_pandas()
    assert back["row_id"].tolist() == cleaned["row_id"].tolist()
    assert back["raw_index"].tolist() == [0, 1]


def test_read_raw_returns_the_raw_columns(tmp_path):
    path = tmp_path / "raw.parquet"
    raw_frame({}).to_parquet(path)
    assert list(reference.read_raw(path).columns) == list(reference.RAW_COLUMNS)


# ---------------------------------------------------------------- command line


def _run(capsys, *argv):
    code = build_reference.main([str(a) for a in argv])
    captured = capsys.readouterr()
    lines = [line for line in (captured.out + captured.err).splitlines() if line.strip()]
    return code, lines


def test_main_writes_both_files(tmp_path, capsys):
    raw_path = tmp_path / "raw.parquet"
    raw_frame({}, {"answer": ""}).to_parquet(raw_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    code, lines = _run(capsys, "--raw", raw_path, "--out-dir", out_dir)
    assert code == 0
    assert len(lines) == 1
    assert (out_dir / reference.CLEANED_FILENAME).is_file()
    assert (out_dir / reference.DROP_LOG_FILENAME).is_file()


def test_main_missing_raw_fails_with_one_line(tmp_path, capsys):
    code, lines = _run(capsys, "--raw", tmp_path / "none.parquet", "--out-dir", tmp_path)
    assert code == 1
    assert len(lines) == 1
    assert "Traceback" not in lines[0]


def test_main_raw_directory_fails_with_one_line(tmp_path, capsys):
    code, lines = _run(capsys, "--raw", tmp_path, "--out-dir", tmp_path)
    assert code == 1
    assert len(lines) == 1


def test_main_out_dir_that_is_a_file_fails_with_one_line(tmp_path, capsys):
    raw_path = tmp_path / "raw.parquet"
    raw_frame({}).to_parquet(raw_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    code, lines = _run(capsys, "--raw", raw_path, "--out-dir", blocker)
    assert code == 1
    assert len(lines) == 1
    assert "Traceback" not in lines[0]


# ---------------------------------------------------------------- tickers.csv


@pytest.fixture(scope="module")
def ticker_rows():
    with REAL_TICKERS.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def test_tickers_csv_shape(ticker_rows):
    fieldnames, rows = ticker_rows
    assert fieldnames == ["ticker", "company_name", "sector", "aliases"]
    assert len(rows) == REAL_TICKER_COUNT
    assert len({r["ticker"] for r in rows}) == REAL_TICKER_COUNT
    assert all(value for r in rows for value in r.values())


def test_tickers_csv_aliases_are_names_in_normal_form(ticker_rows):
    _, rows = ticker_rows
    for row in rows:
        aliases = row["aliases"].split("|")
        assert all(len(a) > 1 for a in aliases), row["ticker"]
        assert all(a == N(a) for a in aliases), row["ticker"]
        assert row["company_name"] == N(row["company_name"]), row["ticker"]


def test_load_tickers_matches_the_csv_file(ticker_rows):
    _, rows = ticker_rows
    loaded = reference.load_tickers()
    assert loaded["ticker"].tolist() == [r["ticker"] for r in rows]


# ---------------------------------------------------------------- real data


@pytest.mark.needs_data
def test_real_raw_tickers_equal_tickers_csv(ticker_rows):
    _, rows = ticker_rows
    raw = reference.read_raw(REAL_RAW)
    assert set(raw["ticker"]) == {r["ticker"] for r in rows}


@pytest.mark.needs_data
def test_real_raw_keeps_expected_rows_and_drops_listed_rows():
    raw = reference.read_raw(REAL_RAW)
    snapshot = copy.deepcopy(raw.to_dict("list"))
    cleaned, drop_log = reference.clean(raw)
    assert raw.to_dict("list") == snapshot
    assert len(cleaned) == REAL_KEPT_ROWS
    found = reasons(drop_log)
    for index in (1727, 1733, 2731):
        assert found[index] == "empty_field"
    for index in (2253, 6471):
        assert found[index] == "non_english_text"


@pytest.mark.needs_data
def test_real_raw_duplicates_repeat_a_kept_row():
    raw = reference.read_raw(REAL_RAW)
    cleaned, drop_log = reference.clean(raw)
    dups = drop_log[drop_log["reason"] == "exact_duplicate"]
    assert len(dups) == REAL_DUPLICATE_ROWS
    kept = cleaned.set_index("row_id")

    def key(row):
        return tuple(
            N(row[f]) if f in {"question", "answer", "context"} else row[f]
            for f in ("ticker", "filing", "question", "answer", "context")
        )

    for _, entry in dups.iterrows():
        assert entry["duplicate_of"] in kept.index
        original = kept.loc[entry["duplicate_of"]]
        assert key(raw.iloc[entry["raw_index"]]) == key(original)
        assert original["raw_index"] < entry["raw_index"]
