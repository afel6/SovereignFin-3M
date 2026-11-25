"""Tests of the dataset validator: it passes the reference output and fails every mutation."""

import hashlib
import re
import shutil
import sys
import unicodedata
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline.contract import build_reference, reference, validate
from data_pipeline.contract.normalize import normalize_text

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_RAW = REPO_ROOT / "data" / "raw" / "financial_qa_10k.parquet"
CLEANED = reference.CLEANED_FILENAME
DROP_LOG = reference.DROP_LOG_FILENAME

# Hand-written expectations for the small fixture (see small_raw): 13 special rows, then one
# plain row per ticker in tickers.csv. Rows 4, 5 and 7 are dropped; the other 10 special rows
# are kept.
SPECIAL_ROWS = 13
DROPPED = {4: "empty_field", 5: "non_english_text", 7: "exact_duplicate"}
EXPECTED_KEPT_SPECIAL = 10
# The full dataset: 7000 rows minus 3 empty, 2 non-English and 8 exact duplicates.
EXPECTED_KEPT = 6987
EMPTY_ROWS = {1727, 1733, 2731}
NON_ENGLISH_ROWS = {2253, 6471}
EXACT_DUPLICATE_COUNT = 8


def make_row(question, answer, context, ticker, filing="2023_10K"):
    return {
        "question": question,
        "answer": answer,
        "context": context,
        "ticker": ticker,
        "filing": filing,
    }


def build_small_raw(tickers: pd.DataFrame) -> pd.DataFrame:
    special = [
        make_row("What were total sales?", "$10 billion", "Sales were $10 billion.", "AAPL"),
        make_row(
            "What is the − 5% change​?",
            "It’s “up”",
            "Text—more  text",
            "JNJ",
        ),
        make_row("Where is the tag?", "Here", "Revenue </...> grew <b>fast</b>.", "JNJ"),
        make_row("What is in the table?", "Rows", "a | b | c", "T"),
        make_row("What is empty?", "​", "Some context.", "AAPL"),
        make_row("What is Russian?", "Nothing", "Привет", "GOOGL"),
        make_row("Duplicate q?", "x", "y", "AAPL"),
        make_row("Duplicate q?", "x", "y", "AAPL"),
        make_row("WHAT WERE TOTAL SALES?", "different", "other context", "AAPL"),
        make_row("How did Apple perform?", "Well", "Apple did well.", "AAPL"),
        make_row("What did F report?", "Little", "F reported little.", "F"),
        make_row("What is the class A price?", "High", "BRK-A is high.", "BRK-A"),
        make_row("What is the search share?", "Large", "Search is large.", "GOOGL"),
    ]
    plain = [
        make_row(f"Question number {i}?", "A", f"Context {i}", t)
        for i, t in enumerate(tickers["ticker"])
    ]
    return pd.DataFrame(special + plain, columns=list(reference.RAW_COLUMNS))


@pytest.fixture(scope="module")
def tickers() -> pd.DataFrame:
    return reference.load_tickers()


@pytest.fixture(scope="module")
def small_raw(tickers) -> pd.DataFrame:
    return build_small_raw(tickers)


@pytest.fixture(scope="module")
def small_ref(small_raw, tickers, tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("small_ref")
    cleaned, drop_log = reference.clean(small_raw, tickers)
    reference.write_outputs(cleaned, drop_log, out_dir)
    return out_dir


def write_raw(df: pd.DataFrame, path: Path) -> str:
    df.to_parquet(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def starts_with_code(lines: list[str], code: str) -> bool:
    return any(line.startswith(code + " ") for line in lines)


# ---------------------------------------------------------------- reference output passes


def test_reference_output_passes(small_raw, small_ref, tickers):
    assert validate.check_outputs(small_raw, small_ref, tickers) == []


def test_reference_normalizer_passes(small_raw):
    assert validate.check_normalizer(small_raw, normalize_text) == []


def test_reference_partition_matches_drop_rules(small_ref, tickers):
    cleaned = pq.read_table(small_ref / CLEANED).to_pandas()
    drop_log = pd.read_csv(small_ref / DROP_LOG)
    assert len(cleaned) == len(tickers) + EXPECTED_KEPT_SPECIAL
    assert EXPECTED_KEPT_SPECIAL == SPECIAL_ROWS - len(DROPPED)
    assert dict(zip(drop_log["raw_index"], drop_log["reason"], strict=True)) == DROPPED
    assert cleaned.loc[cleaned["raw_index"] == 8, "qgroup_id"].item() == cleaned["row_id"].iloc[0]
    assert (
        drop_log["duplicate_of"].iloc[2] == cleaned.loc[cleaned["raw_index"] == 6, "row_id"].item()
    )


def test_check_tickers(small_raw, tickers):
    assert validate.check_tickers(small_raw, tickers) == []
    assert starts_with_code(validate.check_tickers(small_raw, tickers.iloc[1:]), "V7")


# ---------------------------------------------------------------- V12 mutations

Mutation = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]


def swap_rows(df: pd.DataFrame, i: int, j: int) -> pd.DataFrame:
    order = list(range(len(df)))
    order[i], order[j] = order[j], order[i]
    return df.iloc[order].reset_index(drop=True)


def m_drop_kept_row(cleaned, drop_log, raw):
    return cleaned.drop(cleaned.index[3]).reset_index(drop=True), drop_log


def m_edit_one_char(cleaned, drop_log, raw):
    out = cleaned.copy()
    answer = out.at[0, "answer"]
    out.at[0, "answer"] = answer[:-1] + ("x" if answer[-1] != "x" else "y")
    return out, drop_log


def m_swap_row_ids(cleaned, drop_log, raw):
    out = cleaned.copy()
    first, second = out.at[0, "row_id"], out.at[1, "row_id"]
    out.at[0, "row_id"], out.at[1, "row_id"] = second, first
    return out, drop_log


def m_invent_row(cleaned, drop_log, raw):
    invented = cleaned.iloc[[-1]].copy()
    invented["raw_index"] = len(raw)
    invented["row_id"] = f"{len(raw):04d}-" + "0" * 12
    return pd.concat([cleaned, invented], ignore_index=True), drop_log


def m_flip_is_table(cleaned, drop_log, raw):
    out = cleaned.copy()
    out.at[0, "is_table"] = not out.at[0, "is_table"]
    return out, drop_log


def m_flip_mentions_company(cleaned, drop_log, raw):
    out = cleaned.copy()
    out.at[0, "mentions_company"] = not out.at[0, "mentions_company"]
    return out, drop_log


def m_reorder_rows(cleaned, drop_log, raw):
    return swap_rows(cleaned, 0, 1), drop_log


def m_omit_drop_log_row(cleaned, drop_log, raw):
    return cleaned, drop_log.drop(drop_log.index[0]).reset_index(drop=True)


def m_raw_text_as_output(cleaned, drop_log, raw):
    out = cleaned.copy()
    for column in ("question", "answer", "context"):
        out[column] = [raw[column].iloc[i] for i in out["raw_index"]]
    return out, drop_log


MUTATIONS = {
    "drop_kept_row": (m_drop_kept_row, "V5"),
    "edit_one_char": (m_edit_one_char, "V6"),
    "swap_row_ids": (m_swap_row_ids, "V4"),
    "invent_row": (m_invent_row, "V4"),
    "flip_is_table": (m_flip_is_table, "V7"),
    "flip_mentions_company": (m_flip_mentions_company, "V7"),
    "reorder_rows": (m_reorder_rows, "V3"),
    "omit_drop_log_row": (m_omit_drop_log_row, "V5"),
    "raw_text_as_output": (m_raw_text_as_output, "V6"),
}


def write_cleaned(df: pd.DataFrame, path: Path, schema=reference.CLEANED_SCHEMA) -> None:
    pq.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False), path)


def mutated_copy(ref_dir: Path, dst: Path, mutate: Mutation, raw: pd.DataFrame) -> Path:
    shutil.copytree(ref_dir, dst)
    cleaned = pq.read_table(dst / CLEANED).to_pandas()
    drop_log = pd.read_csv(dst / DROP_LOG)
    cleaned, drop_log = mutate(cleaned, drop_log, raw)
    write_cleaned(cleaned, dst / CLEANED)
    drop_log.to_csv(dst / DROP_LOG, index=False, lineterminator="\n", encoding="utf-8")
    return dst


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_mutation_fails_with_its_code(name, small_raw, small_ref, tickers, tmp_path):
    mutate, code = MUTATIONS[name]
    out = mutated_copy(small_ref, tmp_path / "mut", mutate, small_raw)
    lines = validate.check_outputs(small_raw, out, tickers)
    assert starts_with_code(lines, code), lines


def test_reorder_reports_only_v3(small_raw, small_ref, tickers, tmp_path):
    out = mutated_copy(small_ref, tmp_path / "mut", m_reorder_rows, small_raw)
    lines = validate.check_outputs(small_raw, out, tickers)
    assert lines and all(line.startswith("V3 ") for line in lines)


def flip_middle_byte(src: Path, dst: Path) -> None:
    data = bytearray(src.read_bytes())
    data[len(data) // 2] ^= 0x01
    dst.write_bytes(bytes(data))


def test_changed_raw_byte_fails_v1(small_raw, tmp_path):
    original = tmp_path / "raw.parquet"
    sha = write_raw(small_raw, original)
    changed = tmp_path / "changed.parquet"
    flip_middle_byte(original, changed)
    lines = validate.check_raw_file(changed, expected_sha256=sha, expected_rows=len(small_raw))
    assert starts_with_code(lines, "V1"), lines


# ---------------------------------------------------------------- V3 malformed output


def m3_missing_parquet(d: Path) -> None:
    (d / CLEANED).unlink()


def m3_missing_drop_log(d: Path) -> None:
    (d / DROP_LOG).unlink()


def edit_cleaned(d: Path, edit: Callable[[pd.DataFrame], None], schema=reference.CLEANED_SCHEMA):
    df = pq.read_table(d / CLEANED).to_pandas()
    edit(df)
    write_cleaned(df, d / CLEANED, schema)


def m3_int64_raw_index(d: Path) -> None:
    df = pq.read_table(d / CLEANED).to_pandas()
    df["raw_index"] = df["raw_index"].astype("int64")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), d / CLEANED)


def m3_null_text(d: Path) -> None:
    edit_cleaned(d, lambda df: df.__setitem__("answer", [None, *df["answer"].iloc[1:]]))


def m3_bad_row_id(d: Path) -> None:
    edit_cleaned(d, lambda df: df.__setitem__("row_id", ["bad", *df["row_id"].iloc[1:]]))


def m3_duplicate_row_id(d: Path) -> None:
    ids = None

    def edit(df):
        nonlocal ids
        ids = list(df["row_id"])
        ids[1] = ids[0]
        df["row_id"] = ids

    edit_cleaned(d, edit)


MALFORMED = [
    m3_missing_parquet,
    m3_missing_drop_log,
    m3_int64_raw_index,
    m3_null_text,
    m3_bad_row_id,
    m3_duplicate_row_id,
]


@pytest.mark.parametrize("damage", MALFORMED, ids=lambda f: f.__name__)
def test_malformed_output_gives_v3_lines(damage, small_raw, small_ref, tickers, tmp_path):
    dst = tmp_path / "bad"
    shutil.copytree(small_ref, dst)
    damage(dst)
    lines = validate.check_outputs(small_raw, dst, tickers)
    assert lines and all(line.startswith("V3 ") for line in lines), lines


# ---------------------------------------------------------------- V1


def test_raw_file_passes_with_its_own_pin(small_raw, tmp_path):
    path = tmp_path / "raw.parquet"
    sha = write_raw(small_raw, path)
    assert validate.check_raw_file(path, expected_sha256=sha, expected_rows=len(small_raw)) == []


def v1_lines(df: pd.DataFrame, tmp_path: Path, rows: int | None = None) -> list[str]:
    path = tmp_path / "raw.parquet"
    sha = write_raw(df, path)
    return validate.check_raw_file(path, expected_sha256=sha, expected_rows=rows or len(df))


def test_v1_wrong_row_count(small_raw, tmp_path):
    assert starts_with_code(v1_lines(small_raw, tmp_path, len(small_raw) + 1), "V1")


def test_v1_wrong_sha(small_raw, tmp_path):
    path = tmp_path / "raw.parquet"
    write_raw(small_raw, path)
    lines = validate.check_raw_file(path, expected_sha256="0" * 64, expected_rows=len(small_raw))
    assert starts_with_code(lines, "V1")


def test_v1_extra_column(small_raw, tmp_path):
    assert starts_with_code(v1_lines(small_raw.assign(extra="x"), tmp_path), "V1")


def test_v1_missing_column(small_raw, tmp_path):
    assert starts_with_code(v1_lines(small_raw.drop(columns="filing"), tmp_path), "V1")


def test_v1_bad_filing(small_raw, tmp_path):
    bad = small_raw.copy()
    bad.loc[0, "filing"] = "2023_10-K"
    assert starts_with_code(v1_lines(bad, tmp_path), "V1")


# ---------------------------------------------------------------- V2


def test_determinism_identical_dirs(small_ref, tmp_path):
    second = tmp_path / "second"
    shutil.copytree(small_ref, second)
    assert validate.check_determinism(small_ref, second) == []


def test_determinism_differing_drop_log_byte(small_ref, tmp_path):
    second = tmp_path / "second"
    shutil.copytree(small_ref, second)
    path = second / DROP_LOG
    path.write_bytes(path.read_bytes().replace(b"empty_field", b"empty_fielx", 1))
    assert starts_with_code(validate.check_determinism(small_ref, second), "V2")


def test_determinism_missing_file(small_ref, tmp_path):
    second = tmp_path / "second"
    shutil.copytree(small_ref, second)
    (second / CLEANED).unlink()
    assert starts_with_code(validate.check_determinism(small_ref, second), "V2")


# ---------------------------------------------------------------- V9 wrong normalizers


def wrong_nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", normalize_text(s))


def wrong_file_separator_as_space(s: str) -> str:
    return normalize_text(s.replace("\x1c", " "))


def wrong_tag_one_or_more(s: str) -> str:
    # Private-use stand-ins for "<" and ">" keep the reference from removing tags first.
    text = normalize_text(s.replace("<", "").replace(">", ""))
    pattern = re.compile("/?[A-Za-z.][^]+")
    while (stripped := pattern.sub("", text)) != text:
        text = stripped
    return text.replace("", "<").replace("", ">")


def wrong_keeps_zero_width_space(s: str) -> str:
    return normalize_text(s.replace("​", "")).replace("", "​")


def wrong_not_idempotent(s: str) -> str:
    return s + "!"


def wrong_raises(s: str) -> str:
    raise ValueError("boom")


WRONG_NORMALIZERS = [
    wrong_nfkc,
    wrong_file_separator_as_space,
    wrong_tag_one_or_more,
    wrong_keeps_zero_width_space,
    wrong_not_idempotent,
    wrong_raises,
]


@pytest.mark.parametrize("wrong", WRONG_NORMALIZERS, ids=lambda f: f.__name__)
def test_wrong_normalizer_fails_v9(wrong, small_raw):
    assert starts_with_code(validate.check_normalizer(small_raw, wrong), "V9")


# ---------------------------------------------------------------- V10, V11


@pytest.mark.parametrize(("kept", "ok"), [(6974, False), (6975, True), (6995, True), (6996, False)])
def test_kept_count_band(kept, ok):
    lines = validate.check_kept_count(kept)
    assert (lines == []) is ok
    assert ok or starts_with_code(lines, "V10")


@pytest.mark.parametrize(
    ("path", "ok"),
    [
        ("data_pipeline/clean/a.py", True),
        ("data_pipeline/clean/tests/test_a.py", True),
        ("database/tenk/tenk_snippets.sql", True),
        ("database/init.sql", False),
        ("data_pipeline/cleaner/x.py", False),
        ("data_pipeline/contract/validate.py", False),
    ],
)
def test_scope(path, ok):
    lines = validate.check_scope([path])
    assert (lines == []) is ok
    assert ok or starts_with_code(lines, "V11")


# ---------------------------------------------------------------- import ban

PACKAGE = "data_pipeline.clean"
BANNED_SOURCES = {
    "plain": "import data_pipeline.contract\n",
    "aliased": "import data_pipeline.contract.normalize as n\n",
    "from_module": "from data_pipeline.contract.normalize import normalize_text\n",
    "from_package": "from data_pipeline import contract\n",
    "relative": "from ..contract import normalize\n",
    "import_module": 'import importlib\nimportlib.import_module("data_pipeline.contract")\n',
    "dunder_import": '__import__("data_pipeline.contract.normalize")\n',
}
ALLOWED_SOURCES = {
    "pandas": "import pandas\n",
    "download": "from data_pipeline.download import PINNED_SHA256\n",
    "docstring": '"""Mentions data_pipeline.contract only in prose."""\n',
}


def write_package(root: Path, relative: str, source: str) -> Path:
    pkg = root / "pkg"
    (pkg / relative).parent.mkdir(parents=True, exist_ok=True)
    (pkg / relative).write_text(source, encoding="utf-8", newline="\n")
    return pkg


@pytest.mark.parametrize("name", list(BANNED_SOURCES))
def test_import_ban_rejects(name, tmp_path):
    pkg = write_package(tmp_path, "mod.py", BANNED_SOURCES[name])
    lines = validate.check_import_ban(package_dir=pkg, package=PACKAGE)
    assert starts_with_code(lines, "import-ban"), lines


def test_import_ban_rejects_relative_import_in_subpackage(tmp_path):
    pkg = write_package(tmp_path, "sub/mod.py", "from ...contract import x\n")
    lines = validate.check_import_ban(package_dir=pkg, package=PACKAGE)
    assert starts_with_code(lines, "import-ban"), lines


@pytest.mark.parametrize("name", list(ALLOWED_SOURCES))
def test_import_ban_allows(name, tmp_path):
    pkg = write_package(tmp_path, "mod.py", ALLOWED_SOURCES[name])
    assert validate.check_import_ban(package_dir=pkg, package=PACKAGE) == []


def test_import_ban_syntax_error_is_a_line(tmp_path):
    pkg = write_package(tmp_path, "mod.py", "def (:\n")
    assert starts_with_code(
        validate.check_import_ban(package_dir=pkg, package=PACKAGE), "import-ban"
    )


def test_import_ban_real_clean_package_passes():
    assert validate.check_import_ban() == []


# ---------------------------------------------------------------- module_from_cmd


@pytest.mark.parametrize(
    ("cmd", "module"),
    [
        ("python -m data_pipeline.clean", "data_pipeline.clean"),
        ("uv run python -m x.y --flag", "x.y"),
        ("python script.py", None),
    ],
)
def test_module_from_cmd(cmd, module):
    assert validate.module_from_cmd(cmd) == module


# ---------------------------------------------------------------- main, in process

REFERENCE_CMD = "python -m data_pipeline.contract.build_reference"


@pytest.fixture
def pinned_small(small_raw, monkeypatch, tmp_path) -> Path:
    path = tmp_path / "raw.parquet"
    sha = write_raw(small_raw, path)
    kept = len(small_raw) - len(DROPPED)
    monkeypatch.setattr(validate, "PINNED_SHA256", sha)
    monkeypatch.setattr(validate, "EXPECTED_ROWS", len(small_raw))
    monkeypatch.setattr(validate, "KEPT_BAND", (kept, kept))
    return path


def test_main_passes_on_reference_build(pinned_small, capsys):
    assert validate.main(["--cmd", REFERENCE_CMD, "--raw", str(pinned_small)]) == 0
    text = capsys.readouterr()
    combined = text.out + text.err
    assert "validator: passed" in combined
    assert "V11 scope: skipped" in combined


def one_friendly_line(capsys) -> None:
    text = capsys.readouterr()
    combined = text.out + text.err
    assert "Traceback" not in combined
    assert len([line for line in combined.splitlines() if line.strip()]) == 1, combined


def test_main_failing_command(pinned_small, capsys):
    cmd = f'{sys.executable} -c "import sys; sys.exit(3)"'
    assert validate.main(["--cmd", cmd, "--raw", str(pinned_small)]) == 1
    one_friendly_line(capsys)


def test_main_command_that_does_not_exist(pinned_small, capsys):
    assert validate.main(["--cmd", "no-such-command-xyz", "--raw", str(pinned_small)]) == 1
    one_friendly_line(capsys)


def test_main_missing_raw(pinned_small, tmp_path, capsys):
    missing = tmp_path / "absent.parquet"
    assert validate.main(["--cmd", REFERENCE_CMD, "--raw", str(missing)]) == 1
    one_friendly_line(capsys)


# ---------------------------------------------------------------- real data


@pytest.fixture(scope="module")
def real_raw() -> pd.DataFrame:
    return pd.read_parquet(REAL_RAW)


@pytest.fixture(scope="module")
def real_ref(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("real_ref")
    assert build_reference.main(["--raw", str(REAL_RAW), "--out-dir", str(out_dir)]) == 0
    return out_dir


@pytest.mark.needs_data
def test_real_raw_file_passes_v1():
    assert validate.check_raw_file(REAL_RAW) == []


@pytest.mark.needs_data
def test_real_reference_counts(real_ref):
    cleaned = pq.read_table(real_ref / CLEANED).to_pandas()
    drop_log = pd.read_csv(real_ref / DROP_LOG)
    by_reason = {r: set(g["raw_index"]) for r, g in drop_log.groupby("reason")}
    assert len(cleaned) == EXPECTED_KEPT
    assert by_reason["empty_field"] == EMPTY_ROWS
    assert by_reason["non_english_text"] == NON_ENGLISH_ROWS
    assert len(by_reason["exact_duplicate"]) == EXACT_DUPLICATE_COUNT


@pytest.mark.needs_data
def test_real_reference_passes_checks(real_raw, real_ref):
    tickers = reference.load_tickers()
    assert validate.check_outputs(real_raw, real_ref, tickers) == []
    assert validate.check_tickers(real_raw, tickers) == []
    assert validate.check_kept_count(EXPECTED_KEPT) == []
    assert validate.check_normalizer(real_raw, normalize_text) == []


@pytest.mark.needs_data
@pytest.mark.parametrize("name", list(MUTATIONS))
def test_real_mutation_fails_with_its_code(name, real_raw, real_ref, tmp_path):
    mutate, code = MUTATIONS[name]
    out = mutated_copy(real_ref, tmp_path / "mut", mutate, real_raw)
    lines = validate.check_outputs(real_raw, out, reference.load_tickers())
    assert starts_with_code(lines, code), lines[:3]


@pytest.mark.needs_data
def test_real_changed_raw_byte_fails_v1(tmp_path):
    sha = hashlib.sha256(REAL_RAW.read_bytes()).hexdigest()
    changed = tmp_path / "changed.parquet"
    flip_middle_byte(REAL_RAW, changed)
    assert starts_with_code(validate.check_raw_file(changed, expected_sha256=sha), "V1")


@pytest.mark.needs_data
def test_real_full_cli_run(capsys):
    assert validate.main(["--cmd", REFERENCE_CMD]) == 0
    assert "validator: passed" in capsys.readouterr().out
