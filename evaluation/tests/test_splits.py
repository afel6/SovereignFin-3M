import hashlib
import io
from pathlib import Path

import pandas as pd
import pytest

from data_pipeline.contract import build_reference
from evaluation import splits

COMMITTED = Path("evaluation/splits.csv")
REAL_ROWS = 6987


def bucket(row_id: str) -> int:
    return int(hashlib.sha256(row_id.encode("utf-8")).hexdigest(), 16) % 10


def id_with_bucket(wanted: int) -> str:
    return next(c for c in (f"row{i}" for i in range(1000)) if bucket(c) == wanted)


def frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["row_id", "context_id", "qgroup_id"])


def column_map(out: pd.DataFrame, column: str) -> dict[str, str]:
    return dict(zip(out["row_id"], out[column], strict=True))


def test_rows_sharing_context_share_component():
    out = splits.build_splits(frame([("a", "c1", "q1"), ("b", "c1", "q2"), ("c", "c2", "q3")]))
    assert column_map(out, "component") == {"a": "a", "b": "a", "c": "c"}


def test_rows_sharing_qgroup_share_component():
    out = splits.build_splits(frame([("a", "c1", "q1"), ("b", "c2", "q1"), ("c", "c3", "q3")]))
    assert column_map(out, "component") == {"a": "a", "b": "a", "c": "c"}


def test_chain_of_context_then_qgroup_is_one_component():
    out = splits.build_splits(frame([("a", "c1", "q1"), ("b", "c1", "q2"), ("c", "c2", "q2")]))
    assert set(out["component"]) == {"a"}


def test_component_name_is_lowest_row_id_not_first_row():
    out = splits.build_splits(frame([("z", "c1", "q1"), ("m", "c1", "q2"), ("b", "c2", "q2")]))
    assert set(out["component"]) == {"b"}


def test_split_buckets_follow_the_rule():
    one, two, zero = id_with_bucket(1), id_with_bucket(2), id_with_bucket(0)
    rows = [(r, f"c{r}", f"q{r}") for r in (one, two, zero)]
    got = column_map(splits.build_splits(frame(rows)), "split")
    assert got == {one: "dev", two: "test", zero: "dev"}


def test_split_is_decided_by_lowest_row_id_of_component():
    dev_id, test_id = id_with_bucket(0), id_with_bucket(5)
    low, high = sorted([dev_id, test_id])
    out = splits.build_splits(frame([(high, "c", "q1"), (low, "c", "q2")]))
    expected = "dev" if low == dev_id else "test"
    assert set(out["split"]) == {expected}
    assert set(out["component"]) == {low}


def test_output_columns_sorted_and_input_untouched_and_deterministic():
    data = frame([("c", "c3", "q3"), ("a", "c1", "q1"), ("b", "c1", "q2")])
    before = data.copy(deep=True)
    out = splits.build_splits(data)
    assert list(out.columns) == ["row_id", "component", "split"]
    assert out["row_id"].tolist() == ["a", "b", "c"]
    pd.testing.assert_frame_equal(data, before)
    pd.testing.assert_frame_equal(out, splits.build_splits(data))


def write_cleaned(path: Path, rows: list[tuple[str, str, str]]) -> None:
    frame(rows).to_parquet(path)


def test_main_writes_lf_utf8_csv_and_one_line(tmp_path, capsys):
    cleaned, out = tmp_path / "c.parquet", tmp_path / "s.csv"
    write_cleaned(cleaned, [("é1", "c1", "q1"), ("b", "c1", "q2")])
    assert splits.main(["--cleaned", str(cleaned), "--out", str(out)]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 1
    raw = out.read_bytes()
    assert b"\r" not in raw
    text = raw.decode("utf-8")
    assert text.startswith("row_id,component,split\n")
    assert pd.read_csv(io.StringIO(text), dtype=str)["component"].tolist() == ["b", "b"]


def assert_one_friendly_line(argv: list[str], capsys) -> None:
    assert splits.main(argv) == 1
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert "Traceback" not in lines[0]


def test_main_missing_cleaned_file(tmp_path, capsys):
    argv = ["--cleaned", str(tmp_path / "none.parquet"), "--out", str(tmp_path / "s.csv")]
    assert_one_friendly_line(argv, capsys)


def test_main_cleaned_path_is_directory(tmp_path, capsys):
    assert_one_friendly_line(["--cleaned", str(tmp_path), "--out", str(tmp_path / "s.csv")], capsys)


def test_main_missing_columns(tmp_path, capsys):
    cleaned = tmp_path / "c.parquet"
    pd.DataFrame({"row_id": ["a"]}).to_parquet(cleaned)
    assert_one_friendly_line(["--cleaned", str(cleaned), "--out", str(tmp_path / "s.csv")], capsys)


def test_main_out_is_directory(tmp_path, capsys):
    cleaned = tmp_path / "c.parquet"
    write_cleaned(cleaned, [("a", "c1", "q1")])
    assert_one_friendly_line(["--cleaned", str(cleaned), "--out", str(tmp_path)], capsys)


@pytest.mark.needs_data
def test_committed_splits_match_real_cleaned_data(tmp_path):
    assert build_reference.main(["--out-dir", str(tmp_path)]) == 0
    cleaned = pd.read_parquet(tmp_path / "cleaned_10k_reports.parquet")
    assert len(cleaned) == REAL_ROWS
    expected = splits.build_splits(cleaned)
    committed = pd.read_csv(COMMITTED, dtype=str, encoding="utf-8")
    pd.testing.assert_frame_equal(committed, expected.astype(str))
    per_component = expected.drop_duplicates("component")
    assert 0.15 <= (per_component["split"] == "dev").mean() <= 0.25
