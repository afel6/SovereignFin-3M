"""Reference cleaner for the 10-K dataset (docs/contracts.md, section 1).

Implements "Cleaned output schema", "Drop rules" and the derived columns. The validator
recomputes every cleaned value with this module.
"""

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_pipeline.contract.normalize import normalize_text

RAW_COLUMNS = ("question", "answer", "context", "ticker", "filing")
TEXT_COLUMNS = ("question", "answer", "context")
CLEANED_COLUMNS = (
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
FLAG_COLUMNS = ("is_table", "mentions_company")


def _arrow_type(column: str) -> pa.DataType:
    if column == "raw_index":
        return pa.int32()
    return pa.bool_() if column in FLAG_COLUMNS else pa.string()


CLEANED_SCHEMA = pa.schema([pa.field(name, _arrow_type(name)) for name in CLEANED_COLUMNS])
DROP_LOG_COLUMNS = ("raw_index", "row_id", "reason", "duplicate_of")
CLEANED_FILENAME = "cleaned_10k_reports.parquet"
DROP_LOG_FILENAME = "drop_log.csv"
TICKERS_CSV = Path(__file__).resolve().parents[1] / "reference" / "tickers.csv"

FIELD_SEPARATOR = "\x1f"
ROW_ID_HASH_CHARS = 12
CONTEXT_ID_HASH_CHARS = 16
TABLE_PIPE_MIN = 2
ALIAS_SEPARATOR = "|"
_CYRILLIC = re.compile("[Ѐ-ԯ]")


def make_row_id(
    raw_index: int, question: str, answer: str, context: str, ticker: str, filing: str
) -> str:
    """Return "<raw_index:04d>-<12 hex of sha256 of the raw fields joined by U+001F>"."""
    joined = FIELD_SEPARATOR.join((question, answer, context, ticker, filing))
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return f"{raw_index:04d}-{digest[:ROW_ID_HASH_CHARS]}"


def make_context_id(cleaned_context: str) -> str:
    """Return the first 16 hex characters of sha256 of the cleaned context."""
    return hashlib.sha256(cleaned_context.encode("utf-8")).hexdigest()[:CONTEXT_ID_HASH_CHARS]


def load_tickers(path: Path = TICKERS_CSV) -> pd.DataFrame:
    """Read tickers.csv as strings; empty cells stay empty strings."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def aliases_by_ticker(tickers: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Map each ticker to its aliases (the "aliases" column split on "|")."""
    return {
        ticker: tuple(a for a in aliases.split(ALIAS_SEPARATOR) if a)
        for ticker, aliases in zip(tickers["ticker"], tickers["aliases"], strict=True)
    }


def mentions_company(question: str, aliases: Iterable[str]) -> bool:
    """True when a casefolded alias occurs in the casefolded question, not next to [a-z0-9]."""
    folded = question.casefold()
    for alias in aliases:
        pattern = "(?<![a-z0-9])" + re.escape(alias.casefold()) + "(?![a-z0-9])"
        if re.search(pattern, folded):
            return True
    return False


def read_raw(path: Path) -> pd.DataFrame:
    """Read the five raw columns of the parquet file as Python strings."""
    table = pq.read_table(path)
    missing = [c for c in RAW_COLUMNS if c not in table.column_names]
    if missing:
        raise ValueError(f"raw file lacks the columns {', '.join(missing)}")
    data = {
        c: ["" if v is None else str(v) for v in table.column(c).to_pylist()] for c in RAW_COLUMNS
    }
    return pd.DataFrame(data, columns=list(RAW_COLUMNS))


def _drop_reason(
    fields: tuple[str, ...], normalized: tuple[str, ...], seen: dict[tuple, str]
) -> tuple[str, str] | None:
    """Return (reason, duplicate_of) for a row to drop, or None to keep it."""
    if "" in normalized:
        return "empty_field", ""
    if any(_CYRILLIC.search(f) for f in fields):
        return "non_english_text", ""
    key = (fields[3], fields[4], *normalized)
    if key in seen:
        return "exact_duplicate", seen[key]
    return None


def clean(
    raw: pd.DataFrame, tickers: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cleaned, drop_log) for the raw frame; the input is not changed."""
    tickers = load_tickers() if tickers is None else tickers
    names = dict(zip(tickers["ticker"], tickers["company_name"], strict=True))
    aliases = aliases_by_ticker(tickers)
    columns = [raw[c].tolist() for c in RAW_COLUMNS]
    seen: dict[tuple, str] = {}
    qgroups: dict[tuple[str, str], str] = {}
    kept: list[dict] = []
    dropped: list[tuple[int, str, str, str]] = []
    for index, fields in enumerate(zip(*columns, strict=True)):
        row_id = make_row_id(index, *fields)
        normalized = tuple(normalize_text(f) for f in fields[:3])
        verdict = _drop_reason(fields, normalized, seen)
        if verdict is not None:
            dropped.append((index, row_id, *verdict))
            continue
        seen[(fields[3], fields[4], *normalized)] = row_id
        kept.append(_kept_row(index, row_id, fields, normalized, names, aliases, qgroups))
    return _frames(kept, dropped)


def _kept_row(
    index: int,
    row_id: str,
    fields: tuple[str, ...],
    normalized: tuple[str, ...],
    names: dict[str, str],
    aliases: dict[str, tuple[str, ...]],
    qgroups: dict[tuple[str, str], str],
) -> dict:
    question, answer, context = normalized
    ticker, filing = fields[3], fields[4]
    if ticker not in names:
        raise ValueError(f"ticker {ticker!a} (raw row {index}) is not in tickers.csv")
    group = qgroups.setdefault((ticker, question.casefold()), row_id)
    return {
        "row_id": row_id,
        "raw_index": index,
        "ticker": ticker,
        "company_name": names[ticker],
        "filing": filing,
        "question": question,
        "answer": answer,
        "context": context,
        "context_id": make_context_id(context),
        "qgroup_id": group,
        "is_table": context.count("|") >= TABLE_PIPE_MIN,
        "mentions_company": mentions_company(question, aliases.get(ticker, ())),
    }


def _frames(
    kept: list[dict], dropped: list[tuple[int, str, str, str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = pd.DataFrame(kept, columns=list(CLEANED_COLUMNS))
    cleaned["raw_index"] = cleaned["raw_index"].astype("int32")
    drop_log = pd.DataFrame(dropped, columns=list(DROP_LOG_COLUMNS))
    return cleaned, drop_log


def write_outputs(cleaned: pd.DataFrame, drop_log: pd.DataFrame, out_dir: Path) -> None:
    """Write the cleaned parquet (explicit schema) and drop_log.csv into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(cleaned, schema=CLEANED_SCHEMA, preserve_index=False)
    pq.write_table(table, out_dir / CLEANED_FILENAME)
    drop_log.to_csv(out_dir / DROP_LOG_FILENAME, index=False, encoding="utf-8", lineterminator="\n")
