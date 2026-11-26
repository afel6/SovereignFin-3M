"""Shared fixtures for M3 baseline vector search unit tests.

The repository runs pytest with --import-mode=importlib, so `from conftest import ...` fails.
Test files import the helpers by full path:
    from rag_engine.contrib.tests.conftest import fake_embed, token_index, write_cleaned_parquet
"""

import hashlib
import math
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rag_engine.contrib import rag_search

CLEANED_SCHEMA = pa.schema(
    [
        ("row_id", pa.string()),
        ("raw_index", pa.int32()),
        ("ticker", pa.string()),
        ("company_name", pa.string()),
        ("filing", pa.string()),
        ("question", pa.string()),
        ("answer", pa.string()),
        ("context", pa.string()),
        ("context_id", pa.string()),
        ("qgroup_id", pa.string()),
        ("is_table", pa.bool_()),
        ("mentions_company", pa.bool_()),
    ]
)

FIXTURE_ROWS = [
    {
        "row_id": "0000-aaaaaaaaaaaa",
        "raw_index": 0,
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "filing": "2022_10K",
        "question": "What products does Apple sell?",
        "answer": "zebracrossing is the answer",
        "context": "Apple sells iphones and wearable devices worldwide",
        "context_id": "a000000000000001",
        "qgroup_id": "0000-aaaaaaaaaaaa",
        "is_table": False,
        "mentions_company": True,
    },
    {
        "row_id": "0001-bbbbbbbbbbbb",
        "raw_index": 1,
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "filing": "2022_10K",
        "question": "Where does Apple sell devices?",
        "answer": "second answer",
        "context": "Apple sells iphones and wearable devices worldwide",
        "context_id": "a000000000000001",
        "qgroup_id": "0001-bbbbbbbbbbbb",
        "is_table": False,
        "mentions_company": True,
    },
    {
        "row_id": "0002-cccccccccccc",
        "raw_index": 2,
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "filing": "2022_10K",
        "question": "What is included in Apple services revenue?",
        "answer": "third answer",
        "context": "Apple services revenue includes advertising and cloud",
        "context_id": "b000000000000002",
        "qgroup_id": "0002-cccccccccccc",
        "is_table": False,
        "mentions_company": True,
    },
    {
        "row_id": "0003-dddddddddddd",
        "raw_index": 3,
        "ticker": "MSFT",
        "company_name": "Microsoft Corporation",
        "filing": "2022_10K",
        "question": "What software does Microsoft sell?",
        "answer": "fourth answer",
        "context": "Microsoft sells cloud software and azure subscriptions",
        "context_id": "ffffffffffffffff",
        "qgroup_id": "0003-dddddddddddd",
        "is_table": False,
        "mentions_company": True,
    },
    {
        "row_id": "0004-eeeeeeeeeeee",
        "raw_index": 4,
        "ticker": "MSFT",
        "company_name": "Microsoft Corporation",
        "filing": "2022_10K",
        "question": "What subscriptions does Microsoft offer?",
        "answer": "fifth answer",
        "context": "Microsoft sells cloud software and azure subscriptions",
        "context_id": "ffffffffffffffff",
        "qgroup_id": "0004-eeeeeeeeeeee",
        "is_table": False,
        "mentions_company": True,
    },
    {
        "row_id": "0005-ffffffffffff",
        "raw_index": 5,
        "ticker": "MSFT",
        "company_name": "Microsoft Corporation",
        "filing": "2022_10K",
        "question": "How did Microsoft gaming revenue change?",
        "answer": "sixth answer",
        "context": "Microsoft gaming revenue grew with xbox consoles",
        "context_id": "d000000000000004",
        "qgroup_id": "0005-ffffffffffff",
        "is_table": False,
        "mentions_company": True,
    },
]


def token_index(token: str, dim: int = rag_search.VECTOR_SIZE) -> int:
    """Deterministic hash bucket for a lowercased token."""
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % dim


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic unit vector counting lowercased whitespace-separated tokens."""
    vectors: list[list[float]] = []
    for text in texts:
        vec = [0.0] * rag_search.VECTOR_SIZE
        for token in text.lower().split():
            vec[token_index(token)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        vectors.append(vec)
    return vectors


def write_cleaned_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame with the explicit cleaned parquet schema."""
    table = pa.Table.from_pandas(df, schema=CLEANED_SCHEMA, preserve_index=False)
    pq.write_table(table, path)
    return path


@pytest.fixture()
def fixture_df() -> pd.DataFrame:
    """The 6-row fixture DataFrame from the M3 task brief."""
    return pd.DataFrame(FIXTURE_ROWS)


@pytest.fixture()
def fixture_parquet(tmp_path: Path, fixture_df: pd.DataFrame) -> Path:
    """Write the 6-row fixture DataFrame to a parquet file in tmp_path."""
    return write_cleaned_parquet(fixture_df, tmp_path / "cleaned.parquet")


@pytest.fixture()
def patched_rag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect INDEX_DIR to tmp_path / 'index' and replace _embed with fake_embed."""
    index_dir = tmp_path / "index"
    monkeypatch.setattr(rag_search, "INDEX_DIR", index_dir)
    monkeypatch.setattr(rag_search, "_embed", fake_embed)
    return index_dir
