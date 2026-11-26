"""Tests for M3 baseline vector search (rag_engine/contrib/rag_search.py)."""

from pathlib import Path

import pandas as pd
import pytest
from qdrant_client import QdrantClient

from rag_engine.api import Passage
from rag_engine.contrib import build_index, rag_search, search
from rag_engine.contrib.tests.conftest import (
    FIXTURE_ROWS,
    token_index,
    write_cleaned_parquet,
)


def test_fixture_words_have_distinct_hash_indices() -> None:
    """No two distinct words in the fixture contexts and test queries land on the same index."""
    queries = [
        "iphones",
        "revenue",
        "cloud",
        "azure subscriptions cloud",
        "zebracrossing",
        "products",
    ]
    words = {t for row in FIXTURE_ROWS for t in row["context"].lower().split()}
    words |= {t for q in queries for t in q.lower().split()}
    indices = {token_index(w) for w in words}
    assert len(indices) == len(words)


def test_duplicate_contexts_collapse_and_point_count_in_store(
    fixture_parquet: Path, patched_rag: Path
) -> None:
    """build_index returns 4 and stores 4 points in COLLECTION."""
    written = build_index(fixture_parquet)
    assert written == 4

    client = QdrantClient(path=str(patched_rag))
    try:
        count = client.count(collection_name=rag_search.COLLECTION, exact=True).count
    finally:
        client.close()
    assert count == 4


def test_row_ids_grouped(fixture_parquet: Path, patched_rag: Path) -> None:
    """search('iphones', k=5) groups row_ids sharing context_id a000000000000001."""
    build_index(fixture_parquet)
    results = search("iphones", k=5)
    assert results[0].context_id == "a000000000000001"
    assert results[0].row_ids == ["0000-aaaaaaaaaaaa", "0001-bbbbbbbbbbbb"]


def test_row_ids_sorted_when_input_reversed(
    tmp_path: Path, fixture_df: pd.DataFrame, patched_rag: Path
) -> None:
    """Reversed input rows still yield ascending row_ids on every context."""
    reversed_df = fixture_df.iloc[::-1].reset_index(drop=True)
    parquet_path = write_cleaned_parquet(reversed_df, tmp_path / "reversed.parquet")
    build_index(parquet_path)

    client = QdrantClient(path=str(patched_rag))
    try:
        points, _ = client.scroll(
            collection_name=rag_search.COLLECTION,
            limit=100,
            with_payload=True,
        )
    finally:
        client.close()

    by_cid = {
        point.payload["context_id"]: point.payload["row_ids"]
        for point in points
        if point.payload is not None
    }
    assert by_cid["ffffffffffffffff"] == [
        "0003-dddddddddddd",
        "0004-eeeeeeeeeeee",
    ]
    assert by_cid["a000000000000001"] == [
        "0000-aaaaaaaaaaaa",
        "0001-bbbbbbbbbbbb",
    ]


def test_ticker_and_filing_from_lowest_raw_index(tmp_path: Path, patched_rag: Path) -> None:
    """Ticker and filing come from the row with the lowest raw_index."""
    rows = [
        {
            **FIXTURE_ROWS[0],
            "row_id": "0007-777777777777",
            "raw_index": 7,
            "ticker": "BBB",
            "filing": "2022_10K",
            "context_id": "c000000000000009",
            "qgroup_id": "0002-222222222222",
        },
        {
            **FIXTURE_ROWS[0],
            "row_id": "0002-222222222222",
            "raw_index": 2,
            "ticker": "AAA",
            "filing": "2021_10K",
            "context_id": "c000000000000009",
            "qgroup_id": "0002-222222222222",
        },
    ]
    parquet_path = write_cleaned_parquet(pd.DataFrame(rows), tmp_path / "lowest.parquet")
    build_index(parquet_path)

    results = search("iphones", k=5)
    assert len(results) == 1
    assert results[0].ticker == "AAA"
    assert results[0].filing == "2021_10K"
    assert results[0].row_ids == ["0002-222222222222", "0007-777777777777"]


def test_point_id_is_unsigned_64bit_int_of_context_id(
    fixture_parquet: Path, patched_rag: Path
) -> None:
    """Point id is int(context_id, 16), supporting unsigned 64-bit values like ffffffffffffffff."""
    build_index(fixture_parquet)
    expected_id = int("ffffffffffffffff", 16)

    client = QdrantClient(path=str(patched_rag))
    try:
        points = client.retrieve(
            collection_name=rag_search.COLLECTION,
            ids=[expected_id],
            with_payload=True,
        )
    finally:
        client.close()

    assert len(points) == 1
    assert points[0].payload is not None
    assert points[0].payload["context_id"] == "ffffffffffffffff"


def test_answer_never_indexed_and_payload_fields_exact(
    fixture_parquet: Path, patched_rag: Path
) -> None:
    """The answer and question columns are never stored or embedded, and payload keys match."""
    build_index(fixture_parquet)

    client = QdrantClient(path=str(patched_rag))
    try:
        points, _ = client.scroll(
            collection_name=rag_search.COLLECTION,
            limit=100,
            with_payload=True,
        )
    finally:
        client.close()

    expected_contexts = {row["context_id"]: row["context"] for row in FIXTURE_ROWS}
    assert len(points) == 4
    for point in points:
        payload = point.payload
        assert payload is not None
        assert set(payload.keys()) == {"context_id", "ticker", "filing", "text", "row_ids"}
        assert payload["text"] == expected_contexts[payload["context_id"]]
        for value in payload.values():
            assert "zebracrossing" not in str(value)

    results = search("zebracrossing", k=5)
    for passage in results:
        assert "zebracrossing" not in passage.text
    assert all(p.score == pytest.approx(0.0) for p in results)

    question_results = search("products", k=5)
    assert all(p.score == pytest.approx(0.0) for p in question_results)


def test_ticker_filter_and_no_filter_and_unknown_ticker(
    fixture_parquet: Path, patched_rag: Path
) -> None:
    """Ticker filter limits before top-k; no filter returns all; unknown ticker returns []."""
    build_index(fixture_parquet)

    msft_results = search("revenue", tickers=["MSFT"], k=5)
    assert len(msft_results) == 2
    assert {p.ticker for p in msft_results} == {"MSFT"}
    assert len(search("iphones", tickers=["MSFT"], k=1)) == 1

    all_results = search("revenue", k=5)
    assert len(all_results) == 4
    assert {p.ticker for p in all_results} == {"AAPL", "MSFT"}

    unknown_results = search("revenue", tickers=["ZZZZ"])
    assert unknown_results == []


def test_k_respected_and_scores_descending_and_return_type(
    fixture_parquet: Path, patched_rag: Path
) -> None:
    """k caps results, scores are non-increasing, and items are rag_engine.api.Passage."""
    build_index(fixture_parquet)

    top2 = search("cloud", k=2)
    assert len(top2) == 2

    top10 = search("cloud", k=10)
    assert len(top10) == 4

    ranked = search("azure subscriptions cloud", k=4)
    assert len(ranked) == 4
    assert ranked[0].context_id == "ffffffffffffffff"
    scores = [p.score for p in ranked]
    assert scores == sorted(scores, reverse=True)
    for item in ranked:
        assert type(item) is Passage


def test_k_must_be_positive(patched_rag: Path) -> None:
    """search raises ValueError when k is 0 or negative."""
    with pytest.raises(ValueError, match="k must be positive"):
        search("cloud", k=0)


def test_rebuild_is_idempotent(fixture_parquet: Path, patched_rag: Path) -> None:
    """Calling build_index twice returns 4 both times and leaves 4 points in the store."""
    assert build_index(fixture_parquet) == 4
    assert build_index(fixture_parquet) == 4

    client = QdrantClient(path=str(patched_rag))
    try:
        count = client.count(collection_name=rag_search.COLLECTION, exact=True).count
    finally:
        client.close()
    assert count == 4


@pytest.mark.needs_data
def test_real_data_recall_at_5() -> None:
    """Build real index on cleaned 10-K dataset and check recall@5 on 100 sampled questions."""
    cleaned_parquet = Path("data/processed/cleaned_10k_reports.parquet")
    df = pd.read_parquet(cleaned_parquet)
    written = build_index(cleaned_parquet)
    assert written == df["context_id"].nunique()

    sample = df.sample(100, random_state=20231231)
    hits_with_ticker = 0
    hits_without_ticker = 0

    for row in sample.itertuples(index=False):
        passages_filtered = search(row.question, tickers=[row.ticker], k=5)
        if any(p.context_id == row.context_id for p in passages_filtered):
            hits_with_ticker += 1

        passages_unfiltered = search(row.question, k=5)
        if any(p.context_id == row.context_id for p in passages_unfiltered):
            hits_without_ticker += 1

    print(
        f"Recall@5 on 100 sampled questions: "
        f"with ticker filter = {hits_with_ticker}/100, "
        f"without ticker filter = {hits_without_ticker}/100"
    )
    assert hits_with_ticker >= 70
