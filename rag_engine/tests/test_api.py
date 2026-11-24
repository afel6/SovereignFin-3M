"""Contract tests for rag_engine/api.py (docs/contracts.md, RAG engine API)."""

import inspect
import typing
from pathlib import Path

import pytest

from rag_engine import api


def test_passage_fields_exact():
    """Fails if a field is renamed, added or dropped."""
    assert list(api.Passage.model_fields) == [
        "context_id",
        "ticker",
        "filing",
        "text",
        "score",
        "row_ids",
    ]


def test_passage_field_types():
    """Fails if score is not float or row_ids is not list[str]."""
    hints = typing.get_type_hints(api.Passage)
    assert hints["score"] is float
    assert hints["row_ids"] == list[str]
    assert hints["context_id"] is str


def test_search_signature():
    """Fails if names, order, defaults or annotations differ from the spec."""
    sig = inspect.signature(api.search)
    assert list(sig.parameters) == ["query", "tickers", "k"]
    assert sig.parameters["tickers"].default is None
    assert sig.parameters["k"].default == 5
    assert sig.parameters["query"].default is inspect.Parameter.empty
    hints = typing.get_type_hints(api.search)
    assert hints["query"] is str
    assert hints["tickers"] == (list[str] | None)
    assert hints["k"] is int
    assert hints["return"] == list[api.Passage]


def test_build_index_signature():
    """Fails if build_index does not take a Path and return int."""
    sig = inspect.signature(api.build_index)
    assert list(sig.parameters) == ["cleaned_parquet"]
    hints = typing.get_type_hints(api.build_index)
    assert hints["cleaned_parquet"] is Path
    assert hints["return"] is int


def test_search_not_implemented_names_slice():
    """Fails if search returns, or the message omits the slice S2."""
    with pytest.raises(NotImplementedError, match="S2"):
        api.search("revenue")


def test_build_index_not_implemented_names_slice():
    """Fails if build_index returns, or the message omits the slice S2."""
    with pytest.raises(NotImplementedError, match="S2"):
        api.build_index(Path("x.parquet"))
