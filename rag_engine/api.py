"""Public interface of the retrieval engine: passage search and index building."""

from pathlib import Path

from pydantic import BaseModel

SEARCH_NOT_IMPLEMENTED = "rag_engine.search is not implemented yet (slice S2)."
BUILD_INDEX_NOT_IMPLEMENTED = "rag_engine.build_index is not implemented yet (slice S2)."


class Passage(BaseModel):
    """A filing passage returned by search."""

    context_id: str
    ticker: str
    filing: str
    text: str
    score: float  # higher = more relevant
    row_ids: list[str]  # row_ids of the cleaned rows that share this context_id


def search(query: str, tickers: list[str] | None = None, k: int = 5) -> list[Passage]:
    """Return the k passages most relevant to the query, optionally limited to tickers."""
    raise NotImplementedError(SEARCH_NOT_IMPLEMENTED)


def build_index(cleaned_parquet: Path) -> int:
    """Build the index offline: one point per unique context_id, never the answer column.

    Returns the number of points written.
    """
    raise NotImplementedError(BUILD_INDEX_NOT_IMPLEMENTED)
