"""Baseline dense-vector retrieval over cleaned 10-K contexts using fastembed and Qdrant."""

from pathlib import Path

import pandas as pd
from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from rag_engine.api import Passage

INDEX_DIR = Path("data/contrib_index")
COLLECTION = "tenk_contexts"
VECTOR_SIZE = 384
MODEL_NAME = "BAAI/bge-small-en-v1.5"

_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    """Lazily initialize the fastembed TextEmbedding model so imports stay offline."""
    global _model
    if _model is None:
        _model = TextEmbedding(model_name=MODEL_NAME)
    return _model


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed texts into 384-dimensional dense vectors with fastembed."""
    if not texts:
        return []
    model = _get_model()
    return [vec.tolist() for vec in model.embed(texts)]


def build_index(cleaned_parquet: Path) -> int:
    """Build the local Qdrant index from the cleaned 10-K parquet file.

    Writes one point per unique context_id, embedding only the cleaned context text
    (never question or answer). Recreates the collection on each call and returns
    the number of points written.
    """
    df = pd.read_parquet(
        cleaned_parquet,
        columns=["row_id", "raw_index", "ticker", "filing", "context", "context_id"],
    )
    sorted_df = df.sort_values("raw_index", kind="mergesort")

    contexts: list[str] = []
    payloads: list[dict[str, object]] = []
    point_ids: list[int] = []

    for context_id, group in sorted_df.groupby("context_id", sort=False):
        first_row = group.iloc[0]
        cid = str(context_id)
        text = str(first_row["context"])
        row_ids = sorted(str(rid) for rid in group["row_id"].tolist())

        point_ids.append(int(cid, 16))
        contexts.append(text)
        payloads.append(
            {
                "context_id": cid,
                "ticker": str(first_row["ticker"]),
                "filing": str(first_row["filing"]),
                "text": text,
                "row_ids": row_ids,
            }
        )

    vectors = _embed(contexts)
    points = [
        models.PointStruct(id=pid, vector=vec, payload=payload)
        for pid, vec, payload in zip(point_ids, vectors, payloads, strict=True)
    ]

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(INDEX_DIR))
    try:
        if client.collection_exists(collection_name=COLLECTION):
            client.delete_collection(collection_name=COLLECTION)
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
        if points:
            client.upsert(collection_name=COLLECTION, points=points)
    finally:
        client.close()

    return len(points)


def search(query: str, tickers: list[str] | None = None, k: int = 5) -> list[Passage]:
    """Return at most k Passages sorted by cosine similarity score descending."""
    if k <= 0:
        raise ValueError("k must be positive")
    if tickers is not None and len(tickers) == 0:
        return []

    query_vector = _embed([query])[0]
    query_filter: models.Filter | None = None
    if tickers is not None:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="ticker",
                    match=models.MatchAny(any=list(tickers)),
                )
            ]
        )

    client = QdrantClient(path=str(INDEX_DIR))
    try:
        response = client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            query_filter=query_filter,
            limit=k,
            with_payload=True,
        )
    finally:
        client.close()

    passages: list[Passage] = []
    for point in response.points:
        payload = point.payload or {}
        passages.append(
            Passage(
                context_id=str(payload["context_id"]),
                ticker=str(payload["ticker"]),
                filing=str(payload["filing"]),
                text=str(payload["text"]),
                score=float(point.score),
                row_ids=[str(rid) for rid in payload["row_ids"]],
            )
        )
    passages.sort(key=lambda p: p.score, reverse=True)
    return passages
