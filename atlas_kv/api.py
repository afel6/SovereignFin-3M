"""Public interface of the knowledge graph: triples and fact lookup."""

import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from atlas_kv.schema import Relation

LOOKUP_NOT_IMPLEMENTED = "atlas_kv.lookup is not implemented yet (slice S5)."
_NUMBER_WORDS = re.compile(
    r"(?<![a-z])(?:thousand|million|billion|trillion)s?(?![a-z])", re.IGNORECASE
)


def _is_numeric(text: str) -> bool:
    """True when the text, ignoring magnitude words, has a digit and no letter."""
    rest = _NUMBER_WORDS.sub("", text)
    return any(c.isdigit() for c in rest) and not any(c.isalpha() for c in rest)


class Triple(BaseModel):
    """One knowledge-graph statement with its provenance."""

    head: str
    relation: Relation
    tail: str
    ticker: str | None
    source: Literal["synthetic_db", "tenk", "reference"]
    source_id: str
    confidence: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    extractor: str

    @field_validator("tail")
    @classmethod
    def _tail_not_numeric(cls, tail: str) -> str:
        if _is_numeric(tail):
            raise ValueError("tail must be a non-numeric string")
        return tail


class Fact(BaseModel):
    """A fact returned by lookup."""

    head: str
    relation: str
    tail: str
    source: str
    score: float


def lookup(query: str, tickers: list[str] | None = None, k: int = 8) -> list[Fact]:
    """Return the k facts that best match the query, optionally limited to tickers."""
    raise NotImplementedError(LOOKUP_NOT_IMPLEMENTED)
