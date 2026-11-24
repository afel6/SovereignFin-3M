"""Contract tests for atlas_kv/api.py and schema.py (docs/contracts.md, knowledge API)."""

import inspect
import typing

import pytest
from pydantic import ValidationError

from atlas_kv import api
from atlas_kv.schema import Relation

SPEC_RELATIONS = {
    "operates_in_sector",
    "headquartered_in",
    "has_segment",
    "offers_product",
    "competes_with",
    "faces_risk",
    "uses_technology",
    "has_subsidiary",
    "listed_on",
    "supplied_by",
    "pays_late_to",
    "shares_vendor_with",
}


def triple(**kw):
    base = {
        "head": "Acme",
        "relation": "faces_risk",
        "tail": "cloud computing",
        "ticker": "ACM",
        "source": "tenk",
        "source_id": "r1",
        "confidence": 0.5,
        "extractor": "rules",
    }
    return api.Triple(**{**base, **kw})


def test_relation_values_are_the_spec_set():
    """Fails if a relation is missing, extra (e.g. a numeric one) or renamed."""
    assert {r.value for r in Relation} == SPEC_RELATIONS
    assert len(list(Relation)) == 12


def test_relation_is_str_enum():
    """Fails if Relation is not a string enum or lookup by value fails."""
    assert Relation("faces_risk") == "faces_risk"
    assert isinstance(Relation("faces_risk"), str)


def test_triple_fields_exact():
    """Fails if a field is renamed, added or dropped."""
    assert list(api.Triple.model_fields) == [
        "head",
        "relation",
        "tail",
        "ticker",
        "source",
        "source_id",
        "confidence",
        "extractor",
    ]


def test_fact_fields_exact():
    """Fails if a field is renamed, added or dropped."""
    assert list(api.Fact.model_fields) == ["head", "relation", "tail", "source", "score"]


def test_triple_relation_from_string():
    """Fails if a plain string is not turned into the enum."""
    assert triple(relation="pays_late_to").relation == Relation("pays_late_to")


def test_triple_unknown_relation_refused():
    """Fails if relations are not restricted to the enum."""
    with pytest.raises(ValidationError):
        triple(relation="revenue_of")


@pytest.mark.parametrize(
    "tail", ["1,000", "$5.2", "12%", "-3", "2023", "$4.5 billion", "3 Million"]
)
def test_triple_numeric_tail_refused(tail):
    """Fails if a numeric tail passes, incl. digits plus only a scale word (thousand..trillion)."""
    with pytest.raises(ValidationError):
        triple(tail=tail)


@pytest.mark.parametrize("tail", ["3M", "Boeing 737", "10-K", "NYSE", "cloud computing"])
def test_triple_non_numeric_tail_kept(tail):
    """Fails if a tail holding a letter next to digits is refused."""
    assert triple(tail=tail).tail == tail


@pytest.mark.parametrize("c", [0, 1, 0.5])
def test_triple_confidence_in_range(c):
    """Fails if a bound (0 or 1) is refused."""
    assert triple(confidence=c).confidence == c


@pytest.mark.parametrize("c", [-0.01, 1.01, float("nan")])
def test_triple_confidence_out_of_range_refused(c):
    """Fails if out-of-range or NaN confidence is accepted."""
    with pytest.raises(ValidationError):
        triple(confidence=c)


def test_triple_source_other_refused():
    """Fails if sources are not restricted."""
    with pytest.raises(ValidationError):
        triple(source="web")


@pytest.mark.parametrize("s", ["synthetic_db", "tenk", "reference"])
def test_triple_source_ok(s):
    """Fails if a spec source is refused."""
    assert triple(source=s).source == s


def test_triple_ticker_none_ok():
    """Fails if a null ticker is refused."""
    assert triple(ticker=None).ticker is None


def test_lookup_signature():
    """Fails if names, order, defaults or annotations differ from the spec."""
    sig = inspect.signature(api.lookup)
    assert list(sig.parameters) == ["query", "tickers", "k"]
    assert sig.parameters["tickers"].default is None
    assert sig.parameters["k"].default == 8
    hints = typing.get_type_hints(api.lookup)
    assert hints["query"] is str
    assert hints["tickers"] == (list[str] | None)
    assert hints["k"] is int
    assert hints["return"] == list[api.Fact]


def test_lookup_not_implemented_names_slice():
    """Fails if lookup returns, or the message omits the slice S5."""
    with pytest.raises(NotImplementedError, match="S5"):
        api.lookup("risk")
