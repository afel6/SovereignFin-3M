"""Contract tests for the HTTP models (docs/contracts.md section 3)."""

import pytest
from pydantic import ValidationError

from orchestrator.api_contract import (
    AskRequest,
    AskResponse,
    ChartSpec,
    Citation,
    ErrorDetail,
    HealthComponents,
    HealthResponse,
)


def ask(**kw):
    return AskRequest(**{"question": "q", **kw})


def citation(**kw):
    return Citation(**{"id": "1", "source": "rag", "ticker": None, "text": "t", **kw})


def chart(**kw):
    base = {"chart_type": "bar", "title": "t", "x_field": "a", "y_fields": ["b"], "data": []}
    return ChartSpec(**{**base, **kw})


def response(**kw):
    base = {
        "answer": "a",
        "outcome": "answered",
        "citations": [],
        "chart": None,
        "route": ["rag"],
        "latency_ms": 0,
        "mode": "cloud",
        "warnings": [],
    }
    return AskResponse(**{**base, **kw})


# ---- AskRequest ----


@pytest.mark.parametrize("q", ["", "   ", "\t\n"])
def test_question_blank_refused(q):
    """Fails if blank text is accepted or the strip happens after the min-length check."""
    with pytest.raises(ValidationError):
        ask(question=q)


def test_question_is_stripped():
    """Fails if the stored question keeps its surrounding spaces."""
    assert ask(question=" a ").question == "a"


def test_question_2000_ok_2001_refused():
    """Fails if the limit is off by one."""
    assert len(ask(question="x" * 2000).question) == 2000
    with pytest.raises(ValidationError):
        ask(question="x" * 2001)


def test_question_padded_2000_ok():
    """Fails if the length is checked before stripping."""
    q = ask(question="  " + "x" * 2000 + "  ")
    assert len(q.question) == 2000


def test_question_2001_after_strip_refused_even_padded():
    """Fails if padding lets an over-long question through."""
    with pytest.raises(ValidationError):
        ask(question=" " + "x" * 2001 + " ")


def test_mode_default_cloud():
    """Fails if the default mode is not cloud."""
    assert ask().mode == "cloud"


def test_mode_local_ok():
    """Fails if local is refused."""
    assert ask(mode="local").mode == "local"


@pytest.mark.parametrize("m", ["Cloud", "remote", None])
def test_mode_invalid_refused(m):
    """Fails if mode is case-folded or None is accepted."""
    with pytest.raises(ValidationError):
        ask(mode=m)


def test_thread_id_default_none():
    """Fails if thread_id has another default."""
    assert ask().thread_id is None


@pytest.mark.parametrize("t", ["abc_DEF-123", "a" * 64, ""])
def test_thread_id_accepted(t):
    """Fails if a valid id (0 to 64 characters, so the empty string too) is refused."""
    assert ask(thread_id=t).thread_id == t


@pytest.mark.parametrize("t", ["a" * 65, "a/b", "a b", "é", "abc\n"])
def test_thread_id_refused(t):
    """Fails if length or charset is unchecked, or a trailing newline passes (regex $)."""
    with pytest.raises(ValidationError):
        ask(thread_id=t)


def test_question_non_string_json_refused():
    """Fails if a number is coerced to a string."""
    with pytest.raises(ValidationError):
        AskRequest.model_validate_json('{"question": 123}')


def test_ask_request_unknown_key_ignored():
    """Fails if an unknown key is refused or kept."""
    r = AskRequest.model_validate({"question": "q", "extra": 1})
    assert r.question == "q"
    assert not hasattr(r, "extra")


# ---- Citation ----


def test_citation_text_500_ok_501_refused():
    """Fails if the text limit is off by one."""
    assert len(citation(text="x" * 500).text) == 500
    with pytest.raises(ValidationError):
        citation(text="x" * 501)


@pytest.mark.parametrize("s", ["rag", "sql", "anomaly", "kg"])
def test_citation_source_ok(s):
    """Fails if a spec source is refused."""
    assert citation(source=s).source == s


def test_citation_source_other_refused():
    """Fails if sources are not restricted."""
    with pytest.raises(ValidationError):
        citation(source="web")


def test_citation_ticker_none_ok():
    """Fails if a null ticker is refused."""
    assert citation(ticker=None).ticker is None


def test_citation_ticker_required():
    """Fails if ticker has a default."""
    with pytest.raises(ValidationError):
        Citation(id="1", source="rag", text="t")


def test_citation_frozen():
    """Fails if the model is mutable."""
    c = citation()
    with pytest.raises(ValidationError):
        c.text = "other"


# ---- ChartSpec ----


def test_chart_500_rows_ok_501_refused():
    """Fails if the row limit is off by one."""
    assert len(chart(data=[{"a": 1}] * 500).data) == 500
    with pytest.raises(ValidationError):
        chart(data=[{"a": 1}] * 501)


def test_chart_value_types_ok():
    """Fails if str, int, float or None cell values are refused."""
    row = {"s": "x", "i": 3, "f": 1.5, "n": None}
    assert chart(data=[row]).data[0] == row


@pytest.mark.parametrize("v", [float("nan"), float("inf"), float("-inf")])
def test_chart_non_finite_refused(v):
    """Fails if NaN or infinity is accepted as a chart value."""
    with pytest.raises(ValidationError):
        chart(data=[{"a": v}])


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_chart_non_finite_json_refused(token):
    """Fails if the JSON NaN/Infinity tokens are accepted as chart values."""
    raw = (
        '{"chart_type":"bar","title":"t","x_field":null,"y_fields":[],"data":[{"a":' + token + "}]}"
    )
    with pytest.raises(ValidationError):
        ChartSpec.model_validate_json(raw)


def test_chart_bool_refused_not_coerced():
    """Fails if True is accepted or turned into 1 as a chart value."""
    with pytest.raises(ValidationError):
        chart(data=[{"a": True}])


@pytest.mark.parametrize("v", [[1], {"k": 1}])
def test_chart_container_value_refused(v):
    """Fails if a list or dict cell is accepted."""
    with pytest.raises(ValidationError):
        chart(data=[{"a": v}])


def test_chart_type_scatter_refused():
    """Fails if chart types are not restricted."""
    with pytest.raises(ValidationError):
        chart(chart_type="scatter")


@pytest.mark.parametrize("t", ["bar", "line", "pie", "table"])
def test_chart_type_ok(t):
    """Fails if a spec chart type is refused."""
    assert chart(chart_type=t).chart_type == t


def test_chart_x_field_none_ok():
    """Fails if a null x_field is refused."""
    assert chart(x_field=None).x_field is None


# ---- AskResponse ----


def test_latency_zero_ok():
    """Fails if a latency of zero is refused."""
    assert response(latency_ms=0).latency_ms == 0


@pytest.mark.parametrize("o", ["answered", "not_found", "refused", "clarify"])
def test_outcome_ok(o):
    """Fails if a spec outcome is refused."""
    assert response(outcome=o).outcome == o


def test_outcome_error_refused():
    """Fails if outcomes are not restricted."""
    with pytest.raises(ValidationError):
        response(outcome="error")


def test_route_web_refused():
    """Fails if route items are not restricted."""
    with pytest.raises(ValidationError):
        response(route=["web"])


def test_response_mode_required():
    """Fails if mode has a default."""
    with pytest.raises(ValidationError):
        AskResponse(
            answer="a",
            outcome="answered",
            citations=[],
            chart=None,
            route=[],
            latency_ms=0,
            warnings=[],
        )


def test_response_json_chart_null():
    """Fails if a null chart is omitted from the JSON."""
    assert '"chart":null' in response().model_dump_json().replace(" ", "")


def test_response_round_trip():
    """Fails if the JSON form does not reload to an equal model."""
    r = response(
        citations=[citation()],
        chart=chart(data=[{"a": 1, "b": None}]),
        route=["rag", "kg"],
        warnings=["w"],
    )
    assert AskResponse.model_validate_json(r.model_dump_json()) == r


def test_response_frozen():
    """Fails if the model is mutable."""
    r = response()
    with pytest.raises(ValidationError):
        r.answer = "other"


# ---- Health and errors ----


def components(**kw):
    return HealthComponents(**{"llm": "up", "qdrant": "up", "postgres": "down", **kw})


def test_health_valid():
    """Fails if a valid health body is refused."""
    h = HealthResponse(status="degraded", components=components())
    assert h.components.postgres == "down"


def test_health_status_down_refused():
    """Fails if status accepts values besides ok and degraded."""
    with pytest.raises(ValidationError):
        HealthResponse(status="down", components=components())


def test_health_component_unknown_refused():
    """Fails if a component state besides up and down is accepted."""
    with pytest.raises(ValidationError):
        components(llm="unknown")


def test_error_detail():
    """Fails if the FastAPI-style detail body is refused."""
    assert ErrorDetail(detail="x").detail == "x"
