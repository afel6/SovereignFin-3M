"""Tests for the mock HTTP server (orchestrator/mock_server.py)."""

import pytest
from fastapi.testclient import TestClient

from orchestrator.api_contract import AskResponse, HealthResponse
from orchestrator.mock_server import app

client = TestClient(app)

LOCAL_DETAIL = {"detail": "Local mode is not configured on this machine."}
ALLOWED_ORIGINS = ["http://localhost:5173", "http://localhost:8501"]
DENIED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:5173"]
ACAO = "access-control-allow-origin"


def ask(question, **extra):
    return client.post("/ask", json={"question": question, **extra})


def ask_ok(question, **extra) -> AskResponse:
    response = ask(question, **extra)
    assert response.status_code == 200
    return AskResponse.model_validate(response.json())


def test_revenue_question_is_answered_by_rag():
    """Fails if the revenue keyword does not give a cited rag answer in cloud mode."""
    body = ask_ok("What was ACME revenue last year?")
    assert body.route == ["rag"]
    assert body.outcome == "answered"
    assert body.citations
    assert all(c.source == "rag" for c in body.citations)
    assert body.mode == "cloud"
    assert isinstance(body.latency_ms, int) and body.latency_ms >= 0


def test_late_question_routes_to_sql_with_bar_chart():
    """Fails if 'late' does not give route sql with a bar chart."""
    body = ask_ok("Which vendors were paid late?")
    assert body.route == ["sql"]
    assert body.chart is not None and body.chart.chart_type == "bar"


@pytest.mark.parametrize("question", ["Show anomalies for ACME", "anomalous payments"])
def test_anomaly_question_routes_to_anomaly_with_table(question):
    """Fails if the 'anomal' stem does not give route anomaly with a table chart."""
    body = ask_ok(question)
    assert body.route == ["anomaly"]
    assert body.chart is not None and body.chart.chart_type == "table"


@pytest.mark.parametrize(
    ("question", "route"),
    [
        ("REVENUE growth", ["rag"]),
        ("late revenue", ["rag"]),
        ("late anomalies", ["sql"]),
    ],
)
def test_keyword_matching_is_casefolded_and_ordered(question, route):
    """Fails if matching is case-sensitive or the order is not revenue, late, anomal."""
    assert ask_ok(question).route == route


def test_unknown_question_is_not_found():
    """Fails if a question without keyword gets an answer, citations or a chart."""
    body = ask_ok("Tell me a joke")
    assert body.outcome == "not_found"
    assert body.route == []
    assert body.citations == []
    assert body.chart is None


@pytest.mark.parametrize("question", ["What was ACME revenue?", "Tell me a joke"])
def test_local_mode_is_503_before_keywords(question):
    """Fails if local mode is answered, or if the keyword check runs before the mode check."""
    response = ask(question, mode="local")
    assert response.status_code == 503
    assert response.json() == LOCAL_DETAIL


def test_explicit_cloud_mode_is_echoed():
    """Fails if the response mode is not taken from the request."""
    assert ask_ok("revenue", mode="cloud").mode == "cloud"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": 123},
        {"question": "revenue", "mode": "remote"},
        {"question": "revenue", "thread_id": "a/b"},
        {"question": "revenue", "thread_id": "a" * 65},
    ],
)
def test_invalid_body_is_422(payload):
    """Fails if an invalid body is accepted."""
    assert client.post("/ask", json=payload).status_code == 422


def test_non_json_body_is_422():
    """Fails if a body that is not JSON gives anything but 422."""
    response = client.post("/ask", content="not json", headers={"content-type": "application/json"})
    assert response.status_code == 422


@pytest.mark.parametrize(
    "payload",
    [{"question": "x" * 2000}, {"question": "revenue", "thread_id": "a" * 64}],
)
def test_boundary_values_are_accepted(payload):
    """Fails if a value at the documented limit is refused."""
    assert client.post("/ask", json=payload).status_code == 200


def test_health_reports_all_components_up():
    """Fails if /health is not ok with llm, qdrant and postgres up."""
    response = client.get("/health")
    assert response.status_code == 200
    body = HealthResponse.model_validate(response.json())
    assert body.status == "ok"
    assert body.components.model_dump() == {"llm": "up", "qdrant": "up", "postgres": "up"}


def test_ask_response_is_json():
    """Fails if the content type of a 200 reply is not application/json."""
    response = ask("revenue")
    assert response.headers["content-type"].split(";")[0] == "application/json"


@pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
def test_preflight_allows_listed_origin(origin):
    """Fails if a listed origin is not echoed on the preflight."""
    response = client.options(
        "/ask",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers[ACAO] == origin


@pytest.mark.parametrize("origin", DENIED_ORIGINS)
def test_preflight_denies_other_origin(origin):
    """Fails if CORS is open to an origin outside the two listed ones."""
    response = client.options(
        "/ask",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert ACAO not in response.headers


@pytest.mark.parametrize("origin", ALLOWED_ORIGINS)
def test_actual_requests_carry_cors_header_for_listed_origin(origin):
    """Fails if POST /ask or GET /health lack the echoed origin."""
    headers = {"Origin": origin}
    assert (
        client.post("/ask", json={"question": "revenue"}, headers=headers).headers[ACAO] == origin
    )
    assert client.get("/health", headers=headers).headers[ACAO] == origin


@pytest.mark.parametrize("origin", DENIED_ORIGINS)
def test_actual_requests_have_no_cors_header_for_other_origin(origin):
    """Fails if POST /ask or GET /health carry the header for a disallowed origin."""
    headers = {"Origin": origin}
    assert ACAO not in client.post("/ask", json={"question": "revenue"}, headers=headers).headers
    assert ACAO not in client.get("/health", headers=headers).headers
