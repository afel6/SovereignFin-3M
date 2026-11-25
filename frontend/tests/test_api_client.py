import json

import httpx
import pytest
from fastapi.testclient import TestClient

from frontend import connection
from orchestrator.mock_server import app


@pytest.fixture
def api():
    from frontend import api_client

    return api_client


def test_configuration(monkeypatch):
    monkeypatch.delenv("SF_API_BASE", raising=False)
    with connection.create_client() as client:
        assert str(client.base_url) == "http://127.0.0.1:8000"
        assert client.timeout == httpx.Timeout(100)
        assert connection.TIMEOUT_SECONDS == 100
    monkeypatch.setenv("SF_API_BASE", "http://localhost:9000")
    with connection.create_client() as client:
        assert str(client.base_url) == "http://localhost:9000"


def test_mock_contract(api, monkeypatch):
    monkeypatch.setenv("SF_LLM_PROVIDER", "stub")
    with TestClient(app) as client:
        revenue = api.ask(client, "What was revenue in 2023?", "cloud")
        late = api.ask(client, "Which invoices are late?", "cloud")
        anomaly = api.ask(client, "Any anomalies?", "cloud")
        local = api.ask(client, "What was revenue?", "local")
        health = api.health(client)

    assert revenue.outcome == "answered" and revenue.citations and "rag" in revenue.route
    assert late.chart.chart_type == "bar" and "sql" in late.route
    assert anomaly.chart.chart_type == "table" and "anomaly" in anomaly.route
    assert local.status_code == 503
    assert local.message == "Local mode is not configured on this machine."
    assert health.status == "ok"
    assert health.components.model_dump() == dict(llm="up", qdrant="up", postgres="up")


@pytest.mark.parametrize("thread_id", ["t-1", None])
def test_request_body(api, thread_id):
    def respond(request):
        assert request.method == "POST" and request.url.path == "/ask"
        assert request.extensions["timeout"]["read"] == 100
        body = json.loads(request.content)
        assert body.pop("thread_id", None) == thread_id
        assert body == {"question": "q", "mode": "local"}
        return httpx.Response(503, json={"detail": "LLM unavailable"})

    with connection.create_client(transport=httpx.MockTransport(respond)) as client:
        error = api.ask(client, "  q  ", "local", thread_id)
    assert error.message == "LLM unavailable"


@pytest.mark.parametrize(
    ("failure", "message", "status"),
    [
        (httpx.ConnectError, "Cannot reach the API.", None),
        (httpx.ReadTimeout, "The API request timed out.", None),
        (422, "The API rejected the request.", 422),
        (503, "The API is unavailable.", 503),
        (500, "The API returned an unexpected HTTP response.", 500),
        (200, "The API returned an invalid response.", 200),
        ("json", "The API returned an invalid response.", 200),
    ],
)
def test_error_mapping(api, failure, message, status):
    def respond(request):
        if isinstance(failure, type):
            raise failure("Untrusted transport details", request=request)
        if failure == "json":
            return httpx.Response(200, text="<html>")
        return httpx.Response(failure, json={"detail": ["untrusted validation details"]})

    with connection.create_client(transport=httpx.MockTransport(respond)) as client:
        for result in (api.ask(client, "q", "cloud"), api.health(client)):
            assert result.message == message and result.status_code == status


@pytest.mark.parametrize("question", ["   ", "a" * 2001])
def test_invalid_question_never_sent(api, question):
    def respond(request):
        pytest.fail(f"Unexpected request: {request.url.path}")

    with connection.create_client(transport=httpx.MockTransport(respond)) as client:
        result = api.ask(client, question, "cloud")
    assert result.message == "Enter a question of 1 to 2,000 characters."


@pytest.mark.parametrize(
    "base",
    [
        "ftp://localhost",
        "http://",
        "http://name:password@localhost",
        "https://localhost?token=value",
        "https://localhost#fragment",
    ],
)
def test_invalid_api_base_rejected(monkeypatch, base):
    monkeypatch.setenv("SF_API_BASE", base)
    with pytest.raises(connection.ApiConfigurationError, match=r"^The API address is invalid\.$"):
        connection.create_client()


@pytest.mark.parametrize("field", ["chart", "warnings", "citations"])
def test_missing_nullable_or_list_fields_are_invalid(api, field):
    payload = dict(
        answer="Sample",
        outcome="not_found",
        citations=[],
        chart=None,
        route=[],
        latency_ms=1,
        mode="cloud",
        warnings=[],
    )
    payload.pop(field)
    with connection.create_client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        assert (
            api.ask(client, "Question?", "cloud").message == "The API returned an invalid response."
        )
