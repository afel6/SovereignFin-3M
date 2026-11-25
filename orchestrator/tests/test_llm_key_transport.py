"""Tests for orchestrator.llm: SF_LLM_API_KEY travels only over https or straight to loopback."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

import orchestrator.llm as llm

KEY = "sk" + "-" + "k" * 40
LLM_VARIABLES = (
    "SF_LLM_PROVIDER",
    "NVIDIA_API_KEY",
    "SF_LLM_BASE_URL",
    "SF_LLM_API_KEY",
    "SF_LLM_FAST_MODEL",
    "SF_LLM_STRONG_MODEL",
)
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
REPLY = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode("utf-8")


def _no_request(request):
    pytest.fail("no request expected")


@pytest.fixture(autouse=True)
def keyed_compat(monkeypatch):
    """openai_compatible with a key; any request fails the test unless a test allows one."""
    for name in LLM_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm, "_sleep", lambda _seconds: None)
    monkeypatch.setattr(llm, "_transport", httpx.MockTransport(_no_request))
    monkeypatch.setenv("SF_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("SF_LLM_API_KEY", KEY)


def text():
    return llm.complete_text("You are a test.", "Say hi.")


@pytest.mark.parametrize(
    "base",
    [
        "http://llm.example.test/v1",
        "http://10.0.0.5:8000/v1",
        "http://127.0.0.1.evil.example/v1",
        "http://localhost.evil.example/v1",
    ],
)
def test_key_is_never_sent_in_clear_text_to_another_machine(monkeypatch, base):
    """Fails if SF_LLM_API_KEY goes over plain http to a host other than this machine."""
    monkeypatch.setenv("SF_LLM_BASE_URL", base)
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "http"
    assert "SF_LLM_BASE_URL" in str(ei.value)
    assert base not in str(ei.value)


@pytest.mark.parametrize(
    "base",
    [
        "https://llm.example.test/v1",
        "http://127.0.0.1:1234/v1",
        "http://localhost:1234/v1",
        "http://[::1]:1234/v1",
    ],
)
def test_key_goes_over_https_or_to_this_machine(monkeypatch, base):
    """Fails if the clear-text check also refuses https or a server on this machine."""
    monkeypatch.setenv("SF_LLM_BASE_URL", base)
    seen = []

    def answer(request):
        seen.append(request.headers["authorization"])
        return httpx.Response(200, content=REPLY)

    monkeypatch.setattr(llm, "_transport", httpx.MockTransport(answer))
    assert text() == "hi"
    assert seen == ["Bearer " + KEY]


class _Recorder(BaseHTTPRequestHandler):
    """Answers every POST with one chat reply and records its request line and headers."""

    def do_POST(self):
        self.server.seen.append((self.requestline, dict(self.headers)))
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(REPLY)))
        self.end_headers()
        self.wfile.write(REPLY)

    def log_message(self, *args):
        """Keep the test output free of access lines."""


@pytest.fixture
def server():
    """Start recording HTTP servers on 127.0.0.1; all are stopped after the test."""
    started = []

    def start():
        srv = HTTPServer(("127.0.0.1", 0), _Recorder)
        srv.seen = []
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append(srv)
        return srv

    yield start
    for srv in started:
        srv.shutdown()
        srv.server_close()


@pytest.mark.parametrize("variable", ["HTTP_PROXY", "ALL_PROXY"])
def test_keyed_plain_http_never_goes_through_a_proxy(monkeypatch, server, variable):
    """Fails if a proxy taken from the environment receives a keyed plain-http request."""
    target, proxy = server(), server()
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setenv(variable, f"http://127.0.0.1:{proxy.server_port}")
    monkeypatch.setenv("SF_LLM_BASE_URL", f"http://127.0.0.1:{target.server_port}/v1")
    monkeypatch.setattr(llm, "_transport", None)
    assert text() == "hi"
    assert proxy.seen == []
    assert [headers["Authorization"] for _, headers in target.seen] == ["Bearer " + KEY]
