"""Tests for orchestrator.llm when the HTTP client cannot be set up or the URL is invalid.

Each case fails before any connection is made: the real transport is used, and every proxy
variable from the machine is cleared first so only the value a test sets is read.
"""

import traceback

import pytest

import orchestrator.llm as llm

KEY = "nvapi" + "-" + "k" * 40
PROXY_PASSWORD = "proxy" + "pass" + "word42"
LLM_VARIABLES = (
    "SF_LLM_PROVIDER",
    "NVIDIA_API_KEY",
    "SF_LLM_BASE_URL",
    "SF_LLM_API_KEY",
    "SF_LLM_LOCAL_MODEL",
    "SF_LLM_FAST_MODEL",
    "SF_LLM_STRONG_MODEL",
    "OLLAMA_HOST",
)
CLIENT_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "SSL_CERT_FILE")


def _no_retry(seconds):
    pytest.fail("an http error must not be retried")


@pytest.fixture(autouse=True)
def client_env(monkeypatch):
    """Real transport, no LLM, proxy or TLS settings from the machine (either case), no retries."""
    for name in LLM_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for name in CLIENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setattr(llm, "_transport", None)
    monkeypatch.setattr(llm, "_sleep", _no_retry)


def text():
    return llm.complete_text("You are a test.", "Say hi.")


@pytest.mark.parametrize("key", ["", KEY])
@pytest.mark.parametrize("base", ["ftp://llm.example.test/v1", "not a url", "http://[::1"])
def test_invalid_base_url_is_http_naming_variable(monkeypatch, base, key):
    """A bad SF_LLM_BASE_URL fails before any network use, naming the variable."""
    monkeypatch.setenv("SF_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("SF_LLM_BASE_URL", base)
    monkeypatch.setenv("SF_LLM_API_KEY", key)
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "http"
    assert "SF_LLM_BASE_URL" in str(ei.value)


def test_missing_ca_file_is_http_without_key(monkeypatch, tmp_path):
    """Fails if an OS error while building the client (missing SSL_CERT_FILE) escapes as a crash."""
    monkeypatch.setenv("NVIDIA_API_KEY", KEY)
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing-ca.pem"))
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "http"
    assert "SSL_CERT_FILE" in str(ei.value)
    assert KEY not in str(ei.value)


@pytest.mark.parametrize(
    "proxy",
    [
        # socks5 needs the optional socksio package (ImportError); ftp is no proxy scheme
        # (ValueError).
        "socks5://user:" + PROXY_PASSWORD + "@127.0.0.1:9",
        "ftp://user:" + PROXY_PASSWORD + "@127.0.0.1:9",
    ],
    ids=["socks5", "ftp"],
)
def test_bad_proxy_setting_is_http_without_secrets(monkeypatch, proxy):
    """Fails if an error while building the client escapes as ImportError or ValueError (a 500 in
    /ask instead of 503), or if the message shows the key or the proxy credentials."""
    monkeypatch.setenv("NVIDIA_API_KEY", KEY)
    monkeypatch.setenv("HTTPS_PROXY", proxy)
    with pytest.raises(llm.LLMError) as ei:
        text()
    message = str(ei.value)
    assert ei.value.kind == "http"
    assert "HTTPS_PROXY" in message
    formatted = "".join(traceback.format_exception(ei.value))
    for shown in (message, formatted):
        assert KEY not in shown
        assert PROXY_PASSWORD not in shown
    # The report's own funcargs include this test's `proxy` argument, so only the key is checked.
    assert KEY not in str(ei.getrepr(funcargs=True))
