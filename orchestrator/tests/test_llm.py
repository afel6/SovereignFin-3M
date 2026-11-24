"""Tests for orchestrator.llm: modes, endpoints, retries, timeouts, JSON repair, stub, secrecy."""

import itertools
import json
import logging
import threading
import traceback
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

import orchestrator.llm as llm

KEY = "nvapi" + "-" + "k" * 40
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
LOCAL_MSG = "Local mode is not configured on this machine."
NVIDIA_MSG = "Cloud mode is not configured: set NVIDIA_API_KEY."
COMPAT_MSG = "Cloud mode is not configured: set SF_LLM_BASE_URL and SF_LLM_API_KEY."
FAST_DEFAULT = "meta/llama-3.1-8b-instruct"
STRONG_DEFAULT = "meta/llama-3.3-70b-instruct"
ENV_VARS = (
    "SF_LLM_PROVIDER",
    "NVIDIA_API_KEY",
    "SF_LLM_BASE_URL",
    "SF_LLM_API_KEY",
    "SF_LLM_LOCAL_MODEL",
    "SF_LLM_FAST_MODEL",
    "SF_LLM_STRONG_MODEL",
    "OLLAMA_HOST",
)
SYSTEM = "You are a test."
USER = "Say hi."
BASE_MESSAGES = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]


class Answer(BaseModel):
    answer_text: str
    score: int


VALID = '{"answer_text": "hi", "score": 3}'


class Script:
    """MockTransport handler: pops scripted responses or exceptions and records each request."""

    def __init__(self, items):
        self.items = list(items)
        self.requests = []

    def __call__(self, request):
        body = json.loads(request.content) if request.content else None
        self.requests.append(
            SimpleNamespace(url=str(request.url), headers=request.headers, body=body)
        )
        if not self.items:
            pytest.fail("more requests than scripted")
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _fail_handler(request):
    pytest.fail("no request expected")


def ok(content):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


def text(**kw):
    return llm.complete_text(SYSTEM, USER, **kw)


def js(**kw):
    return llm.complete_json(SYSTEM, USER, Answer, **kw)


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Clean LLM env, record backoff sleeps, and forbid any request unless a test scripts one."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    calls = []
    monkeypatch.setattr(llm, "_sleep", calls.append)
    monkeypatch.setattr(llm, "_transport", httpx.MockTransport(_fail_handler))
    return calls


@pytest.fixture
def script(monkeypatch):
    def install(*items):
        s = Script(items)
        monkeypatch.setattr(llm, "_transport", httpx.MockTransport(s))
        return s

    return install


@pytest.fixture
def nvidia(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", KEY)


@pytest.fixture
def compat(monkeypatch):
    monkeypatch.setenv("SF_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("SF_LLM_BASE_URL", "https://llm.example.test/v1")


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setenv("SF_LLM_LOCAL_MODEL", "llama3.2")


def test_llm_error_carries_kind_and_message():
    """A wrong LLMError would lose kind or str(err) != message."""
    err = llm.LLMError("timeout", "slow")
    assert isinstance(err, Exception)
    assert err.kind == "timeout"
    assert str(err) == "slow"


# ---- not configured -------------------------------------------------------------------------


@pytest.mark.parametrize("model", [None, ""])
def test_local_without_model_is_not_configured(script, model, monkeypatch):
    """Local mode needs a non-empty SF_LLM_LOCAL_MODEL; nothing is sent otherwise."""
    if model is not None:
        monkeypatch.setenv("SF_LLM_LOCAL_MODEL", model)
    s = script()
    with pytest.raises(llm.LLMError) as ei:
        text(mode="local")
    assert ei.value.kind == "not_configured"
    assert str(ei.value) == LOCAL_MSG
    assert s.requests == []


@pytest.mark.parametrize("provider", [None, "", "nvidia"])
@pytest.mark.parametrize("key", [None, ""])
def test_nvidia_without_key_is_not_configured(script, monkeypatch, provider, key):
    """Default provider (unset, empty, nvidia) needs a non-empty NVIDIA_API_KEY."""
    if provider is not None:
        monkeypatch.setenv("SF_LLM_PROVIDER", provider)
    if key is not None:
        monkeypatch.setenv("NVIDIA_API_KEY", key)
    s = script()
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "not_configured"
    assert str(ei.value) == NVIDIA_MSG
    assert s.requests == []


@pytest.mark.parametrize("base", [None, ""])
def test_openai_compatible_without_base_url_is_not_configured(script, monkeypatch, base):
    """openai_compatible needs SF_LLM_BASE_URL."""
    monkeypatch.setenv("SF_LLM_PROVIDER", "openai_compatible")
    if base is not None:
        monkeypatch.setenv("SF_LLM_BASE_URL", base)
    s = script()
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "not_configured"
    assert str(ei.value) == COMPAT_MSG
    assert s.requests == []


def test_unknown_provider_is_not_configured_and_names_variable(script, nvidia, monkeypatch):
    """Provider names are case-sensitive: "NVIDIA" is unknown even with a key set."""
    monkeypatch.setenv("SF_LLM_PROVIDER", "NVIDIA")
    s = script()
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "not_configured"
    assert "SF_LLM_PROVIDER" in str(ei.value)
    assert s.requests == []


def test_stub_provider_with_local_mode_needs_local_model(script, monkeypatch):
    """The stub only serves cloud mode: local mode without a model gives the local message."""
    monkeypatch.setenv("SF_LLM_PROVIDER", "stub")
    s = script()
    with pytest.raises(llm.LLMError) as ei:
        text(mode="local")
    assert ei.value.kind == "not_configured"
    assert str(ei.value) == LOCAL_MSG
    assert s.requests == []


def test_complete_json_not_configured_sends_nothing(script):
    """complete_json checks configuration before any request, too."""
    s = script()
    with pytest.raises(llm.LLMError) as ei:
        js()
    assert ei.value.kind == "not_configured"
    assert str(ei.value) == NVIDIA_MSG
    assert s.requests == []


def test_openai_compatible_with_empty_api_key_is_configured(script, compat, monkeypatch):
    """Only the base URL is required: an empty SF_LLM_API_KEY sends no Authorization header."""
    monkeypatch.setenv("SF_LLM_API_KEY", "")
    s = script(ok("hi"))
    assert text() == "hi"
    assert "authorization" not in s.requests[0].headers


# ---- endpoints, headers, models, payloads ---------------------------------------------------


def test_nvidia_request_shape_for_text(script, nvidia):
    """Exact URL, bearer header and exactly the three body keys."""
    s = script(ok("hi"))
    assert text(max_tokens=77) == "hi"
    req = s.requests[0]
    assert req.url == NVIDIA_URL
    assert req.headers["authorization"] == "Bearer " + KEY
    assert set(req.body) == {"model", "messages", "max_tokens"}
    assert req.body["messages"] == BASE_MESSAGES
    assert req.body["max_tokens"] == 77
    assert req.body["model"] == FAST_DEFAULT


def test_default_max_tokens_is_1024(script, nvidia):
    """max_tokens defaults to 1024."""
    s = script(ok("hi"))
    text()
    assert s.requests[0].body["max_tokens"] == 1024


def test_nvidia_request_shape_for_json(script, nvidia):
    """complete_json on nvidia sends the same key set, no response_format."""
    s = script(ok(VALID))
    assert js() == Answer(answer_text="hi", score=3)
    req = s.requests[0]
    assert req.url == NVIDIA_URL
    assert set(req.body) == {"model", "messages", "max_tokens"}
    assert req.body["messages"][1] == {"role": "user", "content": USER}
    assert "answer_text" in req.body["messages"][0]["content"]


@pytest.mark.parametrize(("tier", "default"), [("fast", FAST_DEFAULT), ("strong", STRONG_DEFAULT)])
def test_nvidia_default_models(script, nvidia, tier, default):
    """Each tier has its own default model."""
    s = script(ok("hi"))
    text(tier=tier)
    assert s.requests[0].body["model"] == default


@pytest.mark.parametrize(
    ("tier", "var", "default"),
    [
        ("fast", "SF_LLM_FAST_MODEL", FAST_DEFAULT),
        ("strong", "SF_LLM_STRONG_MODEL", STRONG_DEFAULT),
    ],
)
def test_nvidia_model_override_and_empty_means_default(
    script, nvidia, monkeypatch, tier, var, default
):
    """A non-empty tier variable overrides the model; an empty one means the default."""
    s = script(ok("a"), ok("b"))
    monkeypatch.setenv(var, "vendor/custom-model")
    text(tier=tier)
    monkeypatch.setenv(var, "")
    text(tier=tier)
    assert s.requests[0].body["model"] == "vendor/custom-model"
    assert s.requests[1].body["model"] == default


def test_other_tier_variable_does_not_leak(script, nvidia, monkeypatch):
    """SF_LLM_STRONG_MODEL must not change the fast tier."""
    monkeypatch.setenv("SF_LLM_STRONG_MODEL", "vendor/strong")
    s = script(ok("hi"))
    text(tier="fast")
    assert s.requests[0].body["model"] == FAST_DEFAULT


@pytest.mark.parametrize("base", ["http://llm.example.test/v1", "http://llm.example.test/v1/"])
def test_openai_compatible_url_joins_base(script, monkeypatch, base):
    """A trailing slash on the base URL does not double up."""
    monkeypatch.setenv("SF_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("SF_LLM_BASE_URL", base)
    s = script(ok("hi"))
    text()
    assert s.requests[0].url == "http://llm.example.test/v1/chat/completions"


def test_openai_compatible_sends_own_key_never_nvidia_key(script, compat, monkeypatch):
    """Bearer comes from SF_LLM_API_KEY; NVIDIA_API_KEY stays out of the request."""
    other = "sk" + "-" + "o" * 20
    monkeypatch.setenv("SF_LLM_API_KEY", other)
    monkeypatch.setenv("NVIDIA_API_KEY", KEY)
    s = script(ok("hi"))
    text()
    req = s.requests[0]
    assert req.headers["authorization"] == "Bearer " + other
    assert KEY not in json.dumps(dict(req.headers)) + json.dumps(req.body)


def test_openai_compatible_model_omitted_when_variable_empty(script, compat, monkeypatch):
    """No default model for openai_compatible: empty tier variable means no "model" key."""
    monkeypatch.setenv("SF_LLM_FAST_MODEL", "")
    monkeypatch.setenv("SF_LLM_STRONG_MODEL", "vendor/strong")
    s = script(ok("a"), ok("b"))
    text(tier="fast")
    text(tier="strong")
    assert "model" not in s.requests[0].body
    assert s.requests[1].body["model"] == "vendor/strong"


def test_local_request_shape(script, local, monkeypatch):
    """Local: Ollama default host, fixed model for both tiers, no Authorization header."""
    monkeypatch.setenv("NVIDIA_API_KEY", KEY)
    monkeypatch.setenv("SF_LLM_API_KEY", KEY)
    monkeypatch.setenv("SF_LLM_FAST_MODEL", "other/fast")
    s = script(ok("a"), ok("b"))
    text(mode="local", tier="fast")
    text(mode="local", tier="strong")
    for req in s.requests:
        assert req.url == "http://127.0.0.1:11434/v1/chat/completions"
        assert req.body["model"] == "llama3.2"
        assert "authorization" not in req.headers
        assert "response_format" not in req.body


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("", "http://127.0.0.1:11434/v1/chat/completions"),
        ("http://ollama.example.test:9999", "http://ollama.example.test:9999/v1/chat/completions"),
    ],
)
def test_local_host_comes_from_ollama_host(script, local, monkeypatch, host, expected):
    """OLLAMA_HOST sets the local host; empty means the default."""
    monkeypatch.setenv("OLLAMA_HOST", host)
    s = script(ok("a"))
    text(mode="local")
    assert s.requests[0].url == expected


def test_local_json_sets_response_format(script, local):
    """Only local-mode complete_json asks for JSON mode."""
    s = script(ok(VALID))
    js(mode="local")
    assert s.requests[0].body["response_format"] == {"type": "json_object"}
    assert s.requests[0].body["model"] == "llama3.2"


def test_openai_compatible_json_has_no_response_format(script, compat):
    """openai_compatible uses prompt + parse, not response_format."""
    s = script(ok(VALID))
    js()
    assert "response_format" not in s.requests[0].body


def test_stub_provider_does_not_affect_local_mode(script, local, monkeypatch):
    """With SF_LLM_PROVIDER=stub, mode local still goes over HTTP and never touches the stub."""
    monkeypatch.setenv("SF_LLM_PROVIDER", "stub")
    stub = llm.StubLLM(["unused"])
    monkeypatch.setattr(llm, "stub", stub)
    s = script(ok("from http"))
    assert text(mode="local") == "from http"
    assert len(s.requests) == 1
    assert stub.calls == []


# ---- retries and errors ---------------------------------------------------------------------


def test_429_then_200_retries_after_one_second(script, nvidia, sleeps):
    """One 429 is retried after a 1 s backoff."""
    s = script(httpx.Response(429), ok("hi"))
    assert text() == "hi"
    assert len(s.requests) == 2
    assert sleeps == [1.0]


def test_429_three_times_is_rate_limited(script, nvidia, sleeps):
    """Three attempts, backoff 1 s then 4 s, kind rate_limited."""
    s = script(*[httpx.Response(429) for _ in range(3)])
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "rate_limited"
    assert len(s.requests) == 3
    assert sleeps == [1.0, 4.0]


@pytest.mark.parametrize("status", [500, 503])
def test_5xx_three_times_is_http(script, nvidia, sleeps, status):
    """5xx is retried twice, then kind http."""
    s = script(*[httpx.Response(status) for _ in range(3)])
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "http"
    assert len(s.requests) == 3
    assert sleeps == [1.0, 4.0]


def test_two_500_then_200_succeeds(script, nvidia, sleeps):
    """The third attempt can still succeed."""
    s = script(httpx.Response(500), httpx.Response(500), ok("hi"))
    assert text() == "hi"
    assert len(s.requests) == 3
    assert sleeps == [1.0, 4.0]


def test_400_is_not_retried(script, nvidia, sleeps):
    """A 4xx other than 429 fails at once."""
    s = script(httpx.Response(400))
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "http"
    assert len(s.requests) == 1
    assert sleeps == []


def test_401_names_the_key_variable(script, nvidia, sleeps):
    """401 is http, not retried, and the message names NVIDIA_API_KEY (not its value)."""
    s = script(httpx.Response(401))
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "http"
    assert "NVIDIA_API_KEY" in str(ei.value)
    assert KEY not in str(ei.value)
    assert len(s.requests) == 1
    assert sleeps == []


def test_202_is_http_without_polling(script, nvidia, sleeps):
    """Only 200 is success: 202 pending is an http error after one request."""
    s = script(httpx.Response(202, json={"status": "pending"}))
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "http"
    assert len(s.requests) == 1
    assert sleeps == []


def test_read_timeout_three_times_is_timeout(script, nvidia, sleeps):
    """Timeouts are retried like 429/5xx; the final kind is timeout."""
    s = script(*[httpx.ReadTimeout("slow") for _ in range(3)])
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "timeout"
    assert len(s.requests) == 3
    assert sleeps == [1.0, 4.0]


def test_connect_timeout_then_200_succeeds(script, nvidia, sleeps):
    """A connect timeout counts as a timeout and is retried."""
    s = script(httpx.ConnectTimeout("slow"), ok("hi"))
    assert text() == "hi"
    assert len(s.requests) == 2
    assert sleeps == [1.0]


@pytest.mark.parametrize(
    ("items", "kind"),
    [
        (
            (
                lambda: httpx.Response(429),
                lambda: httpx.Response(500),
                lambda: httpx.ReadTimeout("x"),
            ),
            "timeout",
        ),
        (
            (
                lambda: httpx.ReadTimeout("x"),
                lambda: httpx.Response(500),
                lambda: httpx.Response(429),
            ),
            "rate_limited",
        ),
        (
            (lambda: httpx.Response(429), lambda: httpx.Response(429), lambda: httpx.Response(500)),
            "http",
        ),
    ],
    ids=["ends-timeout", "ends-429", "ends-500"],
)
def test_final_kind_is_kind_of_last_attempt(script, nvidia, items, kind):
    """After three failures the error kind is that of the last attempt."""
    s = script(*[make() for make in items])
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == kind
    assert len(s.requests) == 3


def test_connect_error_is_http_without_retry(script, nvidia, sleeps):
    """Connect/DNS errors are not retried and are kind http."""
    s = script(httpx.ConnectError("refused"))
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "http"
    assert len(s.requests) == 1
    assert sleeps == []


def test_abort_timer_ends_a_hanging_attempt(nvidia, monkeypatch, sleeps):
    """A handler that never answers is cut off at ATTEMPT_TIMEOUT_S, three times."""
    monkeypatch.setattr(llm, "ATTEMPT_TIMEOUT_S", 0.05)
    release = threading.Event()

    def hang(request):
        release.wait(60)  # hang guard only; the test releases it in finally
        return ok("late")

    monkeypatch.setattr(llm, "_transport", httpx.MockTransport(hang))
    try:
        with pytest.raises(llm.LLMError) as ei:
            text()
        assert ei.value.kind == "timeout"
        assert sleeps == [1.0, 4.0]
    finally:
        release.set()


def _clock(step):
    ticks = itertools.count(0.0, step)
    return lambda: next(ticks)


def test_slow_complete_reply_counts_as_timeout(script, nvidia, monkeypatch, sleeps):
    """A 200 that arrives after more than ATTEMPT_TIMEOUT_S on the monotonic clock is a timeout."""
    monkeypatch.setattr(llm, "_monotonic", _clock(25.0))
    s = script(ok("a"), ok("b"), ok("c"))
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "timeout"
    assert len(s.requests) == 3
    assert sleeps == [1.0, 4.0]


def test_fast_clock_reply_is_accepted(script, nvidia, monkeypatch):
    """The elapsed check does not reject a quick reply."""
    monkeypatch.setattr(llm, "_monotonic", _clock(0.001))
    script(ok("hi"))
    assert text() == "hi"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
        httpx.Response(200, json={"choices": [{"message": {"content": 5}}]}),
    ],
    ids=["not-json", "no-choices", "null-content", "int-content"],
)
def test_unreadable_envelope_is_bad_output_without_repair(script, nvidia, sleeps, response):
    """A 200 with an unusable envelope is bad_output after one request, in text and JSON."""
    s = script(response)
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "bad_output"
    assert len(s.requests) == 1
    assert sleeps == []


def test_unreadable_envelope_in_json_does_not_repair(script, nvidia):
    """complete_json repairs only content that is a string; a broken envelope is final."""
    s = script(httpx.Response(200, json={"choices": []}))
    with pytest.raises(llm.LLMError) as ei:
        js()
    assert ei.value.kind == "bad_output"
    assert len(s.requests) == 1


def test_non_ascii_key_is_http_naming_variable(script, monkeypatch):
    """A key that cannot go into a header fails with a friendly message and never shows the key."""
    bad = "nvapi" + "-" + "é" * 10
    monkeypatch.setenv("NVIDIA_API_KEY", bad)
    script()
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "http"
    assert "NVIDIA_API_KEY" in str(ei.value)
    assert bad not in str(ei.value)


# ---- secrecy --------------------------------------------------------------------------------


def _rep(n, make):
    return [make() for _ in range(n)]


SECRET_SCENARIOS = {
    "401": (lambda: [httpx.Response(401, text="bad " + KEY)], text),
    "429x3": (lambda: _rep(3, lambda: httpx.Response(429, text=KEY)), text),
    "500x3": (lambda: _rep(3, lambda: httpx.Response(500, text=KEY)), text),
    "timeout-x3": (lambda: _rep(3, lambda: httpx.ReadTimeout("slow " + KEY)), text),
    "connect-error": (lambda: [httpx.ConnectError("refused " + KEY)], text),
    # A 200 body holds model output only; the key is never part of the prompt.
    "bad-envelope": (lambda: [httpx.Response(200, text="not json")], text),
    "bad-json-x2": (lambda: _rep(2, lambda: ok("nope " + KEY)), js),
}


@pytest.mark.parametrize("name", list(SECRET_SCENARIOS))
def test_key_never_appears_in_error_or_logs(script, nvidia, caplog, name):
    """The key is absent from str, repr, the formatted traceback and every log record."""
    caplog.set_level(logging.DEBUG)
    make_items, call = SECRET_SCENARIOS[name]
    script(*make_items())
    with pytest.raises(llm.LLMError) as ei:
        call()
    err = ei.value
    assert KEY not in str(err)
    assert KEY not in repr(err)
    assert KEY not in "".join(traceback.format_exception(err))
    assert KEY not in str(ei.getrepr(funcargs=True))
    assert all(KEY not in r.getMessage() for r in caplog.records)
    assert KEY not in caplog.text


# ---- complete_json --------------------------------------------------------------------------


def test_json_valid_first_reply_makes_one_request(script, nvidia):
    """A valid reply is parsed into the schema with no repair call."""
    s = script(ok(VALID))
    result = js()
    assert isinstance(result, Answer)
    assert result == Answer(answer_text="hi", score=3)
    assert len(s.requests) == 1


@pytest.mark.parametrize(
    "bad",
    ["  {not json}\n", '{"answer_text": "only"}', "[1, 2]"],
    ids=["invalid-json", "missing-field", "not-an-object"],
)
def test_json_bad_reply_gets_one_repair_call(script, nvidia, bad):
    """The repair call replays system+user, the bad reply verbatim, then a new user message."""
    s = script(ok(bad), ok(VALID))
    assert js() == Answer(answer_text="hi", score=3)
    assert len(s.requests) == 2
    first = s.requests[0].body["messages"]
    repair = s.requests[1].body["messages"]
    assert len(repair) == 4
    assert repair[:2] == first
    assert repair[2] == {"role": "assistant", "content": bad}
    assert repair[3]["role"] == "user"
    assert isinstance(repair[3]["content"], str) and repair[3]["content"]


def test_json_wrong_field_type_gets_repair(script, nvidia):
    """A reply that parses but fails pydantic validation is repaired once."""
    s = script(ok('{"answer_text": 5, "score": "x"}'), ok(VALID))
    assert js().score == 3
    assert len(s.requests) == 2


def test_json_two_bad_replies_is_bad_output_after_two_requests(script, nvidia, sleeps):
    """Exactly one repair: a second failure ends in bad_output."""
    s = script(ok("nope"), ok('{"answer_text": "x"}'), ok(VALID))
    with pytest.raises(llm.LLMError) as ei:
        js()
    assert ei.value.kind == "bad_output"
    assert len(s.requests) == 2
    assert sleeps == []


# ---- StubLLM --------------------------------------------------------------------------------


@pytest.fixture
def stub_env(monkeypatch):
    monkeypatch.setenv("SF_LLM_PROVIDER", "stub")

    def install(*replies):
        stub = llm.StubLLM(list(replies))
        monkeypatch.setattr(llm, "stub", stub)
        return stub

    return install


def test_stub_text_returns_scripted_reply_and_records_call(stub_env):
    """The stub returns the scripted str and records tier and sent messages; no network."""
    stub = stub_env("Answer text")
    assert text(tier="strong") == "Answer text"
    assert len(stub.calls) == 1
    assert stub.calls[0].tier == "strong"
    assert stub.calls[0].messages == BASE_MESSAGES


def test_stub_default_tier_is_fast(stub_env):
    """The recorded tier defaults to fast."""
    stub = stub_env("x")
    text()
    assert stub.calls[0].tier == "fast"


def test_stub_json_parses_scripted_reply(stub_env):
    """complete_json parses the scripted reply; the schema is in the system message."""
    stub = stub_env(VALID)
    assert js() == Answer(answer_text="hi", score=3)
    assert len(stub.calls) == 1
    assert "answer_text" in stub.calls[0].messages[0]["content"]


def test_stub_json_repair_takes_next_item(stub_env):
    """A bad first reply makes the next scripted item the repair reply."""
    stub = stub_env("nope", VALID)
    assert js() == Answer(answer_text="hi", score=3)
    assert len(stub.calls) == 2
    repair = stub.calls[1].messages
    assert repair[:2] == stub.calls[0].messages
    assert repair[2] == {"role": "assistant", "content": "nope"}
    assert repair[3]["role"] == "user"


def test_stub_json_two_bad_replies_is_bad_output(stub_env):
    """Two bad scripted replies end in bad_output after two stub calls."""
    stub = stub_env("nope", "still nope", VALID)
    with pytest.raises(llm.LLMError) as ei:
        js()
    assert ei.value.kind == "bad_output"
    assert len(stub.calls) == 2


def test_stub_raises_scripted_error(stub_env):
    """A scripted LLMError is raised as is."""
    stub_env(llm.LLMError("rate_limited", "slow down"))
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "rate_limited"
    assert str(ei.value) == "slow down"


def test_stub_runs_scripted_items_in_order(stub_env):
    """Items are consumed in order across calls."""
    stub_env("one", llm.LLMError("timeout", "t"), "three")
    assert text() == "one"
    with pytest.raises(llm.LLMError):
        text()
    assert text() == "three"


def test_stub_empty_script_is_bad_output(stub_env):
    """An exhausted stub raises bad_output instead of returning something."""
    stub_env()
    with pytest.raises(llm.LLMError) as ei:
        text()
    assert ei.value.kind == "bad_output"
