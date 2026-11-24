"""Chat client for the free hosted LLM (NVIDIA build API by default), OpenAI-compatible over httpx.

`complete_text` and `complete_json` follow docs/contracts.md section 3b. The provider and models
come from environment variables read at each call: SF_LLM_PROVIDER (nvidia, openai_compatible or
stub) for mode "cloud"; SF_LLM_LOCAL_MODEL and OLLAMA_HOST for mode "local". A key is sent only to
its own host, over https or to this machine's loopback, and no error message ever holds a key or a
configured URL. SF_LLM_PROVIDER=stub
answers from the scripted `stub` (a `StubLLM`) without any network call.

This module imports only the standard library, pydantic and httpx, so any folder may import it.
"""

import ipaddress
import json
import os
import time
from collections import deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

import httpx
from pydantic import BaseModel, ValidationError

Tier = Literal["fast", "strong"]
Mode = Literal["cloud", "local"]
ErrorKind = Literal["not_configured", "rate_limited", "timeout", "bad_output", "http"]
Message = dict[str, str]

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
OLLAMA_DEFAULT_HOST = "http://127.0.0.1:11434"
OLLAMA_API_PATH = "v1"
CHAT_PATH = "chat/completions"
DEFAULT_MODELS: dict[str, str] = {
    "fast": "meta/llama-3.1-8b-instruct",
    "strong": "meta/llama-3.3-70b-instruct",
}
TIER_MODEL_VARIABLES: dict[str, str] = {
    "fast": "SF_LLM_FAST_MODEL",
    "strong": "SF_LLM_STRONG_MODEL",
}
DEFAULT_MAX_TOKENS = 1024

ATTEMPT_TIMEOUT_S = 20.0
RETRY_BACKOFF_S = (1.0, 4.0)
MAX_ATTEMPTS = len(RETRY_BACKOFF_S) + 1
HTTP_OK = 200
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERRORS = range(500, 600)
HTTP_KEY_REFUSED = (401, 403)

LOCAL_NOT_CONFIGURED = "Local mode is not configured on this machine."
NVIDIA_NOT_CONFIGURED = "Cloud mode is not configured: set NVIDIA_API_KEY."
OPENAI_COMPATIBLE_NOT_CONFIGURED = (
    "Cloud mode is not configured: set SF_LLM_BASE_URL and SF_LLM_API_KEY."
)
UNKNOWN_PROVIDER = (
    "Cloud mode is not configured: set SF_LLM_PROVIDER to nvidia, openai_compatible or stub."
)
CLEAR_TEXT_KEY = (
    "LLM request refused: SF_LLM_API_KEY is sent only over https or to this machine; "
    "check SF_LLM_BASE_URL."
)
RATE_LIMITED = "The LLM provider is rate limiting requests (HTTP 429)."
TIMED_OUT = f"The LLM provider did not answer within {ATTEMPT_TIMEOUT_S:g} s."
UNREADABLE_REPLY = "The LLM reply could not be read."
SCHEMA_MISMATCH = "The LLM reply did not match the expected JSON schema."
STUB_EXHAUSTED = "The stub LLM has no scripted reply left."
CLIENT_SETUP_FAILED = (
    "LLM request failed: the HTTP client could not be set up ({error}); check HTTPS_PROXY, "
    "HTTP_PROXY, ALL_PROXY, SSL_CERT_FILE and SSL_CERT_DIR."
)

JSON_INSTRUCTION = (
    "Reply with one JSON object and nothing else: no prose and no code fences. "
    "It must match this JSON schema:\n"
)
REPAIR_INSTRUCTION = (
    "Your reply did not match the JSON schema. Problems: {problems}\n"
    "Reply again with only the corrected JSON object."
)

# Test hooks: tests replace these with an httpx.MockTransport, a recording sleep and a fake clock.
_transport: httpx.BaseTransport | None = None
_sleep = time.sleep
_monotonic = time.monotonic


class LLMError(Exception):
    """An LLM call failed; `kind` is one of the ErrorKind values, the message is safe to show."""

    def __init__(self, kind: ErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind: ErrorKind = kind


class StubCall(NamedTuple):
    """One call received by a StubLLM: the tier and the chat messages as they would be sent."""

    tier: str
    messages: list[Message]


class StubLLM:
    """Scripted LLM for tests, used when SF_LLM_PROVIDER=stub and mode is "cloud".

    Each call takes the next scripted item: a str is the reply, an LLMError is raised. Install one
    with `monkeypatch.setattr(orchestrator.llm, "stub", StubLLM([...]))`.
    """

    def __init__(self, replies: Iterable[str | LLMError] = ()) -> None:
        self._replies = deque(replies)
        self.calls: list[StubCall] = []

    def reply(self, tier: str, messages: list[Message]) -> str:
        self.calls.append(StubCall(tier, messages))
        if not self._replies:
            raise LLMError("bad_output", STUB_EXHAUSTED)
        item = self._replies.popleft()
        if isinstance(item, LLMError):
            raise item
        return item


stub = StubLLM()


@dataclass(frozen=True)
class _Endpoint:
    url: str
    url_variable: str  # variable that configured the URL; "" for the fixed NVIDIA URL
    key: str = field(repr=False)  # "" sends no Authorization header
    key_variable: str
    model: str  # "" lets the endpoint choose its default model
    json_mode: bool  # the provider documents response_format {"type": "json_object"}


def complete_text(
    system: str,
    user: str,
    *,
    tier: Tier = "fast",
    mode: Mode = "cloud",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Return the model's reply to one system + user message pair."""
    backend = _backend(mode, tier)
    return _complete(backend, tier, _messages(system, user), max_tokens, json_mode=False)


def complete_json[T: BaseModel](
    system: str,
    user: str,
    schema: type[T],
    *,
    tier: Tier = "fast",
    mode: Mode = "cloud",
) -> T:
    """Return the model's reply validated as `schema`, after at most one repair request."""
    backend = _backend(mode, tier)
    instruction = JSON_INSTRUCTION + json.dumps(schema.model_json_schema())
    messages = _messages(f"{system}\n\n{instruction}", user)
    reply = _complete(backend, tier, messages, DEFAULT_MAX_TOKENS, json_mode=True)
    try:
        return schema.model_validate_json(reply)
    except ValidationError as error:
        problems = _problems(error)
    repair = [
        *messages,
        {"role": "assistant", "content": reply},
        {"role": "user", "content": REPAIR_INSTRUCTION.format(problems=problems)},
    ]
    reply = _complete(backend, tier, repair, DEFAULT_MAX_TOKENS, json_mode=True)
    try:
        return schema.model_validate_json(reply)
    except ValidationError:
        raise LLMError("bad_output", SCHEMA_MISMATCH) from None


def _env(name: str) -> str:
    return os.environ.get(name, "")


def _join(base: str, path: str) -> str:
    return base + path if base.endswith("/") else f"{base}/{path}"


def _messages(system: str, user: str) -> list[Message]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _backend(mode: Mode, tier: Tier) -> _Endpoint | StubLLM:
    if mode == "local":
        return _local_endpoint()
    provider = _env("SF_LLM_PROVIDER") or "nvidia"
    if provider == "stub":
        return stub
    model = _env(TIER_MODEL_VARIABLES[tier])
    if provider == "nvidia":
        key = _env("NVIDIA_API_KEY")
        if not key:
            raise LLMError("not_configured", NVIDIA_NOT_CONFIGURED)
        url = _join(NVIDIA_BASE_URL, CHAT_PATH)
        return _Endpoint(url, "", key, "NVIDIA_API_KEY", model or DEFAULT_MODELS[tier], False)
    if provider == "openai_compatible":
        base = _env("SF_LLM_BASE_URL")
        if not base:
            raise LLMError("not_configured", OPENAI_COMPATIBLE_NOT_CONFIGURED)
        key = _env("SF_LLM_API_KEY")
        url = _join(base, CHAT_PATH)
        if key:
            _check_key_transport(url)
        return _Endpoint(url, "SF_LLM_BASE_URL", key, "SF_LLM_API_KEY", model, False)
    raise LLMError("not_configured", UNKNOWN_PROVIDER)


def _check_key_transport(url: str) -> None:
    """Refuse a URL that would carry the key in clear text off this machine."""
    try:
        parsed = httpx.URL(url)  # the parser the client connects with
    except httpx.InvalidURL:
        raise _invalid_url("SF_LLM_BASE_URL") from None
    if parsed.scheme == "https" or (parsed.scheme == "http" and _is_loopback(parsed.host)):
        return
    raise LLMError("http", CLEAR_TEXT_KEY)


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _invalid_url(variable: str) -> LLMError:
    return LLMError("http", f"LLM request failed: {variable} is not a valid http or https URL.")


def _local_endpoint() -> _Endpoint:
    model = _env("SF_LLM_LOCAL_MODEL")
    if not model:
        raise LLMError("not_configured", LOCAL_NOT_CONFIGURED)
    host = _env("OLLAMA_HOST") or OLLAMA_DEFAULT_HOST
    url = _join(_join(host, OLLAMA_API_PATH), CHAT_PATH)
    return _Endpoint(url, "OLLAMA_HOST", "", "", model, True)


def _complete(
    backend: _Endpoint | StubLLM,
    tier: Tier,
    messages: list[Message],
    max_tokens: int,
    *,
    json_mode: bool,
) -> str:
    if isinstance(backend, StubLLM):
        return backend.reply(tier, messages)
    payload: dict[str, object] = {"messages": messages, "max_tokens": max_tokens}
    if backend.model:
        payload["model"] = backend.model
    if json_mode and backend.json_mode:
        payload["response_format"] = {"type": "json_object"}
    # ASCII escapes keep every character sendable, lone surrogates included.
    return _read_content(_send(backend, json.dumps(payload).encode("ascii")))


def _send(endpoint: _Endpoint, body: bytes) -> bytes:
    """POST with up to MAX_ATTEMPTS attempts; retry on 429, 5xx and timeouts only."""
    failure = LLMError("http", UNREADABLE_REPLY)  # replaced by every failed attempt
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            _sleep(RETRY_BACKOFF_S[attempt - 1])
        try:
            status, reply = _attempt(endpoint, body)
        except LLMError as error:
            if error.kind != "timeout":
                raise
            failure = error
            continue
        if status == HTTP_OK:
            return reply
        failure = _status_error(status, endpoint)
        if status != HTTP_TOO_MANY_REQUESTS and status not in HTTP_SERVER_ERRORS:
            raise failure
    raise failure


def _attempt(endpoint: _Endpoint, body: bytes) -> tuple[int, bytes]:
    """One request, from connect to the last body byte, limited to ATTEMPT_TIMEOUT_S.

    The exchange runs on a worker thread so the wait can end at the limit even while the body is
    still arriving; closing the client then aborts the request.
    """
    started = _monotonic()
    client = _build_client(endpoint)
    worker = ThreadPoolExecutor(max_workers=1)
    try:
        future = worker.submit(_exchange, client, endpoint, body)
        status, reply = future.result(timeout=ATTEMPT_TIMEOUT_S)
    except (FutureTimeout, httpx.TimeoutException):
        raise LLMError("timeout", TIMED_OUT) from None
    except (httpx.InvalidURL, httpx.UnsupportedProtocol):
        raise _invalid_url(endpoint.url_variable) from None
    except UnicodeEncodeError:
        message = f"LLM request failed: {endpoint.key_variable} must hold ASCII characters only."
        raise LLMError("http", message) from None
    except (httpx.HTTPError, OSError) as error:
        message = f"Could not reach the LLM provider ({type(error).__name__})."
        raise LLMError("http", message) from None
    finally:
        client.close()
        worker.shutdown(wait=False)
    if _monotonic() - started > ATTEMPT_TIMEOUT_S:
        raise LLMError("timeout", TIMED_OUT)
    return status, reply


def _build_client(endpoint: _Endpoint) -> httpx.Client:
    """Build the HTTP client; proxy and TLS settings read from the environment can make this fail.

    Any failure becomes one line naming those variables, never their values (a proxy URL can hold
    credentials).
    """
    # A keyed plain-http request goes to loopback only, so it skips any configured proxy.
    mounts = {"http://": None} if endpoint.key else None
    try:
        return httpx.Client(transport=_transport, timeout=ATTEMPT_TIMEOUT_S, mounts=mounts)
    except Exception as error:  # ImportError (SOCKS), ValueError (proxy scheme), OSError (TLS)
        message = CLIENT_SETUP_FAILED.format(error=type(error).__name__)
        raise LLMError("http", message) from None


def _exchange(client: httpx.Client, endpoint: _Endpoint, body: bytes) -> tuple[int, bytes]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if endpoint.key:
        headers["Authorization"] = f"Bearer {endpoint.key}"
    with client.stream("POST", endpoint.url, content=body, headers=headers) as response:
        # Only a 200 body is used; an error body is never read, so it is never held or shown.
        reply = response.read() if response.status_code == HTTP_OK else b""
        return response.status_code, reply


def _status_error(status: int, endpoint: _Endpoint) -> LLMError:
    if status == HTTP_TOO_MANY_REQUESTS:
        return LLMError("rate_limited", RATE_LIMITED)
    if status in HTTP_KEY_REFUSED and endpoint.key_variable:
        return LLMError(
            "http", f"LLM request was refused with HTTP {status}: check {endpoint.key_variable}."
        )
    return LLMError("http", f"LLM request failed with HTTP {status}.")


def _read_content(reply: bytes) -> str:
    try:
        content = json.loads(reply)["choices"][0]["message"]["content"]
    except (ValueError, LookupError, TypeError, RecursionError):
        raise LLMError("bad_output", UNREADABLE_REPLY) from None
    if not isinstance(content, str):
        raise LLMError("bad_output", UNREADABLE_REPLY)
    return content


def _problems(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in item['loc']) or 'reply'}: {item['msg']}"
        for item in error.errors(include_url=False)
    )
