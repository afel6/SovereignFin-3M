"""The shipped environment template lists the agreed keys and no secrets."""

from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parent.parent
ENV_EXAMPLE = ROOT / ".env.example"
LOCAL_HOST = "127.0.0.1"
EXPECTED_KEYS = {
    "DATABASE_URL",
    "DATABASE_URL_OWNER",
    "QDRANT_URL",
    "DATABASE_URL_ADMIN",
    "POSTGRES_PASSWORD",
    "SF_OWNER_PASSWORD",
    "SF_READER_PASSWORD",
    "SF_LLM_PROVIDER",
    "NVIDIA_API_KEY",
    "SF_LLM_BASE_URL",
    "SF_LLM_API_KEY",
    "SF_LLM_LOCAL_MODEL",
    "SF_LLM_FAST_MODEL",
    "SF_LLM_STRONG_MODEL",
    "OLLAMA_HOST",
    "SF_API_BASE",
}
SECRET_KEYS = [
    "NVIDIA_API_KEY",
    "SF_LLM_API_KEY",
    "POSTGRES_PASSWORD",
    "SF_OWNER_PASSWORD",
    "SF_READER_PASSWORD",
]
LOCAL_URL_KEYS = [
    "QDRANT_URL",
    "OLLAMA_HOST",
    "SF_API_BASE",
    "DATABASE_URL",
    "DATABASE_URL_OWNER",
    "DATABASE_URL_ADMIN",
]


def parse_env(text: str) -> list[tuple[str, str]]:
    """Return (key, value) pairs; comments and blank lines are skipped."""
    pairs = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"not KEY=value: {line!r}")
        pairs.append((key.strip(), value.strip()))
    return pairs


@pytest.fixture(scope="module")
def pairs() -> list[tuple[str, str]]:
    return parse_env(ENV_EXAMPLE.read_text(encoding="utf-8"))


def test_parse_env_skips_comments_and_rejects_bad_lines():
    """A parser that kept comments or accepted a bare word would hide template mistakes."""
    assert parse_env("# c\n\nA=1\nB=\n") == [("A", "1"), ("B", "")]
    with pytest.raises(ValueError):
        parse_env("JUSTAWORD\n")


def test_keys_are_exactly_the_agreed_set(pairs):
    """A template with a missing or an extra key (such as HF_HUB_OFFLINE) fails."""
    assert {key for key, _ in pairs} == EXPECTED_KEYS


def test_no_key_appears_twice(pairs):
    """A template that repeats a key lets the later value silently win."""
    repeated = [key for key, count in Counter(k for k, _ in pairs).items() if count > 1]
    assert repeated == []


@pytest.mark.parametrize("key", SECRET_KEYS)
def test_secret_values_are_empty(pairs, key):
    """A template that ships a filled-in secret leaks a real credential."""
    assert dict(pairs)[key] == ""


@pytest.mark.parametrize("key", LOCAL_URL_KEYS)
def test_services_point_at_local_host(pairs, key):
    """A template pointing at another host sends data off the machine."""
    assert urlsplit(dict(pairs)[key]).hostname == LOCAL_HOST
