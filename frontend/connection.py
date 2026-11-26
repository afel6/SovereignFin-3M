"""Construct the dashboard HTTP session from runtime configuration."""

import os

import httpx

TIMEOUT_SECONDS = 100
DEFAULT_API_BASE = "http://127.0.0.1:8000"


class ApiConfigurationError(ValueError):
    def __init__(self) -> None:
        super().__init__("The API address is invalid.")


def create_client(*, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    base = os.getenv("SF_API_BASE") or DEFAULT_API_BASE
    try:
        url = httpx.URL(base)
    except httpx.InvalidURL:
        raise ApiConfigurationError from None
    if (
        url.scheme not in {"http", "https"}
        or not url.host
        or url.userinfo
        or "?" in base
        or "#" in base
    ):
        raise ApiConfigurationError
    return httpx.Client(
        base_url=url,
        timeout=TIMEOUT_SECONDS,
        transport=transport,
        trust_env=False,
        follow_redirects=False,
    )
