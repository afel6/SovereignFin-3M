"""Fetch the pinned 10-K question/answer file from Hugging Face and verify its sha256.

Usage: uv run python -m data_pipeline.download [--out data/raw/financial_qa_10k.parquet]

Runs online by design: it talks to huggingface.co over HTTPS directly, so HF_HUB_OFFLINE
(set only by the query-time server) does not affect it. A file whose sha256 differs from the
pin is never written.
"""

import argparse
import hashlib
import os
import sys
from contextlib import closing
from pathlib import Path

import httpx

REPO_ID = "virattt/financial-qa-10K"
REVISION = "30e48b49ef40389cce6488b0bddef7d6f00f4c7a"
FILENAME = "data/train-00000-of-00001.parquet"
PINNED_SHA256 = "4b71e8cb8d0ce7a01eb117d3ef55afa994ab47571728ac8b2fa052e061d0a066"
URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{FILENAME}"
DEFAULT_OUT = Path("data/raw/financial_qa_10k.parquet")
CONNECT_TIMEOUT_SECONDS = 10.0
READ_TIMEOUT_SECONDS = 60.0
CHUNK_BYTES = 1 << 16
USER_AGENT = "dataset-download"
SETTINGS_LINE = (
    "download: the proxy or TLS settings cannot be used; "
    "check HTTPS_PROXY, HTTP_PROXY, ALL_PROXY and SSL_CERT_FILE."
)


class DownloadError(Exception):
    """An expected failure, reported as one line."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_response(client: httpx.Client) -> httpx.Response:
    """Send the request and return the streaming response (redirects followed)."""
    try:
        return client.send(client.build_request("GET", URL), stream=True)
    except UnicodeError as err:
        # A proxy or redirect host that cannot be encoded, such as "a..b", fails while
        # connecting with an error httpx does not wrap.
        raise DownloadError(
            f"download: the request to huggingface.co failed ({type(err).__name__})."
        ) from None


def _fetch_to(client: httpx.Client, part: Path) -> str:
    """Stream URL into part; return the sha256 of what was written."""
    digest = hashlib.sha256()
    with closing(_open_response(client)) as response:
        if response.status_code != httpx.codes.OK:
            raise DownloadError(f"download: Hugging Face answered HTTP {response.status_code}.")
        with part.open("wb") as handle:
            for chunk in response.iter_bytes(CHUNK_BYTES):
                digest.update(chunk)
                handle.write(chunk)
    return digest.hexdigest()


def _open_client(transport: httpx.BaseTransport | None) -> httpx.Client:
    """Build the client; httpx reads the proxy and CA bundle variables here."""
    timeout = httpx.Timeout(READ_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
    headers = {"User-Agent": USER_AGENT}
    try:
        return httpx.Client(
            transport=transport, timeout=timeout, headers=headers, follow_redirects=True
        )
    except (ImportError, ValueError, OSError, httpx.InvalidURL):
        # The variable values may hold proxy credentials, so the line never repeats them.
        raise DownloadError(SETTINGS_LINE) from None


def download(out: Path, transport: httpx.BaseTransport | None = None) -> str:
    """Make out hold the pinned file; return a one-line status message."""
    if out.is_dir():
        raise DownloadError(f"download: {out} is a directory; pass a file path.")
    if out.is_file() and file_sha256(out) == PINNED_SHA256:
        return f"download: {out} is already present and matches the pinned sha256."
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    try:
        with _open_client(transport) as client:
            digest = _fetch_to(client, part)
        if digest != PINNED_SHA256:
            raise DownloadError(
                "download: the fetched file does not match the pinned sha256; nothing was written."
            )
        os.replace(part, out)
    finally:
        part.unlink(missing_ok=True)
    return f"download: wrote {out} (sha256 verified)."


def _os_error_line(out: Path, err: OSError) -> str:
    if isinstance(err, PermissionError):
        return f"download: permission denied writing {out}."
    if isinstance(err, (NotADirectoryError, FileExistsError)):
        return f"download: a file is in the way of the folder for {out}."
    return f"download: cannot write {out} ({err.strerror or type(err).__name__})."


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the pinned raw 10-K Q/A parquet file.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="destination file")
    args = parser.parse_args(argv)
    try:
        print(download(args.out, transport))
    except DownloadError as err:
        print(err)
        return 1
    except httpx.TimeoutException:
        print("download: the request to huggingface.co timed out.")
        return 1
    except httpx.TransportError as err:
        print(f"download: could not reach huggingface.co ({type(err).__name__}).")
        return 1
    except httpx.HTTPError as err:
        print(f"download: the request to huggingface.co failed ({type(err).__name__}).")
        return 1
    except OSError as err:
        print(_os_error_line(args.out, err))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
