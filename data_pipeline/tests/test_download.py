import hashlib
from pathlib import Path

import httpx
import pytest

from data_pipeline import download

PAYLOAD = b"fake parquet bytes"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()
REAL_PIN = "4b71e8cb8d0ce7a01eb117d3ef55afa994ab47571728ac8b2fa052e061d0a066"
REVISION = "30e48b49ef40389cce6488b0bddef7d6f00f4c7a"
ENV_SETTINGS_LINE = (
    "download: the proxy or TLS settings cannot be used; "
    "check HTTPS_PROXY, HTTP_PROXY, ALL_PROXY and SSL_CERT_FILE."
)
NETWORK_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
)


class Recorder:
    """Transport stand-in that answers with a fixed response and records requests."""

    def __init__(
        self,
        status: int = 200,
        body: bytes = PAYLOAD,
        error: Exception | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.requests: list[httpx.Request] = []
        self._status, self._body, self._error = status, body, error
        self._headers = headers

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return httpx.Response(
            self._status, headers=self._headers, stream=httpx.ByteStream(self._body)
        )

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setattr(download, "PINNED_SHA256", PAYLOAD_SHA)


@pytest.fixture
def clean_network_env(monkeypatch):
    """Clear proxy and TLS variables and make any real request fail the test."""
    for name in NETWORK_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)

    def refuse(client, part):
        raise AssertionError("a request was attempted")

    monkeypatch.setattr(download, "_fetch_to", refuse)


def run(out: Path, recorder: Recorder, capsys) -> tuple[int, list[str]]:
    code = download.main(["--out", str(out)], transport=recorder.transport)
    return code, capsys.readouterr().out.splitlines()


def test_good_bytes_are_written_exactly(pinned, tmp_path, capsys):
    out = tmp_path / "f.parquet"
    code, lines = run(out, Recorder(), capsys)
    assert code == 0
    assert len(lines) == 1
    assert out.read_bytes() == PAYLOAD
    assert not out.with_name("f.parquet.part").exists()


def test_mismatching_bytes_are_refused(pinned, tmp_path, capsys):
    out = tmp_path / "f.parquet"
    code, lines = run(out, Recorder(body=b"other"), capsys)
    assert code == 1
    assert len(lines) == 1
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


def test_wrong_existing_file_is_replaced_by_verified_bytes(pinned, tmp_path, capsys):
    out = tmp_path / "f.parquet"
    out.write_bytes(b"stale")
    code, _ = run(out, Recorder(), capsys)
    assert code == 0
    assert out.read_bytes() == PAYLOAD


def test_wrong_existing_file_is_kept_when_fetched_bytes_mismatch(pinned, tmp_path, capsys):
    out = tmp_path / "f.parquet"
    out.write_bytes(b"stale")
    code, _ = run(out, Recorder(body=b"other"), capsys)
    assert code == 1
    assert out.read_bytes() == b"stale"
    assert [p.name for p in tmp_path.iterdir()] == ["f.parquet"]


def test_matching_existing_file_needs_no_request(pinned, tmp_path, capsys):
    out = tmp_path / "f.parquet"
    out.write_bytes(PAYLOAD)
    recorder = Recorder()
    code, lines = run(out, recorder, capsys)
    assert code == 0
    assert len(lines) == 1
    assert recorder.requests == []


@pytest.mark.parametrize("status", [404, 500])
def test_http_error_status_is_named(pinned, tmp_path, capsys, status):
    out = tmp_path / "f.parquet"
    code, lines = run(out, Recorder(status=status), capsys)
    assert code == 1
    assert len(lines) == 1
    assert str(status) in lines[0]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectError("no route"), httpx.ReadTimeout("slow")],
    ids=["connect-error", "read-timeout"],
)
def test_network_errors_give_one_line(pinned, tmp_path, capsys, error):
    out = tmp_path / "f.parquet"
    code, lines = run(out, Recorder(error=error), capsys)
    assert code == 1
    assert len(lines) == 1
    assert "Traceback" not in lines[0]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("HTTPS_PROXY", "socks5://127.0.0.1:9"),
        ("HTTPS_PROXY", "ftp://127.0.0.1:9"),
        ("HTTPS_PROXY", "http://127.0.0.1:abc"),
        ("SSL_CERT_FILE", "{tmp}/missing-ca.pem"),
    ],
    ids=["socks-without-extra", "unknown-scheme", "bad-port", "missing-ca-file"],
)
def test_unusable_proxy_or_tls_settings_give_one_line(
    clean_network_env, monkeypatch, tmp_path, capsys, name, value
):
    monkeypatch.setenv(name, value.format(tmp=tmp_path))
    out = tmp_path / "f.parquet"
    code = download.main(["--out", str(out)])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out.splitlines() == [ENV_SETTINGS_LINE]
    assert captured.err == ""
    assert not out.exists()


@pytest.mark.parametrize(
    ("recorder", "name"),
    [
        (Recorder(status=302, headers={"Location": download.URL}), "TooManyRedirects"),
        (
            Recorder(headers={"Content-Encoding": "gzip"}, body=b"not gzip data"),
            "DecodingError",
        ),
        # A proxy or redirect host such as "a..b" fails to encode while connecting.
        (Recorder(error=UnicodeError("label empty or too long")), "UnicodeError"),
    ],
    ids=["redirect-loop", "undecodable-body", "unencodable-host-name"],
)
def test_other_request_failures_give_one_line(pinned, tmp_path, capsys, recorder, name):
    code, lines = run(tmp_path / "f.parquet", recorder, capsys)
    assert code == 1
    assert lines == [f"download: the request to huggingface.co failed ({name})."]
    assert list(tmp_path.iterdir()) == []


def test_directory_destination_fails_with_one_line(pinned, tmp_path, capsys):
    code, lines = run(tmp_path, Recorder(), capsys)
    assert code == 1
    assert len(lines) == 1


def test_missing_parent_folders_are_created(pinned, tmp_path, capsys):
    out = tmp_path / "a" / "b" / "f.parquet"
    code, _ = run(out, Recorder(), capsys)
    assert code == 0
    assert out.read_bytes() == PAYLOAD


def test_offline_hub_variable_does_not_stop_download(pinned, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    out = tmp_path / "f.parquet"
    code, _ = run(out, Recorder(), capsys)
    assert code == 0
    assert out.read_bytes() == PAYLOAD


def test_request_targets_pinned_file_without_personal_user_agent(pinned, tmp_path, capsys):
    recorder = Recorder()
    run(tmp_path / "f.parquet", recorder, capsys)
    request = recorder.requests[0]
    url = str(request.url)
    assert "virattt/financial-qa-10K" in url
    assert REVISION in url
    assert "data/train-00000-of-00001.parquet" in url
    agent = request.headers["user-agent"]
    assert "@" not in agent
    assert "sovereignfin" not in agent.lower()


def test_pin_matches_documented_sha256():
    assert download.PINNED_SHA256 == REAL_PIN


@pytest.mark.needs_data
def test_downloaded_raw_file_matches_pin():
    assert download.file_sha256(Path("data/raw/financial_qa_10k.parquet")) == REAL_PIN
