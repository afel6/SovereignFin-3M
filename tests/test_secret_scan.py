"""The secret scan flags committed keys by file name only and lets documentation placeholders pass.

Every fake key is built by concatenation so the repository's own scan never sees a key literal.
"""

import importlib.util
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / ".github" / "scripts" / "secret_scan.py"
EXIT_CLEAN = 0
EXIT_FOUND = 1
EXIT_ERROR = 2

_spec = importlib.util.spec_from_file_location("secret_scan", SCRIPT)
secret_scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(secret_scan)

PEM_TAIL = "-----"
KEYS = {
    "nvidia": "nvapi-" + "a" * 30,
    "nvidia-mixed": "nvapi-" + "Ab_-" * 6,
    "openai": "sk-" + "b" * 30,
    "openai-project": "sk-proj-" + "c" * 30,
    "openrouter": "sk-or-v1-" + "0123456789abcdef" * 4,
    "groq": "gsk_" + "d" * 52,
    "google": "AIza" + "e" * 35,
    "huggingface": "hf_" + "f" * 34,
    "github-ghp": "ghp_" + "g" * 30,
    "github-gho": "gho_" + "g" * 30,
    "github-ghu": "ghu_" + "g" * 30,
    "github-ghs": "ghs_" + "g" * 30,
    "github-ghr": "ghr_" + "g" * 30,
    "github-pat": "github_pat_" + "h" * 30,
    "aws-akia": "AKIA" + "A1B2C3D4E5F6G7H8",
    "aws-asia": "ASIA" + "A1B2C3D4E5F6G7H8",
    "pem-pkcs8": "-----BEGIN " + "PRIVATE KEY" + PEM_TAIL,
    "pem-rsa": "-----BEGIN " + "RSA PRIVATE KEY" + PEM_TAIL,
    "pem-ec": "-----BEGIN " + "EC PRIVATE KEY" + PEM_TAIL,
    "pem-openssh": "-----BEGIN " + "OPENSSH PRIVATE KEY" + PEM_TAIL,
    "pem-encrypted": "-----BEGIN " + "ENCRYPTED PRIVATE KEY" + PEM_TAIL,
}
FAKE_KEY = KEYS["nvidia"]

PLACEHOLDERS = {
    "database-url": "postgresql://sf_reader:...@127.0.0.1:5432/sovereignfin",
    "empty-assignment": "NVIDIA_API_KEY=",
    "ellipsis-key": "NVIDIA_API_KEY=nvapi-...",
    "short-sk": "sk-xxxx",
    "risk-prefix": "risk-" + "a" * 25,
    "task-prefix": "task-" + "a" * 25,
    "hub-function": "hf_hub_download",
    "public-key": "-----BEGIN PUBLIC KEY-----",
    "certificate": "-----BEGIN CERTIFICATE-----",
    "google-short": "AIza" + "e" * 20,
    "aws-short": "AKIA" + "A1B2C3D4E5F6G7H",
}


@pytest.mark.parametrize("key", KEYS.values(), ids=KEYS.keys())
def test_each_provider_key_is_found(key):
    """A scan that misses one provider's key format lets that key reach the public repo."""
    assert secret_scan.contains_key(key.encode()) is True


@pytest.mark.parametrize("template", ["NVIDIA_API_KEY={}", 'token = "{}"', "x {} y"])
def test_key_inside_a_line_is_found(template):
    """A scan that only matches whole lines misses keys in env files and quoted strings."""
    assert secret_scan.contains_key(template.format(FAKE_KEY).encode()) is True


@pytest.mark.parametrize("text", PLACEHOLDERS.values(), ids=PLACEHOLDERS.keys())
def test_placeholders_and_lookalikes_pass(text):
    """A scan that flags documentation placeholders makes CI fail on harmless text."""
    assert secret_scan.contains_key(text.encode()) is False


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(monkeypatch):
    """A fresh git repository with nothing tracked; the folder is gone after the test."""
    with tempfile.TemporaryDirectory(prefix="secret-scan-test-") as tmp:
        base = Path(tmp)
        work = base / "work"
        work.mkdir()
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(base))
        _git(work, "init", "-q")
        monkeypatch.chdir(work)
        yield work
        monkeypatch.chdir(ROOT)  # a folder cannot be removed while it is the working directory


def _track(repo, name, text):
    (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", name)


def test_clean_repository_reports_no_keys(repo, capsys):
    """A scan that prints nothing or fails on a clean repository breaks every build."""
    _track(repo, "notes.txt", "NVIDIA_API_KEY=\n")
    assert secret_scan.main() == EXIT_CLEAN
    assert capsys.readouterr().out.strip() == "No key patterns found."


def test_tracked_key_fails_and_names_file_without_the_key(repo, capsys):
    """A scan that echoes the matched text leaks the key into public CI logs."""
    _track(repo, "settings.env", "NVIDIA_API_KEY=" + FAKE_KEY + "\n")
    _track(repo, "clean.txt", "nothing here\n")
    assert secret_scan.main() == EXIT_FOUND
    out = capsys.readouterr().out
    assert "settings.env" in out
    assert "clean.txt" not in out
    assert FAKE_KEY not in out
    assert FAKE_KEY[:12] not in out


def test_untracked_key_is_ignored(repo, capsys):
    """A scan of the whole folder instead of tracked files flags local, git-ignored secrets."""
    (repo / ".env").write_text("NVIDIA_API_KEY=" + FAKE_KEY + "\n", encoding="utf-8")
    assert secret_scan.main() == EXIT_CLEAN
    assert capsys.readouterr().out.strip() == "No key patterns found."


def test_tracked_file_deleted_from_disk_is_skipped(repo, capsys):
    """A scan that crashes on a deleted tracked file breaks every build with a pending removal."""
    _track(repo, "old.env", "NVIDIA_API_KEY=" + FAKE_KEY + "\n")
    (repo / "old.env").unlink()
    assert secret_scan.main() == EXIT_CLEAN
    assert capsys.readouterr().out.strip() == "No key patterns found."


def test_outside_a_repository_prints_one_line_and_exits_two(monkeypatch, capsys):
    """A scan that shows a traceback outside git, or exits 0, hides that it did not run."""
    with tempfile.TemporaryDirectory(prefix="secret-scan-test-") as tmp:
        work = Path(tmp) / "work"  # git stops searching only at a ceiling above the folder
        work.mkdir()
        monkeypatch.setenv("GIT_CEILING_DIRECTORIES", tmp)
        monkeypatch.chdir(work)
        code = secret_scan.main()
        monkeypatch.chdir(ROOT)
    out = capsys.readouterr().out
    assert code == EXIT_ERROR
    assert len(out.strip().splitlines()) == 1
    assert "Traceback" not in out
