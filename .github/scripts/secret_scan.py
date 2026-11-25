"""Fail when a tracked file contains a committed API key or private key.

Usage: python .github/scripts/secret_scan.py   (stdlib only; run from anywhere inside the repo)
Prints only the names of matching files, never the matched text, because CI logs are public.
Exit codes: 0 clean, 1 key pattern found, 2 the scan could not run.
"""

import re
import subprocess
import sys
from pathlib import Path

EXIT_CLEAN = 0
EXIT_FOUND = 1
EXIT_ERROR = 2
MIN_KEY_CHARS = 20
GOOGLE_KEY_CHARS = 35
AWS_KEY_ID_CHARS = 16

# A key must start at a word boundary, so names such as "risk-adjusted-..." do not match.
_START = rb"(?<![A-Za-z0-9_-])"
KEY_PATTERNS = (
    rb"nvapi-[A-Za-z0-9_-]{%d,}" % MIN_KEY_CHARS,  # NVIDIA build API
    rb"sk-[A-Za-z0-9_-]{%d,}" % MIN_KEY_CHARS,  # OpenAI-style, incl. OpenRouter sk-or-
    rb"gsk_[A-Za-z0-9]{%d,}" % MIN_KEY_CHARS,  # Groq
    rb"AIza[0-9A-Za-z_-]{%d}" % GOOGLE_KEY_CHARS,  # Google API key
    rb"hf_[A-Za-z0-9]{%d,}" % MIN_KEY_CHARS,  # Hugging Face
    rb"gh[pousr]_[A-Za-z0-9]{%d,}" % MIN_KEY_CHARS,  # GitHub tokens
    rb"github_pat_[A-Za-z0-9_]{%d,}" % MIN_KEY_CHARS,  # GitHub fine-grained token
    rb"(?:AKIA|ASIA)[0-9A-Z]{%d}" % AWS_KEY_ID_CHARS,  # AWS access key id
)
KEY_RE = re.compile(_START + rb"(?:" + rb"|".join(KEY_PATTERNS) + rb")")
PRIVATE_KEY_RE = re.compile(rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")


class ScanError(Exception):
    """The scan could not run; the message is one friendly line."""


def contains_key(data: bytes) -> bool:
    """True when the bytes hold an API key or a private-key header."""
    return bool(KEY_RE.search(data) or PRIVATE_KEY_RE.search(data))


def git_output(args: list[str], cwd: Path, failure_line: str) -> str:
    """Stdout of one git command; ScanError with a friendly line when it cannot run."""
    try:
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True)
    except FileNotFoundError as err:
        raise ScanError("secret scan: 'git' was not found on PATH.") from err
    except subprocess.CalledProcessError as err:
        raise ScanError(failure_line) from err
    return done.stdout.decode("utf-8", errors="surrogateescape")


def tracked_files(cwd: Path) -> tuple[Path, list[str]]:
    """Repository root and the paths git tracks there (relative, as git prints them)."""
    top = git_output(
        ["rev-parse", "--show-toplevel"], cwd, "secret scan: not inside a git repository."
    )
    root = Path(top.rstrip("\n"))
    listed = git_output(["ls-files", "-z"], root, "secret scan: git could not list tracked files.")
    return root, [name for name in listed.split("\0") if name]


def files_with_keys(root: Path, names: list[str]) -> list[str]:
    """Names of the files whose content contains a key."""
    found = []
    for name in names:
        path = root / name
        if path.is_dir():
            continue  # a submodule or a link to a folder: git grep skips these too
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            continue  # deleted in the working tree: no content left to scan
        except OSError as err:
            line = f"secret scan: could not read a tracked file ({err.strerror})."
            raise ScanError(line) from err
        if contains_key(data):
            found.append(name)
    return found


def main() -> int:
    try:
        root, names = tracked_files(Path.cwd())
        found = files_with_keys(root, names)
    except ScanError as err:
        print(err)
        return EXIT_ERROR
    if not found:
        print("No key patterns found.")
        return EXIT_CLEAN
    for name in found:
        print(f"KEY PATTERN IN: {name!r}")
    print("::error::Committed key pattern found. Remove it and rotate the key.")
    return EXIT_FOUND


if __name__ == "__main__":
    sys.exit(main())
