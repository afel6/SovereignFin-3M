"""Dataset validator (docs/contracts.md, "Validator checks").

Usage: python -m data_pipeline.contract.validate --cmd "python -m data_pipeline.clean"
Runs the command twice on the raw file and checks V1 to V11 and the import ban.
"""

import argparse
import ast
import importlib
import io
import re
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data_pipeline.contract.checks import cap, check_outputs, emit, report_outputs
from data_pipeline.contract.normalize import normalize_text
from data_pipeline.contract.reference import (
    CLEANED_FILENAME,
    DROP_LOG_FILENAME,
    RAW_COLUMNS,
    TEXT_COLUMNS,
    load_tickers,
    read_raw,
)
from data_pipeline.download import PINNED_SHA256, file_sha256

__all__ = ["check_outputs", "main"]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROWS = 7000
FILING_PATTERN = re.compile(r"^\d{4}_10K$")
KEPT_BAND = (6975, 6995)
ALLOWED_PREFIXES = ("data_pipeline/clean/", "database/tenk/")
CLEAN_PACKAGE_DIR = REPO_ROOT / "data_pipeline" / "clean"
CLEAN_PACKAGE = "data_pipeline.clean"
BANNED_MODULE = "data_pipeline.contract"
CMD_TIMEOUT_SECONDS = 300
GIT_TIMEOUT_SECONDS = 60
DEFAULT_RAW = Path("data/raw/financial_qa_10k.parquet")
ADVERSARIAL_STRINGS: tuple[str, ...] = (
    "<<a>/b>",
    "a b",
    "a​b",
    "−5",
    "x¹",
    "загрузить",
    "a\tb",
    "a\nb",
    "a\x1cb",
    "a\x1c b",
    "</...>",
    "‘q’ “q”",
    "™ ﬁ",
    "é",
    "a­b﻿",
    "‑—–－",
    "a\u0085b　c",
    "a<1>b",
    "᠎",
    "",
    "  a   b  ",
    "<.x>y",
    "a b c",
    "Å é α ı € §",
    "<a\n>b",
)


class ValidationError(Exception):
    """An expected failure, reported as one line."""


def _first_line(error: BaseException) -> str:
    lines = ascii(str(error)).splitlines()
    return lines[0][:160] if lines else type(error).__name__


def check_raw_file(
    path: Path, expected_sha256: str = PINNED_SHA256, expected_rows: int = EXPECTED_ROWS
) -> list[str]:
    """V1: the raw file matches the pin, the row count, the columns and the filing format."""
    try:
        digest = file_sha256(path)
        table = pq.read_table(path)
    except (OSError, pa.ArrowException) as error:
        return [f"V1 raw file {ascii(str(path))} unreadable: {_first_line(error)}"]
    lines = []
    if digest != expected_sha256:
        lines.append(f"V1 sha256 is {digest}, expected {expected_sha256}")
    if table.num_rows != expected_rows:
        lines.append(f"V1 row count is {table.num_rows}, expected {expected_rows}")
    names = table.column_names
    if len(names) != len(RAW_COLUMNS) or set(names) != set(RAW_COLUMNS):
        lines.append(f"V1 columns are {ascii(names)}, expected {list(RAW_COLUMNS)}")
    else:
        filings = table.column("filing").to_pylist()
        bad = [f for f in filings if not isinstance(f, str) or not FILING_PATTERN.match(f)]
        lines += [f"V1 filing {ascii(f)} does not match ^\\d{{4}}_10K$" for f in bad]
    return cap("V1", lines)


def check_normalizer(raw: pd.DataFrame, normalize_fn: Callable[[str], str]) -> list[str]:
    """V9: normalize_fn is idempotent on raw values and equals the reference on edge cases."""
    lines: list[str] = []
    values = {v for column in TEXT_COLUMNS for v in raw[column].tolist()}
    for value in sorted(values):
        try:
            once = normalize_fn(value)
            if normalize_fn(once) != once:
                lines.append(f"V9 not idempotent on {ascii(value)[:80]}")
        except Exception as error:  # reported as a failure line
            lines.append(f"V9 normalize_text raised {type(error).__name__} on {ascii(value)[:80]}")
    for text in ADVERSARIAL_STRINGS:
        try:
            got = normalize_fn(text)
        except Exception as error:  # reported as a failure line
            lines.append(f"V9 normalize_text raised {type(error).__name__} on {ascii(text)}")
            continue
        if got != normalize_text(text):
            lines.append(
                f"V9 {ascii(text)} gives {ascii(got)}, expected {ascii(normalize_text(text))}"
            )
    return cap("V9", lines)


def _read_pair(first: Path, second: Path) -> list[str]:
    try:
        if not pq.read_table(first).equals(pq.read_table(second)):
            return ["V2 the two cleaned tables differ"]
    except (OSError, pa.ArrowException) as error:
        return [f"V2 cleaned parquet unreadable: {_first_line(error)}"]
    return []


def check_determinism(first_dir: Path, second_dir: Path) -> list[str]:
    """V2: two runs give equal tables and identical drop-log bytes."""
    lines = []
    for name in (CLEANED_FILENAME, DROP_LOG_FILENAME):
        lines += [
            f"V2 {name} is missing" for d in (first_dir, second_dir) if not (d / name).is_file()
        ]
    if lines:
        return lines
    lines = _read_pair(first_dir / CLEANED_FILENAME, second_dir / CLEANED_FILENAME)
    try:
        same = (first_dir / DROP_LOG_FILENAME).read_bytes() == (
            second_dir / DROP_LOG_FILENAME
        ).read_bytes()
    except OSError as error:
        return [*lines, f"V2 drop log unreadable: {_first_line(error)}"]
    return lines if same else [*lines, "V2 the two drop logs differ"]


def check_kept_count(kept: int, band: tuple[int, int] = KEPT_BAND) -> list[str]:
    """V10: the kept row count lies in the inclusive band."""
    low, high = band
    return [] if low <= kept <= high else [f"V10 kept {kept} rows, expected {low} to {high}"]


def check_tickers(raw: pd.DataFrame, tickers: pd.DataFrame) -> list[str]:
    """V7: tickers.csv has unique tickers and exactly the tickers of the raw file."""
    listed = tickers["ticker"].tolist()
    lines = [
        f"V7 ticker {ascii(t)} is listed twice"
        for t in sorted({t for t in listed if listed.count(t) > 1})
    ]
    present = set(raw["ticker"].tolist())
    lines += [
        f"V7 raw ticker {ascii(t)} is not in tickers.csv" for t in sorted(present - set(listed))
    ]
    lines += [
        f"V7 tickers.csv ticker {ascii(t)} is not in the raw file"
        for t in sorted(set(listed) - present)
    ]
    return cap("V7", lines)


def check_scope(paths: Iterable[str]) -> list[str]:
    """V11: every changed path starts with an allowed prefix."""
    lines = [
        f"V11 {ascii(p)} is outside the allowed paths"
        for p in paths
        if not p.startswith(ALLOWED_PREFIXES)
    ]
    return cap("V11", lines)


def changed_paths(base: str) -> list[str]:
    """Return the repo-relative paths changed between base and HEAD."""
    if base.startswith("-"):
        raise ValidationError(f"--base {ascii(base)} is not a git revision")
    try:
        done = subprocess.run(
            ["git", "diff", "--name-only", f"{base}...HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValidationError(f"git diff failed: {_first_line(error)}") from error
    if done.returncode != 0:
        raise ValidationError(
            f"git diff against {ascii(base)} failed (exit code {done.returncode})"
        )
    return [line for line in done.stdout.splitlines() if line]


def _is_banned(parts: list[str]) -> bool:
    banned = BANNED_MODULE.split(".")
    return parts[: len(banned)] == banned


def _import_base(file: Path, root: Path, package: str, level: int) -> list[str]:
    """The package a relative import of the given level starts from, as name parts."""
    here = [*package.split("."), *file.relative_to(root).parent.parts]
    return here[: len(here) - (level - 1)] if level - 1 <= len(here) else []


def _banned_names(tree: ast.AST, file: Path, root: Path, package: str) -> list[int]:
    """Line numbers of imports that reach the banned module."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hit = any(_is_banned(a.name.split(".")) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _import_base(file, root, package, node.level) if node.level else []
            target = [*base, *(node.module.split(".") if node.module else [])]
            hit = _is_banned(target) or any(_is_banned([*target, a.name]) for a in node.names)
        elif isinstance(node, ast.Call):
            hit = _dynamic_import_hits(node)
        else:
            continue
        if hit:
            found.append(node.lineno)
    return found


def _dynamic_import_hits(node: ast.Call) -> bool:
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if name not in ("import_module", "__import__") or not node.args:
        return False
    first = node.args[0]
    return (
        isinstance(first, ast.Constant)
        and isinstance(first.value, str)
        and _is_banned(first.value.split("."))
    )


def check_import_ban(
    package_dir: Path = CLEAN_PACKAGE_DIR, package: str = CLEAN_PACKAGE
) -> list[str]:
    """Import ban: no file under package_dir imports data_pipeline.contract."""
    if not package_dir.is_dir():
        return []
    lines = []
    for file in sorted(package_dir.rglob("*.py")):
        label = ascii(file.relative_to(package_dir).as_posix())
        try:
            tree = ast.parse(file.read_bytes())
        except (SyntaxError, ValueError, OSError) as error:
            lines.append(f"import-ban {label} cannot be parsed: {_first_line(error)}")
            continue
        lines += [
            f"import-ban {label} line {n} imports {BANNED_MODULE}"
            for n in _banned_names(tree, file, package_dir, package)
        ]
    return cap("import-ban", lines)


def module_from_cmd(cmd: str) -> str | None:
    """Return the module named after "-m" in the command, or None."""
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        return None
    for token, following in zip(tokens, tokens[1:], strict=False):
        if token == "-m":
            return following
    return None


def _run_command(tokens: list[str], raw: Path, out_dir: Path) -> str | None:
    """Run the cleaner once; return a one-line failure summary or None."""
    argv = [*tokens, "--raw", str(raw), "--out-dir", str(out_dir)]
    if argv[0] in ("python", "python3"):
        argv[0] = sys.executable
    try:
        done = subprocess.run(
            argv, cwd=REPO_ROOT, capture_output=True, timeout=CMD_TIMEOUT_SECONDS, check=False
        )
    except subprocess.TimeoutExpired:
        return f"the command timed out after {CMD_TIMEOUT_SECONDS} s"
    except OSError:
        return "the command could not start"
    return None if done.returncode == 0 else f"the command exited with code {done.returncode}"


def _check_module(cmd: str, raw: pd.DataFrame) -> list[str]:
    name = module_from_cmd(cmd)
    if name is None:
        return ["V9 --cmd has no -m module to import"]
    try:
        module = importlib.import_module(name)
    except Exception as error:  # reported as a failure line
        return [f"V9 cannot import {ascii(name)}: {type(error).__name__}"]
    fn = getattr(module, "normalize_text", None)
    if not callable(fn):
        return [f"V9 {ascii(name)} has no normalize_text"]
    return check_normalizer(raw, fn)


def _kept_rows(out_dir: Path) -> int:
    try:
        return pq.ParquetFile(out_dir / CLEANED_FILENAME).metadata.num_rows
    except (OSError, pa.ArrowException):
        return 0


def _validate(args: argparse.Namespace) -> bool:
    try:
        tokens = shlex.split(args.cmd)
    except ValueError as error:
        raise ValidationError(f"--cmd cannot be parsed: {_first_line(error)}") from error
    if not tokens:
        raise ValidationError("--cmd is empty")
    if not args.raw.is_file():
        raise ValidationError(
            f"raw file {args.raw!a} is missing or not a file; run python -m data_pipeline.download"
        )
    if not emit("V1 raw input", check_raw_file(args.raw, PINNED_SHA256, EXPECTED_ROWS)):
        return False
    ok = emit("import-ban", check_import_ban())
    with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
        outs = (Path(one), Path(two))
        with ThreadPoolExecutor(max_workers=len(outs)) as pool:
            failures = [f for f in pool.map(lambda o: _run_command(tokens, args.raw, o), outs) if f]
        if failures:
            raise ValidationError(failures[0])
        raw, tickers = read_raw(args.raw), load_tickers()
        ok &= emit("V2 determinism", check_determinism(Path(one), Path(two)))
        ok &= report_outputs(check_outputs(raw, Path(one), tickers))
        ok &= emit("V9 normalize_text", _check_module(args.cmd, raw))
        ok &= emit("V10 kept count", check_kept_count(_kept_rows(Path(one)), KEPT_BAND))
        ok &= emit("V7 tickers", check_tickers(raw, tickers))
    if args.base is None:
        print("V11 scope: skipped (no --base)")
        return ok
    return emit("V11 scope", check_scope(changed_paths(args.base))) and ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="validate", description=__doc__)
    parser.add_argument("--cmd", required=True)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--base", default=None)
    args = parser.parse_args(argv)
    report = io.StringIO()
    try:
        with redirect_stdout(report):
            passed = _validate(args)
    except (OSError, ValidationError, ValueError, pa.ArrowException) as error:
        print(f"validator: {_first_line(error)}")
        return 1
    print(report.getvalue(), end="")
    print("validator: passed" if passed else "validator: failed")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
