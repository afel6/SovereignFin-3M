"""The local gate runs the dependency sync, then every check, and reports failures plainly."""

import subprocess
from pathlib import Path

import pytest

from scripts import gate

ROOT = Path(__file__).resolve().parent.parent
SYNC = ["uv", "sync", "--locked", "--all-groups"]
CHECKS = [
    ["uv", "run", "--no-sync", "ruff", "check"],
    ["uv", "run", "--no-sync", "ruff", "format", "--check"],
    ["uv", "run", "--no-sync", "pytest", "--cov"],
]
ALL_COMMANDS = [SYNC, *CHECKS]
FAILED_EXIT = 3
ERRNO_IO = 5


class FakeRun:
    """Stand-in for subprocess.run: one scripted outcome (exit code or exception) per call."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((list(args), kwargs))
        outcome = self.outcomes[len(self.calls) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return subprocess.CompletedProcess(args, outcome)


@pytest.fixture
def run(monkeypatch):
    def install(outcomes):
        fake = FakeRun(outcomes)
        monkeypatch.setattr(gate.subprocess, "run", fake)
        return fake

    return install


def test_all_pass_returns_zero_with_one_line(run, capsys):
    """A gate that printed nothing or several lines on success hides its verdict."""
    run([0, 0, 0, 0])
    assert gate.main() == 0
    assert len(capsys.readouterr().out.splitlines()) == 1


def test_commands_run_in_order_from_repo_root(run):
    """Running from another folder or in another order checks the wrong project state."""
    fake = run([0, 0, 0, 0])
    gate.main()
    assert [c for c, _ in fake.calls] == ALL_COMMANDS
    assert all(Path(kw["cwd"]).resolve() == ROOT for _, kw in fake.calls)


def test_failed_sync_stops_everything(run, capsys):
    """Running checks on an unsynced environment gives misleading results."""
    fake = run([FAILED_EXIT])
    assert gate.main() == 1
    assert len(fake.calls) == 1
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert "dependency sync" in lines[0]


def test_one_failed_check_still_runs_the_rest(run, capsys):
    """Stopping at the first failed check hides further problems."""
    fake = run([0, FAILED_EXIT, 0, 0])
    assert gate.main() == 1
    assert len(fake.calls) == 4
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert "lint" in lines[0]


def test_two_failed_checks_give_two_lines(run, capsys):
    """Reporting only one of two failures leaves the other unseen."""
    run([0, FAILED_EXIT, 0, FAILED_EXIT])
    assert gate.main() == 1
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert "lint" in lines[0]
    assert "tests" in lines[1]


def test_missing_uv_gives_one_friendly_line(run, capsys):
    """A traceback instead of advice to install uv is unfriendly."""
    run([FileNotFoundError(2, "No such file")])
    assert gate.main() == 1
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert len(lines) == 1
    assert "uv" in lines[0]
    assert "Traceback" not in captured.out + captured.err


@pytest.mark.parametrize(
    "error",
    [PermissionError(13, "Permission denied"), OSError(ERRNO_IO, "Input/output error")],
    ids=["permission", "generic"],
)
def test_other_start_errors_give_one_line(run, capsys, error):
    """An unhandled start error would crash with a traceback."""
    run([error])
    assert gate.main() == 1
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert "Traceback" not in captured.out + captured.err
