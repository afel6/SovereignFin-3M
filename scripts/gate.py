"""Local quality gate: the same commands as the gate step of CI's lint-unit job.

Usage: uv run python scripts/gate.py
Prints one line per failed step and exits non-zero if any step fails.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNC_STEP = ("dependency sync", ["uv", "sync", "--locked", "--all-groups"])
CHECK_STEPS = (
    ("lint", ["uv", "run", "--no-sync", "ruff", "check"]),
    ("format check", ["uv", "run", "--no-sync", "ruff", "format", "--check"]),
    ("tests", ["uv", "run", "--no-sync", "pytest", "--cov"]),
)


class ToolNotStarted(Exception):
    """The command could not be started at all (for example, uv is not installed)."""


def start_error_line(command: list[str], err: OSError) -> str:
    """One friendly line for an OS error raised while starting a command."""
    tool = command[0]
    if isinstance(err, FileNotFoundError):
        return f"gate: '{tool}' was not found on PATH; install uv (https://docs.astral.sh/uv/)."
    if isinstance(err, PermissionError):
        return f"gate: permission denied when starting '{tool}'."
    return f"gate: could not start '{tool}' ({err.strerror or type(err).__name__})."


def run_step(name: str, command: list[str]) -> str | None:
    """Run one step; return a failure line, or None when it passed."""
    try:
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    except OSError as err:
        raise ToolNotStarted(start_error_line(command, err)) from err
    if result.returncode != 0:
        return f"gate: {name} failed (exit {result.returncode}): {' '.join(command)}"
    return None


def main() -> int:
    try:
        sync_failure = run_step(*SYNC_STEP)
        if sync_failure:
            print(sync_failure)
            return 1
        failures = [line for line in (run_step(*step) for step in CHECK_STEPS) if line]
    except ToolNotStarted as err:
        print(err)
        return 1
    for line in failures:
        print(line)
    if failures:
        return 1
    print("gate: all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
