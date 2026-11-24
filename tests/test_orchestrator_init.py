"""The orchestrator package marker carries no code."""

from pathlib import Path

INIT_FILE = Path(__file__).resolve().parent.parent / "orchestrator" / "__init__.py"


def test_orchestrator_init_is_zero_bytes():
    """An init file holding a docstring or an import would run code on package import."""
    assert INIT_FILE.is_file()
    assert INIT_FILE.stat().st_size == 0
