"""Import rules: leaf modules stay light, engine folders reach orchestrator only via two modules."""

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LEAF_ALLOWED = set(sys.stdlib_module_names) | {"pydantic", "httpx", "__future__"}
SHARED = {"llm", "api_contract"}
ENGINE_FOLDERS = ["rag_engine", "rdl_engine", "atlas_kv"]


def leaf_violations(source: str) -> list[str]:
    """Return imports of a leaf module that are not stdlib, pydantic or httpx, or are relative."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] not in LEAF_ALLOWED]
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                found.append("." * node.level + (node.module or ""))
            elif node.module.split(".")[0] not in LEAF_ALLOWED:
                found.append(node.module)
    return found


def engine_violations(source: str) -> list[str]:
    """Return imports that reach orchestrator other than orchestrator.llm and api_contract."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "orchestrator" and (len(parts) == 1 or parts[1] not in SHARED):
                    found.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            parts = node.module.split(".")
            if parts[0] != "orchestrator":
                continue
            if len(parts) == 1:
                found += [f"orchestrator.{a.name}" for a in node.names if a.name not in SHARED]
            elif parts[1] not in SHARED:
                found.append(node.module)
    return found


@pytest.mark.parametrize("name", ["llm.py", "api_contract.py"])
def test_leaf_modules_import_only_light_packages(name):
    """Fails if llm.py or api_contract.py imports a heavy, third-party or relative module."""
    source = (ROOT / "orchestrator" / name).read_text(encoding="utf-8")
    assert leaf_violations(source) == []


def test_engine_folders_do_not_import_other_orchestrator_modules():
    """Fails if any file in an engine folder imports orchestrator beyond llm/api_contract."""
    report = []
    for folder in ENGINE_FOLDERS:
        for path in sorted((ROOT / folder).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    line = ast.unparse(node)
                    if engine_violations(line):
                        report.append(f"{path.relative_to(ROOT)}:{node.lineno}: {line}")
    assert report == []


@pytest.mark.parametrize(
    "source",
    [
        "import orchestrator",
        "import orchestrator as o",
        "import orchestrator.app",
        "from orchestrator import mock_server",
        "from orchestrator import llm, app",
        "from orchestrator.mock_server import app",
        "def f():\n    import orchestrator.app",
    ],
)
def test_engine_checker_flags(source):
    """Fails if the engine checker misses a forbidden import form."""
    assert engine_violations(source)


@pytest.mark.parametrize(
    "source",
    [
        "import orchestrator.llm",
        "from orchestrator import llm, api_contract",
        "from orchestrator.llm import complete_text",
        "from orchestrator.api_contract import ChartSpec",
        "import orchestrator_tools",
    ],
)
def test_engine_checker_allows(source):
    """Fails if the engine checker flags an allowed import or a differently named package."""
    assert engine_violations(source) == []


@pytest.mark.parametrize(
    "source",
    [
        "import fastapi",
        "from orchestrator import api_contract",
        "from . import x",
        "def f():\n    import numpy",
    ],
)
def test_leaf_checker_flags(source):
    """Fails if the leaf checker misses a forbidden import form."""
    assert leaf_violations(source)


@pytest.mark.parametrize(
    "source",
    [
        "import json",
        "from typing import Literal",
        "import httpx",
        "from pydantic import BaseModel",
        "from __future__ import annotations",
    ],
)
def test_leaf_checker_allows(source):
    """Fails if the leaf checker flags an allowed import."""
    assert leaf_violations(source) == []
