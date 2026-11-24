"""Keeps the contributor task briefs in sync with docs/contracts.md.

Every contract block copied into docs/tasks/M*.md must equal the block in
docs/contracts.md, and every file a brief lists must sit inside its allowed paths.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "docs" / "contracts.md"
TASKS = ROOT / "docs" / "tasks"

EXPECTED_ALLOWED = {
    "M2": {"data_pipeline/clean/", "database/tenk/"},
    "M3": {"rag_engine/contrib/"},
    "M4": {"database/charts/"},
    "M5": {"atlas_kv/contrib/"},
    "M6": {"frontend/"},
    "M7": {"evaluation/safety/"},
    "M8": {"evaluation/benchmark/", "docs/presentation/"},
}
EXCLUDED = {"M8": ("docs/presentation/core/",)}

OPEN_PREFIX = "```contract"
ID_RE = re.compile(r"^[a-z0-9_-]+$")
CLOSE = "```"
ITEM_RE = re.compile(r"^\s*[-*+] (.*)$")
TICKED_RE = re.compile(r"`([^`]*)`")


def _lines(text: str) -> list[str]:
    """Split text into lines on newline only, treating CRLF as one newline."""
    return text.replace("\r\n", "\n").split("\n")


def _opening_id(line: str) -> str | None:
    """Return the id text of an opening fence line, or None when the line is not one."""
    if not line.startswith(OPEN_PREFIX):
        return None
    if not line.startswith(OPEN_PREFIX + " "):
        return line  # wrong separator or no id: the whole line is kept, so it fails ID_RE
    return line[len(OPEN_PREFIX) + 1 :]


def _blocks(text: str) -> list[tuple[str, str, int]]:
    """Return a list of (id, body, line index of the opening fence) for each contract block.

    A block without a closing fence raises ValueError.
    """
    lines = _lines(text)
    found = []
    i = 0
    while i < len(lines):
        block_id = _opening_id(lines[i])
        if block_id is not None:
            j = i + 1
            while j < len(lines) and lines[j] != CLOSE:
                j += 1
            if j >= len(lines):
                raise ValueError(f"contract block {block_id!r} has no closing fence")
            found.append((block_id, "\n".join(lines[i + 1 : j]), i))
            i = j
        i += 1
    return found


def contract_blocks(text: str) -> dict[str, str]:
    """Map each contract id to its body; a duplicate or malformed id raises ValueError."""
    result: dict[str, str] = {}
    for block_id, body, _ in _blocks(text):
        if not ID_RE.match(block_id):
            raise ValueError(f"malformed contract id: {block_id!r}")
        if block_id in result:
            raise ValueError(f"duplicate contract id: {block_id}")
        result[block_id] = body
    return result


def _slug(heading: str) -> str:
    """GitHub-style slug of a heading text."""
    return re.sub(r"[^a-z0-9 _-]", "", heading.lower()).replace(" ", "-")


def heading_problems(contracts_text: str) -> list[str]:
    """List blocks whose nearest preceding '### ' heading slug differs from the id."""
    lines = _lines(contracts_text)
    problems = []
    for block_id, _, start in _blocks(contracts_text):
        heading = next((ln[4:] for ln in reversed(lines[:start]) if ln.startswith("### ")), None)
        if not ID_RE.match(block_id):
            problems.append(f"{block_id!r}: malformed contract id")
        elif heading is None:
            problems.append(f"{block_id}: no preceding '### ' heading")
        elif _slug(heading) != block_id:
            problems.append(f"{block_id}: heading slug is '{_slug(heading)}'")
    return problems


def _first_difference(want: str, got: str) -> str:
    """Describe the first differing line between two block bodies."""
    a, b = want.split("\n"), got.split("\n")
    for n in range(max(len(a), len(b))):
        x = a[n] if n < len(a) else "<missing>"
        y = b[n] if n < len(b) else "<missing>"
        if x != y:
            return f"line {n + 1}: contracts has {x!r}, brief has {y!r}"
    return "bodies differ"


def excerpt_problems(brief_text: str, contracts: dict[str, str]) -> list[str]:
    """List brief blocks that are unknown, malformed or differ from the contracts text."""
    blocks = _blocks(brief_text)
    if not blocks:
        return ["brief has no contract block"]
    problems = []
    for block_id, body, _ in blocks:
        if not ID_RE.match(block_id):
            problems.append(f"{block_id!r}: unknown or malformed contract id")
        elif block_id not in contracts:
            problems.append(f"{block_id}: unknown contract id")
        elif body != contracts[block_id]:
            problems.append(f"{block_id}: {_first_difference(contracts[block_id], body)}")
    return problems


def _section_lines(text: str, heading: str) -> list[str]:
    """Return the lines under '## <heading>', up to the next '## ' heading."""
    lines, inside = [], False
    for line in _lines(text):
        if line.startswith("## "):
            inside = line[3:].strip() == heading
        elif inside:
            lines.append(line)
    return lines


def _item_texts(text: str, heading: str) -> list[str]:
    """Return the text of every list item ('-', '*' or '+') in '## <heading>'."""
    matches = (ITEM_RE.match(line) for line in _section_lines(text, heading))
    return [m.group(1) for m in matches if m]


def section_items(text: str, heading: str) -> list[str]:
    """Return each list item's first backticked token, or its raw text without one."""
    items = []
    for item in _item_texts(text, heading):
        ticked = TICKED_RE.search(item)
        items.append(ticked.group(1) if ticked else item)
    return items


def commit_targets(text: str, heading: str) -> list[str]:
    """Return each item's first token plus every backticked token containing '/' in the section."""
    targets = section_items(text, heading)
    for line in _section_lines(text, heading):
        targets.extend(t for t in TICKED_RE.findall(line) if "/" in t and t not in targets)
    return targets


def excluded_problems(text: str, excluded: tuple[str, ...]) -> list[str]:
    """List excluded prefixes that the brief's '## Allowed paths' section does not name."""
    section = "\n".join(_section_lines(text, "Allowed paths"))
    return [f"{prefix}: not named in Allowed paths" for prefix in excluded if prefix not in section]


def write_target_problems(
    paths: list[str], allowed: set[str], excluded: tuple[str, ...] = ()
) -> list[str]:
    """List paths that are absolute, use backslashes or dot segments, or leave the allowed area."""
    problems = []
    for path in paths:
        segments = path.split("/")
        if path.startswith("/") or re.match(r"^[A-Za-z]:", path):
            problems.append(f"{path}: absolute path")
        elif "\\" in path:
            problems.append(f"{path}: backslash")
        elif "." in segments or ".." in segments:
            problems.append(f"{path}: dot segment")
        elif not any(path.startswith(prefix) for prefix in allowed):
            problems.append(f"{path}: outside allowed paths")
        elif any(path.startswith(prefix) for prefix in excluded):
            problems.append(f"{path}: excluded path")
    return problems


def _brief_text(task: str) -> str:
    matches = sorted(TASKS.glob(f"{task}-*.md"))
    assert len(matches) == 1, f"expected one brief for {task}, found {len(matches)}"
    return matches[0].read_bytes().decode("utf-8")


def _contracts() -> dict[str, str]:
    return contract_blocks(CONTRACTS.read_bytes().decode("utf-8"))


def test_contract_ids_match_headings() -> None:
    text = CONTRACTS.read_bytes().decode("utf-8")
    assert contract_blocks(text)
    assert heading_problems(text) == []


def test_one_brief_per_task() -> None:
    names = [p.name.split("-")[0] for p in TASKS.glob("M*.md")]
    assert sorted(names) == sorted(EXPECTED_ALLOWED)


@pytest.mark.parametrize("task", sorted(EXPECTED_ALLOWED))
def test_brief_excerpts_are_verbatim(task: str) -> None:
    assert excerpt_problems(_brief_text(task), _contracts()) == []


@pytest.mark.parametrize("task", sorted(EXPECTED_ALLOWED))
def test_brief_allowed_paths(task: str) -> None:
    text = _brief_text(task)
    assert set(section_items(text, "Allowed paths")) == EXPECTED_ALLOWED[task]
    assert excluded_problems(text, EXCLUDED.get(task, ())) == []


@pytest.mark.parametrize("task", sorted(EXPECTED_ALLOWED))
def test_brief_files_to_commit(task: str) -> None:
    items = commit_targets(_brief_text(task), "Files to commit")
    assert items
    problems = write_target_problems(items, EXPECTED_ALLOWED[task], EXCLUDED.get(task, ()))
    assert problems == []


def _fenced(block_id: str, body: str) -> str:
    return f"```contract {block_id}\n{body}\n```\n"


def test_changed_character_is_reported() -> None:
    contracts = _contracts()
    body = contracts["raw-input"]
    assert excerpt_problems(_fenced("raw-input", body), contracts) == []
    changed = body[:-1] + ("x" if body[-1] != "x" else "y")
    problems = excerpt_problems(_fenced("raw-input", changed), contracts)
    assert len(problems) == 1 and "raw-input" in problems[0] and "line" in problems[0]


def test_unknown_id_and_missing_block_are_reported() -> None:
    contracts = {"a": "x"}
    assert excerpt_problems(_fenced("b", "x"), contracts)
    assert excerpt_problems("no blocks here\n", contracts)


def test_heading_mismatch_and_duplicate_id() -> None:
    assert heading_problems("### Raw input\n" + _fenced("raw-input", "x")) == []
    assert heading_problems("### Other\n" + _fenced("raw-input", "x"))
    assert heading_problems(_fenced("raw-input", "x"))
    with pytest.raises(ValueError):
        contract_blocks(_fenced("a", "x") + _fenced("a", "y"))


@pytest.mark.parametrize(
    "path",
    [
        "frontend_evil/app.py",
        "frontend/../orchestrator/api.py",
        "/etc/passwd",
        "C:/x/app.py",
        "frontend\\app.py",
    ],
)
def test_flags_paths_outside_allowed(path: str) -> None:
    assert write_target_problems([path], {"frontend/"})


def test_excluded_prefix_is_flagged_and_good_paths_pass() -> None:
    allowed = EXPECTED_ALLOWED["M8"]
    assert write_target_problems(["docs/presentation/core/deck.md"], allowed, EXCLUDED["M8"])
    assert write_target_problems(["frontend/app.py", "frontend/tests/"], {"frontend/"}) == []


@pytest.mark.parametrize(
    "opening",
    ["```contract raw-input ", "```contract Raw-input", "```contract", "```contract-raw-input"],
)
def test_malformed_opening_fence_is_reported(opening: str) -> None:
    brief = _fenced("raw-input", "x") + opening + "\nx\n```\n"
    problems = excerpt_problems(brief, {"raw-input": "x"})
    assert len(problems) == 1 and "malformed" in problems[0]
    assert heading_problems("### Raw input\n" + brief)
    with pytest.raises(ValueError):
        contract_blocks(brief)


def test_underscore_is_kept_in_slug_and_ids() -> None:
    assert heading_problems("### Raw_input\n" + _fenced("raw_input", "x")) == []


@pytest.mark.parametrize("item", ["- orchestrator/", "* `orchestrator/`", "  + `orchestrator/`"])
def test_every_list_item_counts_in_allowed_paths(item: str) -> None:
    text = f"## Allowed paths\n\n- `frontend/`\n{item}\n"
    assert set(section_items(text, "Allowed paths")) != {"frontend/"}
    assert section_items(text, "Allowed paths")[-1] == "orchestrator/"


def test_missing_excluded_sentence_is_reported() -> None:
    bare = "## Allowed paths\n\n- `evaluation/benchmark/`\n- `docs/presentation/`\n"
    assert excluded_problems(bare, EXCLUDED["M8"])
    named = bare + "Not allowed: `docs/presentation/core/`.\n"
    assert excluded_problems(named, EXCLUDED["M8"]) == []


def test_every_slashed_token_in_files_to_commit_is_checked() -> None:
    text = "## Files to commit\n\n- `frontend/a.py` and `orchestrator/b.py`\n"
    targets = commit_targets(text, "Files to commit")
    assert targets == ["frontend/a.py", "orchestrator/b.py"]
    assert write_target_problems(targets, {"frontend/"})


def test_unicode_line_separator_is_drift() -> None:
    contracts = {"a": "x\ny"}
    assert excerpt_problems(_fenced("a", "x\ny"), contracts) == []
    assert excerpt_problems(_fenced("a", "x\u2028y"), contracts)


def test_block_without_closing_fence_raises() -> None:
    brief = _fenced("a", "x") + "```contract b\ny\n"
    with pytest.raises(ValueError):
        excerpt_problems(brief, {"a": "x", "b": "y"})
