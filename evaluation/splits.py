"""Deterministic dev/test split of the cleaned dataset.

Rows that share a context_id or a qgroup_id (directly or through a chain of
such rows) form one component. A component goes to dev or test as a whole,
decided by the hash of its lowest row_id, so related questions never straddle
the two sets.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

DEV_BUCKETS = 2
BUCKETS = 10
SPLIT_COLUMNS = ("row_id", "component", "split")
DEFAULT_CLEANED = Path("data/processed/cleaned_10k_reports.parquet")
DEFAULT_OUT = Path("evaluation/splits.csv")
REQUIRED_COLUMNS = ("row_id", "context_id", "qgroup_id")
EXIT_OK = 0
EXIT_FAILURE = 1


def _find(parent: list[int], i: int) -> int:
    """Return the root of i, compressing the path on the way."""
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:
        parent[i], i = root, parent[i]
    return root


def _components(contexts: list[str], qgroups: list[str]) -> list[int]:
    """Return, per row position, the root position of its component."""
    parent = list(range(len(contexts)))
    first_seen: dict[tuple[str, str], int] = {}
    for i, keys in enumerate(zip(contexts, qgroups, strict=True)):
        for key in (("c", keys[0]), ("q", keys[1])):
            j = first_seen.setdefault(key, i)
            if j != i:
                parent[_find(parent, i)] = _find(parent, j)
    return [_find(parent, i) for i in range(len(parent))]


def _split_of(component: str) -> str:
    digest = hashlib.sha256(component.encode("utf-8")).hexdigest()
    return "dev" if int(digest, 16) % BUCKETS < DEV_BUCKETS else "test"


def build_splits(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Return (row_id, component, split) for every row, sorted by row_id."""
    row_ids = cleaned["row_id"].tolist()
    roots = _components(cleaned["context_id"].tolist(), cleaned["qgroup_id"].tolist())
    lowest: dict[int, str] = {}
    for root, row_id in zip(roots, row_ids, strict=True):
        if root not in lowest or row_id < lowest[root]:
            lowest[root] = row_id
    components = [lowest[root] for root in roots]
    out = pd.DataFrame(
        {
            "row_id": row_ids,
            "component": components,
            "split": [_split_of(c) for c in components],
        },
        columns=list(SPLIT_COLUMNS),
    )
    return out.sort_values("row_id", kind="stable").reset_index(drop=True)


def _read_cleaned(path: Path) -> pd.DataFrame:
    if path.is_dir():
        raise OSError(f"cleaned path is a directory: {path}")
    return pd.read_parquet(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.splits")
    parser.add_argument("--cleaned", type=Path, default=DEFAULT_CLEANED)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    try:
        cleaned = _read_cleaned(args.cleaned)
    except FileNotFoundError:
        print(f"splits: cleaned file not found: {args.cleaned}")
        return EXIT_FAILURE
    except (OSError, ValueError):
        print(f"splits: cannot read cleaned file: {args.cleaned}")
        return EXIT_FAILURE
    missing = [c for c in REQUIRED_COLUMNS if c not in cleaned.columns]
    if missing:
        print(f"splits: cleaned file lacks required columns: {missing}")
        return EXIT_FAILURE
    splits = build_splits(cleaned)
    if args.out.is_dir():
        print(f"splits: output path is a directory: {args.out}")
        return EXIT_FAILURE
    try:
        splits.to_csv(args.out, index=False, encoding="utf-8", lineterminator="\n")
    except PermissionError:
        print(f"splits: permission denied writing {args.out}")
        return EXIT_FAILURE
    except OSError as exc:
        print(f"splits: cannot write {args.out}: {exc}")
        return EXIT_FAILURE
    dev = int((splits["split"] == "dev").sum())
    total = len(splits)
    count = splits["component"].nunique()
    print(
        f"splits: {total} rows, {dev} dev / {total - dev} test in {count} components -> {args.out}"
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
