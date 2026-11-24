"""Command line for the reference cleaner.

Usage: python -m data_pipeline.contract.build_reference [--raw PATH] --out-dir DIR
"""

import argparse
import sys
from pathlib import Path

import pyarrow as pa

from data_pipeline.contract.normalize import normalize_text
from data_pipeline.contract.reference import clean, read_raw, write_outputs

__all__ = ["main", "normalize_text"]

DEFAULT_RAW = Path("data/raw/financial_qa_10k.parquet")


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="build_reference", description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _run(raw_path: Path, out_dir: Path) -> str:
    if raw_path.is_dir():
        raise OSError(f"{raw_path} is a directory; pass the raw parquet file")
    if out_dir.is_file():
        raise OSError(f"{out_dir} is a file; pass a directory")
    cleaned, drop_log = clean(read_raw(raw_path))
    write_outputs(cleaned, drop_log, out_dir)
    return f"build_reference: kept {len(cleaned)} rows, dropped {len(drop_log)} rows -> {out_dir}"


def _first_line(error: Exception) -> str:
    lines = str(error).splitlines()
    return lines[0] if lines else type(error).__name__


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    try:
        print(_run(args.raw, args.out_dir))
    except FileNotFoundError:
        print(f"build_reference: raw file {args.raw} not found")
    except PermissionError:
        print("build_reference: permission denied")
    except (pa.ArrowException, OSError, ValueError) as error:
        print(f"build_reference: {_first_line(error)}")
    else:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
