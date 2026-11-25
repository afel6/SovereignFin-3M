"""Command line: python -m data_pipeline.synth --out-dir data"""

import argparse
import sys
from pathlib import Path

from data_pipeline.synth.errors import GenerationError
from data_pipeline.synth.generator import generate
from data_pipeline.synth.output import write_outputs


def _fail(message: str) -> int:
    print(f"synth: {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Generate the data and write it; one line of output, exit code 0 or 1."""
    parser = argparse.ArgumentParser(prog="python -m data_pipeline.synth")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    out_dir: Path = args.out_dir
    try:
        if out_dir.is_file():
            raise NotADirectoryError(out_dir.name)
        data = generate()
        write_outputs(data, out_dir)
    except NotADirectoryError:
        return _fail("the output path is a file, not a folder")
    except PermissionError:
        return _fail("no permission to write to the output folder")
    except GenerationError as exc:
        return _fail(f"data generation failed: {exc}")
    except OSError as exc:
        return _fail(f"could not write the output ({exc.strerror or 'I/O error'})")
    print(
        f"synth: wrote {len(data.companies)} companies, {len(data.vendors)} vendors, "
        f"{len(data.transactions)} transactions, {len(data.labels)} labels to {out_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
