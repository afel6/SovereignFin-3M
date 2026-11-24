"""Writing the generated tables, the labels and the manifest."""

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from data_pipeline.synth.constants import SEED
from data_pipeline.synth.generator import SynthData

MANIFEST_NAME = "manifest.json"


def _write_csv(frame: pd.DataFrame, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"rows": len(frame), "sha256": digest}


def write_outputs(data: SynthData, out_dir: Path, snapshot: int | None = None) -> dict:
    """Write the CSVs and manifest.json under `out_dir`; return the manifest."""
    out_dir = Path(out_dir)
    tables = {
        "companies.csv": data.companies,
        "vendors.csv": data.vendors,
        "transactions.csv": data.transactions,
        "labels/transaction_labels.csv": data.labels,
    }
    files = {name: _write_csv(frame, out_dir / name) for name, frame in tables.items()}
    manifest = {
        "seed": SEED,
        "data_snapshot": int(time.time()) if snapshot is None else snapshot,
        "files": files,
    }
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (out_dir / MANIFEST_NAME).write_bytes(text.encode("utf-8"))
    return manifest
