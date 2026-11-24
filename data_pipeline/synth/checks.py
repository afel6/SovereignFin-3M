"""Quality checks of generated data against the label requirements."""

import pandas as pd

from data_pipeline.synth import constants as c
from data_pipeline.synth.errors import GenerationError


def window_counts(transactions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Positives per (type, window, basis); basis is paid_on or invoice_date."""
    joined = labels.merge(
        transactions[["txn_id", "invoice_date", "paid_on"]], on="txn_id", how="left"
    )
    rows = []
    for name, first, last in c.WINDOWS:
        lo, hi = first.isoformat(), last.isoformat()
        for basis in ("paid_on", "invoice_date"):
            col = joined[basis]
            inside = joined[(col != "") & (col >= lo) & (col <= hi)]
            counts = inside.groupby("anomaly_type").size()
            for kind in c.TYPE_NAMES:
                rows.append((kind, name, basis, int(counts.get(kind, 0))))
    return pd.DataFrame(rows, columns=["anomaly_type", "window", "basis", "positives"])


def check_label_counts(transactions: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Raise when any type has too few positives in any window."""
    counts = window_counts(transactions, labels)
    low = counts[counts["positives"] < c.MIN_POSITIVES_PER_WINDOW]
    if len(low):
        first = low.iloc[0]
        raise GenerationError(
            f"too few {first['anomaly_type']} positives in the {first['window']} "
            f"window by {first['basis']} ({first['positives']})"
        )
