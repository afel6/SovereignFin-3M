"""Final pass: transaction ids, invoice numbers, text formatting and labels."""

import numpy as np
import pandas as pd

from data_pipeline.synth import constants as c

_YEAR_FORMAT = 1
_DAYS_IN_FIRST_YEAR = 365
_SCOPE_SLOTS = 4
_PLAIN_SCOPE = 2


def _cumcount(keys: np.ndarray) -> np.ndarray:
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    first = np.searchsorted(sorted_keys, sorted_keys, side="left")
    out = np.empty(len(keys), dtype=np.int64)
    out[order] = np.arange(len(keys)) - first
    return out


def invoice_numbers(
    rng: np.random.Generator,
    vendor_id: np.ndarray,
    invoice_day: np.ndarray,
    vendors: pd.DataFrame,
) -> list[str]:
    """Per-vendor running numbers in the given row order, in the vendor format."""
    fmt = vendors["invoice_format"].to_numpy()[vendor_id - 1]
    year_idx = (invoice_day >= _DAYS_IN_FIRST_YEAR).astype(np.int64)
    scope = np.where(fmt == _YEAR_FORMAT, year_idx, _PLAIN_SCOPE)
    seq = _cumcount(vendor_id * _SCOPE_SLOTS + scope)
    start = rng.integers(1, c.MAX_INVOICE_SEQ_START, len(vendors))
    seq = seq + start[vendor_id - 1]
    year = c.EPOCH.year + year_idx
    out = []
    for f, s, y in zip(fmt.tolist(), seq.tolist(), year.tolist(), strict=True):
        if f == 0:
            out.append(f"INV-{s:06d}")
        elif f == _YEAR_FORMAT:
            out.append(f"{y}-{s:05d}")
        else:
            out.append(f"A{s:05d}")
    return out


def format_cents(cents: np.ndarray) -> list[str]:
    """Amounts as text with two decimals, from integer cents."""
    return [f"{v // 100}.{v % 100:02d}" for v in cents.tolist()]


def finalize(
    rng: np.random.Generator,
    rows: pd.DataFrame,
    paid: np.ndarray,
    weekend: np.ndarray,
    vendors: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sort by invoice date, assign ids and numbers, and build the labels."""
    tiebreak = rng.random(len(rows))
    order = np.lexsort((tiebreak, rows["invoice_day"].to_numpy()))
    rows = rows.iloc[order].reset_index(drop=True)
    paid, weekend = paid[order], weekend[order]
    day = rows["invoice_day"].to_numpy()
    due = day + rows["terms"].to_numpy()
    has_paid = paid != c.UNPAID_DAY
    txn_id = np.arange(1, len(rows) + 1, dtype=np.int64)
    vendor_id = rows["vendor_id"].to_numpy()
    tx = pd.DataFrame(
        {
            "txn_id": txn_id,
            "company_id": rows["company_id"].to_numpy(),
            "vendor_id": vendor_id,
            "invoice_number": invoice_numbers(rng, vendor_id, day, vendors),
            "invoice_date": c.iso_dates(day),
            "due_date": c.iso_dates(due),
            "paid_on": np.where(has_paid, c.iso_dates(paid), ""),
            "amount": format_cents(rows["amount_cents"].to_numpy()),
            "payment_method": np.array(c.PAYMENT_METHODS)[rows["method"].to_numpy()],
        }
    )[c.TRANSACTION_COLUMNS]
    masks = {
        "amount_spike": rows["spike"].to_numpy(),
        "duplicate_invoice": rows["duplicate"].to_numpy(),
        "late_payment": has_paid & (paid - due > c.LATE_LABEL_DAYS),
        "new_vendor_large": rows["new_vendor"].to_numpy(),
        "split_payment": rows["split"].to_numpy(),
        "weekend_payment": weekend,
    }
    return tx, build_labels(txn_id, masks)


def build_labels(txn_id: np.ndarray, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    """One row per (transaction, type), sorted by transaction id then type."""
    parts = [
        pd.DataFrame({"txn_id": txn_id[mask], "anomaly_type": name}) for name, mask in masks.items()
    ]
    labels = pd.concat(parts, ignore_index=True)
    labels = labels.sort_values(c.LABEL_COLUMNS, kind="stable")
    return labels.reset_index(drop=True)[c.LABEL_COLUMNS]
