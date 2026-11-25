"""Payment dates: normal behaviour, late vendors and company distress windows."""

import numpy as np
import pandas as pd

from data_pipeline.synth import constants as c


def weekday(days: np.ndarray) -> np.ndarray:
    """Weekday (Monday = 0) of day offsets."""
    return (days + c.EPOCH_WEEKDAY) % 7


def next_business_day(days: np.ndarray) -> np.ndarray:
    """Move Saturdays and Sundays to the following Monday."""
    wd = weekday(days)
    return days + np.where(wd == c.SATURDAY, 2, np.where(wd == c.SATURDAY + 1, 1, 0))


def distress_windows(rng: np.random.Generator, n_companies: int) -> tuple[np.ndarray, np.ndarray]:
    """First and (exclusive) last invoice day of each company's distress window.

    Companies without a window get (-1, -1). Indexed by company_id.
    """
    count = len(c.DISTRESS_START_MONTHS)
    chosen = np.sort(rng.choice(n_companies, size=count, replace=False)) + 1
    starts = rng.permutation(np.array(c.DISTRESS_START_MONTHS))
    length = rng.integers(c.DISTRESS_MONTHS[0], c.DISTRESS_MONTHS[1] + 1, count)
    ends = np.minimum(starts + length, c.N_MONTHS)
    first = np.full(n_companies + 1, -1, dtype=np.int64)
    last = np.full(n_companies + 1, -1, dtype=np.int64)
    first[chosen] = c.MONTH_START_DAYS[starts]
    last[chosen] = c.MONTH_START_DAYS[ends]
    return first, last


def assign_payments(
    rng: np.random.Generator,
    rows: pd.DataFrame,
    vendors: pd.DataFrame,
    n_companies: int,
) -> np.ndarray:
    """Payment day per row; `UNPAID_DAY` when payment falls after the period."""
    n = len(rows)
    inv = rows["invoice_day"].to_numpy()
    due = inv + rows["terms"].to_numpy()
    offset = np.rint(rng.normal(c.PAID_OFFSET_MEAN, c.PAID_OFFSET_SD, n)).astype(np.int64)
    vendor_late = vendors["always_late"].to_numpy()[rows["vendor_id"].to_numpy() - 1]
    lo, hi = c.VENDOR_LATE_DAYS
    offset = np.where(vendor_late, rng.integers(lo, hi + 1, n), offset)
    first, last = distress_windows(rng, n_companies)
    comp = rows["company_id"].to_numpy()
    in_window = (inv >= first[comp]) & (inv < last[comp]) & (first[comp] >= 0)
    distressed = in_window & (rng.random(n) < c.DISTRESS_LATE_SHARE)
    lo, hi = c.DISTRESS_LATE_DAYS
    offset = np.where(distressed, rng.integers(lo, hi + 1, n), offset)
    paid = np.maximum(due + offset, inv + c.MIN_PAID_AFTER_INVOICE_DAYS)
    return next_business_day(paid)


def new_vendor_payments(rng: np.random.Generator, invoice_day: np.ndarray) -> np.ndarray:
    """Early payment, well before the 30-day due date, on a business day."""
    lo, hi = c.NEW_VENDOR_PAID_AFTER_DAYS
    return next_business_day(invoice_day + rng.integers(lo, hi + 1, len(invoice_day)))


def cap_unpaid(paid: np.ndarray) -> np.ndarray:
    """Payments after the last day of the period are unpaid."""
    return np.where(paid > c.LAST_DAY, c.UNPAID_DAY, paid)
