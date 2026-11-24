"""Injection of labelled anomalies into the ordinary invoice rows."""

import numpy as np
import pandas as pd

from data_pipeline.synth import constants as c
from data_pipeline.synth.payments import new_vendor_payments, weekday

FLAGS = ("spike", "duplicate", "split", "new_vendor")


def _median_int(values: np.ndarray) -> int:
    ordered = np.sort(values)
    m = len(ordered)
    if m % 2:
        return int(ordered[m // 2])
    return int((ordered[m // 2 - 1] + ordered[m // 2]) // 2)


def inject_spikes(rng: np.random.Generator, rows: pd.DataFrame) -> pd.DataFrame:
    """Raise some amounts to 8-25 times the pair's median of earlier invoices."""
    pair = rows["pair_id"].to_numpy()
    cents = rows["amount_cents"].to_numpy()
    order = np.lexsort((rows["invoice_day"].to_numpy(), pair))
    sorted_pair = pair[order]
    first = np.searchsorted(sorted_pair, sorted_pair, side="left")
    eligible = np.flatnonzero(np.arange(len(rows)) - first >= c.SPIKE_MIN_HISTORY)
    k = int(c.SPIKE_RATE * len(rows))
    chosen = np.sort(rng.choice(eligible, size=k, replace=False))
    factor = np.rint(rng.uniform(*c.SPIKE_FACTOR, k) * 1000).astype(np.int64)
    new_cents = cents.copy()
    spike = np.zeros(len(rows), dtype=bool)
    for pos, milli in zip(chosen, factor, strict=True):
        median = _median_int(cents[order[first[pos] : pos]])
        new_cents[order[pos]] = median * int(milli) // 1000
        spike[order[pos]] = True
    return rows.assign(amount_cents=new_cents, spike=spike)


def inject_duplicates(rng: np.random.Generator, rows: pd.DataFrame) -> pd.DataFrame:
    """Append copies of real invoices dated 1-30 days later."""
    candidates = np.flatnonzero(~rows["spike"].to_numpy())
    k = int(c.DUPLICATE_RATE * len(rows))
    chosen = np.sort(rng.choice(candidates, size=k, replace=False))
    delay = rng.integers(c.DUPLICATE_DELAY_DAYS[0], c.DUPLICATE_DELAY_DAYS[1] + 1, k)
    copies = rows.iloc[chosen].reset_index(drop=True)
    copies = copies.assign(invoice_day=copies["invoice_day"].to_numpy() + delay)
    copies = copies[copies["invoice_day"] <= c.LAST_DAY]
    copies = copies.assign(spike=False, duplicate=True)
    return pd.concat([rows, copies], ignore_index=True)


def inject_splits(
    rng: np.random.Generator, rows: pd.DataFrame, pairs: pd.DataFrame, n_base: int
) -> pd.DataFrame:
    """Append events of 3-5 invoices just under the approval threshold."""
    lo, hi = c.SPLIT_SIZE
    events = int(c.SPLIT_RATE * n_base / ((lo + hi) / 2))
    pool = np.flatnonzero(pairs["is_old"].to_numpy())
    pick = pairs.iloc[rng.choice(pool, size=events)]
    size = rng.integers(lo, hi + 1, events)
    day0 = rng.integers(c.SPLIT_FIRST_DAY, c.LAST_DAY - c.SPLIT_SPAN_DAYS + 1, events)
    rep = np.repeat(np.arange(events), size)
    day = day0[rep] + rng.integers(0, c.SPLIT_SPAN_DAYS + 1, len(rep))
    share = rng.uniform(*c.SPLIT_SHARE, len(rep))
    cents = np.rint(share * c.APPROVAL_THRESHOLD_CENTS).astype(np.int64)
    return pd.concat([rows, _new_rows(pick.iloc[rep], day, cents, split=True)], ignore_index=True)


def _new_rows(src: pd.DataFrame, day, cents, **flags) -> pd.DataFrame:
    n = len(src)
    out = pd.DataFrame(
        {
            "company_id": src["company_id"].to_numpy(),
            "vendor_id": src["vendor_id"].to_numpy(),
            "invoice_day": np.asarray(day, dtype=np.int64),
            "terms": src["terms"].to_numpy(),
            "amount_cents": np.asarray(cents, dtype=np.int64),
            "method": src["method"].to_numpy(),
            "pair_id": np.full(n, -1, dtype=np.int64),
        }
    )
    for name in FLAGS:
        out[name] = False
    return out.assign(**flags)


def large_amount_cents(rows: pd.DataFrame, n_companies: int) -> tuple[np.ndarray, np.ndarray]:
    """Per company: 99th percentile and median of all its invoices so far, in integer cents."""
    p99 = np.zeros(n_companies + 1, dtype=np.int64)
    p50 = np.zeros(n_companies + 1, dtype=np.int64)
    for cid, grp in rows.groupby("company_id", sort=True):
        arr = np.sort(grp["amount_cents"].to_numpy())
        n = len(arr)
        p99[cid] = arr[(c.LARGE_PERCENTILE * n + 99) // 100 - 1]
        p50[cid] = arr[n // 2]
    return p99, p50


def inject_new_vendor_large(
    rng: np.random.Generator,
    rows: pd.DataFrame,
    vendors: pd.DataFrame,
    n_companies: int,
) -> pd.DataFrame:
    """Append a large first invoice, plus ordinary follow-ups, for injected vendors."""
    p99, p50 = large_amount_cents(rows, n_companies)
    inj = vendors[vendors["injected_company"] > 0]
    n = len(inj)
    comp = inj["injected_company"].to_numpy()
    first_day = inj["onboarded_day"].to_numpy() + rng.integers(
        0, c.NEW_VENDOR_FIRST_INVOICE_DAYS + 1, n
    )
    extra = np.rint(p99[comp] * rng.uniform(0, c.NEW_VENDOR_EXTRA_SHARE, n))
    method = rng.choice(len(c.PAYMENT_METHODS), size=n, p=c.PAYMENT_METHOD_WEIGHTS)
    src = pd.DataFrame(
        {
            "company_id": comp,
            "vendor_id": inj["vendor_id"].to_numpy(),
            "terms": inj["payment_terms_days"].to_numpy(),
            "method": method,
        }
    )
    big = _new_rows(src, first_day, p99[comp] + extra.astype(np.int64), new_vendor=True)
    follow = _follow_ups(rng, src, first_day, p50[comp], p99[comp])
    return pd.concat([rows, big, follow], ignore_index=True)


def _follow_ups(rng, src, first_day, p50, p99) -> pd.DataFrame:
    lo, hi = c.NEW_VENDOR_FOLLOW_UPS
    n = len(src)
    count = rng.integers(lo, hi + 1, n)
    gaps = rng.integers(*c.NEW_VENDOR_FOLLOW_GAP_DAYS, (n, hi))
    day = first_day[:, None] + np.cumsum(gaps, axis=1)
    noise = np.exp(rng.normal(0.0, c.NEW_VENDOR_FOLLOW_SIGMA, (n, hi)))
    cents = np.rint(p50[:, None] * noise).astype(np.int64)
    ceiling = np.maximum(p99 - 1, c.MIN_AMOUNT_CENTS)[:, None]
    cents = np.clip(cents, c.MIN_AMOUNT_CENTS, ceiling)
    keep = (np.arange(hi)[None, :] < count[:, None]) & (day <= c.LAST_DAY)
    rep = np.broadcast_to(np.arange(n)[:, None], keep.shape)[keep]
    return _new_rows(src.iloc[rep], day[keep], cents[keep])


def inject_weekend(
    rng: np.random.Generator, paid: np.ndarray, protected: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Move some payments to a Saturday or Sunday; returns (paid, flag)."""
    ok = (paid <= c.LAST_DAY - c.WEEKEND_MAX_SHIFT_DAYS) & ~protected
    k = int(c.WEEKEND_RATE * len(paid))
    chosen = np.sort(rng.choice(np.flatnonzero(ok), size=k, replace=False))
    shift = (c.SATURDAY - weekday(paid[chosen])) + rng.integers(0, 2, k)
    new_paid = paid.copy()
    new_paid[chosen] = paid[chosen] + shift
    flag = np.zeros(len(paid), dtype=bool)
    flag[chosen] = True
    return new_paid, flag


def apply_new_vendor_payments(
    rng: np.random.Generator, rows: pd.DataFrame, paid: np.ndarray
) -> np.ndarray:
    """New-vendor first invoices are paid before their due date."""
    mask = rows["new_vendor"].to_numpy()
    out = paid.copy()
    out[mask] = new_vendor_payments(rng, rows["invoice_day"].to_numpy()[mask])
    return out
