"""Ordinary invoices: company-vendor relationships and their invoice streams."""

import numpy as np
import pandas as pd

from data_pipeline.synth import constants as c

_Triple = tuple[np.ndarray, np.ndarray, np.ndarray]


def day_weights_cdf() -> np.ndarray:
    """Cumulative invoice volume by day; month ends and quarter ends weigh more."""
    days = np.arange(c.LAST_DAY + 1)
    dates = np.datetime64(c.EPOCH.isoformat()) + days.astype("timedelta64[D]")
    months = dates.astype("datetime64[M]")
    shifted = (dates + np.timedelta64(c.PERIOD_END_SPAN_DAYS, "D")).astype("datetime64[M]")
    month_end = shifted != months
    quarter_end = month_end & ((months.astype(np.int64) % 12 + 1) % 3 == 0)
    weight = np.where(month_end, c.PERIOD_END_WEIGHT, 1.0) * np.where(
        quarter_end, c.PERIOD_END_WEIGHT, 1.0
    )
    return np.concatenate([[0.0], np.cumsum(weight)])


def choose_pairs(
    rng: np.random.Generator,
    companies: pd.DataFrame,
    vendors: pd.DataFrame,
) -> pd.DataFrame:
    """Company-vendor pairs: Zipf-like vendor popularity, every vendor used."""
    n_companies = len(companies)
    eligible = np.flatnonzero(vendors["injected_company"].to_numpy() == 0)
    weight = 1.0 / (rng.permutation(len(eligible)) + 1.0) ** c.ZIPF_EXPONENT
    weight = weight / weight.sum()
    comp_parts, vend_parts = [], []
    for company_id in range(1, n_companies + 1):
        k = rng.integers(c.MIN_VENDORS_PER_COMPANY, c.MAX_VENDORS_PER_COMPANY + 1)
        pick = rng.choice(len(eligible), size=k, replace=False, p=weight)
        comp_parts.append(np.full(k, company_id, dtype=np.int64))
        vend_parts.append(eligible[pick] + 1)
    vendor_ids = np.concatenate(vend_parts)
    missing = np.setdiff1d(eligible + 1, vendor_ids)
    comp_parts.append(rng.integers(1, n_companies + 1, len(missing)))
    vend_parts.append(missing)
    return pd.DataFrame(
        {
            "company_id": np.concatenate(comp_parts).astype(np.int64),
            "vendor_id": np.concatenate(vend_parts).astype(np.int64),
        }
    )


def describe_pairs(
    rng: np.random.Generator,
    pairs: pd.DataFrame,
    companies: pd.DataFrame,
    vendors: pd.DataFrame,
) -> pd.DataFrame:
    """Add start day, stream kind, base amount, method and rate to each pair."""
    n = len(pairs)
    vend = vendors.iloc[pairs["vendor_id"].to_numpy() - 1]
    onboarded = vend["onboarded_day"].to_numpy()
    is_old = onboarded < 0
    start = np.where(
        is_old,
        rng.integers(0, c.OLD_PAIR_START_SPAN_DAYS, n),
        onboarded + rng.integers(0, c.NEW_PAIR_START_SPAN_DAYS, n),
    )
    tier = companies["size_tier"].to_numpy()[pairs["company_id"].to_numpy() - 1]
    median = vend["category"].map(c.CATEGORY_MEDIAN_USD).to_numpy()
    mult = pd.Series(tier).map(c.SIZE_TIER_MULTIPLIER).to_numpy()
    usd = median * mult * np.exp(rng.normal(0.0, c.PAIR_AMOUNT_SIGMA, n))
    cents = np.maximum(np.rint(usd * 100).astype(np.int64), c.MIN_AMOUNT_CENTS)
    return pairs.assign(
        start=start.astype(np.int64),
        is_old=is_old,
        kind=rng.choice(len(c.PAIR_KINDS), size=n, p=c.PAIR_KIND_WEIGHTS),
        base_cents=cents,
        method=rng.choice(len(c.PAYMENT_METHODS), size=n, p=c.PAYMENT_METHOD_WEIGHTS),
        rate=c.IRREGULAR_RATE_MEDIAN * np.exp(rng.normal(0.0, c.IRREGULAR_RATE_SIGMA, n)),
        terms=vend["payment_terms_days"].to_numpy(),
    )


def _masked(idx: np.ndarray, day: np.ndarray, cents: np.ndarray) -> _Triple:
    ok = day <= c.LAST_DAY
    return (
        np.broadcast_to(idx[:, None], day.shape)[ok],
        day[ok],
        np.broadcast_to(cents, day.shape)[ok],
    )


def _jittered(rng: np.random.Generator, base: np.ndarray, shape) -> np.ndarray:
    factor = 1.0 + rng.uniform(-c.RECURRING_JITTER, c.RECURRING_JITTER, shape)
    return np.rint(base[:, None] * factor).astype(np.int64)


def _recurring(rng: np.random.Generator, idx, start, base) -> _Triple:
    shape = (len(idx), c.MONTHLY_ROWS)
    step = np.rint(np.arange(c.MONTHLY_ROWS) * c.DAYS_PER_MONTH).astype(np.int64)
    jit = c.MONTH_STEP_JITTER_DAYS
    day = start[:, None] + step + rng.integers(-jit, jit + 1, shape)
    return _masked(idx, np.maximum(day, start[:, None]), _jittered(rng, base, shape))


def _subscription(rng: np.random.Generator, idx, start, base) -> _Triple:
    lo, hi = c.SUBSCRIPTION_GAP_DAYS
    gaps = rng.integers(lo, hi + 1, (len(idx), c.SUBSCRIPTION_ROWS))
    gaps[:, 0] = 0
    return _masked(idx, start[:, None] + np.cumsum(gaps, axis=1), base[:, None])


def _annual(rng: np.random.Generator, idx, start, base) -> _Triple:
    shape = (len(idx), 2)
    jit = c.MONTH_STEP_JITTER_DAYS + 1
    day = start[:, None] + np.arange(2) * c.DAYS_PER_YEAR
    day = np.maximum(day + rng.integers(-jit, jit + 1, shape), start[:, None])
    return _masked(idx, day, _jittered(rng, base * c.ANNUAL_MONTHS, shape))


def _irregular(rng: np.random.Generator, idx, start, base, rate, is_new) -> _Triple:
    cdf = day_weights_cdf()
    count = rng.poisson(rate * (c.LAST_DAY + 1 - start) / c.DAYS_PER_MONTH) + 1
    rep = np.repeat(np.arange(len(idx)), count)
    u = cdf[start[rep]] + rng.random(len(rep)) * (cdf[-1] - cdf[start[rep]])
    day = np.clip(np.searchsorted(cdf, u, side="right") - 1, start[rep], c.LAST_DAY)
    first = np.cumsum(count) - count
    day[first[is_new]] = start[is_new]
    noise = np.exp(rng.normal(0.0, c.ROW_AMOUNT_SIGMA, len(rep)))
    cents = np.rint(base[rep] * noise).astype(np.int64)
    return idx[rep], day.astype(np.int64), cents


def build_base_rows(rng: np.random.Generator, pairs: pd.DataFrame) -> pd.DataFrame:
    """Invoice rows of every pair; `pair_id` is the pair's position in `pairs`."""
    kind = pairs["kind"].to_numpy()
    parts = []
    for code, build in enumerate((_recurring, _subscription, _annual)):
        sel = np.flatnonzero(kind == code)
        sub = pairs.iloc[sel]
        parts.append(build(rng, sel, sub["start"].to_numpy(), sub["base_cents"].to_numpy()))
    sel = np.flatnonzero(kind == len(c.PAIR_KINDS) - 1)
    sub = pairs.iloc[sel]
    parts.append(
        _irregular(
            rng,
            sel,
            sub["start"].to_numpy(),
            sub["base_cents"].to_numpy(),
            sub["rate"].to_numpy(),
            ~sub["is_old"].to_numpy(),
        )
    )
    pair_id = np.concatenate([p[0] for p in parts]).astype(np.int64)
    day = np.concatenate([p[1] for p in parts]).astype(np.int64)
    cents = np.maximum(np.concatenate([p[2] for p in parts]), c.MIN_AMOUNT_CENTS)
    return pd.DataFrame(
        {
            "company_id": pairs["company_id"].to_numpy()[pair_id],
            "vendor_id": pairs["vendor_id"].to_numpy()[pair_id],
            "invoice_day": day,
            "terms": pairs["terms"].to_numpy()[pair_id],
            "amount_cents": cents.astype(np.int64),
            "method": pairs["method"].to_numpy()[pair_id],
            "pair_id": pair_id,
            "spike": False,
            "duplicate": False,
            "split": False,
            "new_vendor": False,
        }
    )
