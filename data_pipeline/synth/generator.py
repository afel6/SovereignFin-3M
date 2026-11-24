"""Deterministic synthetic ERP data: companies, vendors, transactions, labels."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_pipeline.synth import anomalies as an
from data_pipeline.synth import payments, schedule
from data_pipeline.synth.assemble import finalize
from data_pipeline.synth.checks import check_label_counts
from data_pipeline.synth.constants import SEED
from data_pipeline.synth.entities import (
    build_companies,
    build_vendors,
    public_vendors,
)


@dataclass(frozen=True)
class SynthData:
    """The four generated tables."""

    companies: pd.DataFrame
    vendors: pd.DataFrame
    transactions: pd.DataFrame
    labels: pd.DataFrame


def _build_rows(rng, companies, vendors, pairs) -> pd.DataFrame:
    rows = schedule.build_base_rows(rng, pairs)
    n_base = len(rows)
    rows = an.inject_spikes(rng, rows)
    rows = an.inject_duplicates(rng, rows)
    rows = an.inject_splits(rng, rows, pairs, n_base)
    return an.inject_new_vendor_large(rng, rows, vendors, len(companies))


def generate(seed: int = SEED) -> SynthData:
    """Generate all tables from `seed`; the same seed gives identical tables."""
    rng = np.random.default_rng(seed)
    companies = build_companies(rng)
    vendors = build_vendors(rng, seed, len(companies))
    pairs = schedule.describe_pairs(
        rng, schedule.choose_pairs(rng, companies, vendors), companies, vendors
    )
    rows = _build_rows(rng, companies, vendors, pairs)
    paid = payments.assign_payments(rng, rows, vendors, len(companies))
    paid = an.apply_new_vendor_payments(rng, rows, paid)
    paid, weekend = an.inject_weekend(rng, paid, rows["new_vendor"].to_numpy())
    paid = payments.cap_unpaid(paid)
    transactions, labels = finalize(rng, rows, paid, weekend, vendors)
    check_label_counts(transactions, labels)
    return SynthData(companies, public_vendors(vendors), transactions, labels)
