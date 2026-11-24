"""Companies and vendors of the synthetic ERP."""

import numpy as np
import pandas as pd
from faker import Faker

from data_pipeline.synth import constants as c


def build_companies(rng: np.random.Generator) -> pd.DataFrame:
    """One row per reference ticker, ids in sorted-ticker order, with a size tier."""
    ref = pd.read_csv(c.TICKERS_PATH, dtype=str, keep_default_na=False, encoding="utf-8")
    ref = ref.sort_values("ticker", kind="stable").reset_index(drop=True)
    tiers = rng.choice(c.SIZE_TIERS, size=len(ref), p=c.SIZE_TIER_WEIGHTS)
    out = pd.DataFrame(
        {
            "company_id": np.arange(1, len(ref) + 1, dtype=np.int64),
            "ticker": ref["ticker"],
            "company_name": ref["company_name"],
            "sector": ref["sector"],
            "size_tier": tiers.tolist(),
        }
    )
    return out[c.COMPANY_COLUMNS]


def _unique_names(names_raw: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for name in names_raw:
        candidate, n = name, 1
        while candidate in seen:
            n += 1
            candidate = f"{name} ({n})"
        seen.add(candidate)
        out.append(candidate)
    return out


def _onboard_span(month: np.ndarray) -> np.ndarray:
    """Days of the month an injected vendor may be onboarded, so its first
    invoice and payment still fall inside the period."""
    room = c.LAST_DAY - c.INJECTED_TAIL_DAYS - c.MONTH_START_DAYS[month] + 1
    return np.clip(room, 1, c.INJECTED_ONBOARD_SPAN_DAYS)


def _onboarding(rng: np.random.Generator, n_companies: int):
    n_inj = c.INJECTED_PER_MONTH * c.N_MONTHS
    month = np.arange(n_inj) % c.N_MONTHS
    onboard = np.concatenate(
        [
            rng.integers(c.OLD_FIRST_DAY, 0, c.N_OLD_VENDORS),
            rng.integers(0, c.NEW_ORDINARY_LAST_DAY + 1, c.N_NEW_ORDINARY_VENDORS),
            c.MONTH_START_DAYS[month] + rng.integers(0, _onboard_span(month), n_inj),
        ]
    )
    n_regular = c.N_OLD_VENDORS + c.N_NEW_ORDINARY_VENDORS
    company = np.concatenate(
        [np.zeros(n_regular, dtype=np.int64), rng.integers(1, n_companies + 1, n_inj)]
    )
    return onboard.astype(np.int64), company.astype(np.int64)


def build_vendors(rng: np.random.Generator, seed: int, n_companies: int) -> pd.DataFrame:
    """Vendors with final ids, plus the generator-only columns onboarded_day,
    injected_company, invoice_format and always_late."""
    onboard, injected = _onboarding(rng, n_companies)
    n = len(onboard)
    weights = [w for _, _, w in c.LOCALES]
    loc = rng.choice(len(c.LOCALES), size=n, p=weights)
    fakers = []
    for i, (locale, _, _) in enumerate(c.LOCALES):
        faker = Faker(locale)
        faker.seed_instance(seed + i + 1)
        fakers.append(faker)
    names = _unique_names([fakers[k].company() for k in loc])
    terms = rng.choice(c.PAYMENT_TERMS, size=n, p=c.PAYMENT_TERMS_WEIGHTS)
    terms = np.where(injected > 0, c.INJECTED_TERMS_DAYS, terms)
    always_late = (rng.random(n) < c.ALWAYS_LATE_VENDOR_SHARE) & (injected == 0)
    frame = pd.DataFrame(
        {
            "vendor_name": names,
            "country": [c.LOCALES[k][1] for k in loc],
            "category": rng.choice(c.CATEGORIES, size=n, p=c.CATEGORY_WEIGHTS),
            "onboarded_on": c.iso_dates(onboard),
            "payment_terms_days": terms.astype(np.int64),
            "onboarded_day": onboard,
            "injected_company": injected,
            "invoice_format": rng.integers(0, c.N_INVOICE_FORMATS, n),
            "always_late": always_late,
        }
    )
    frame = frame.sort_values(["onboarded_on", "vendor_name"], kind="stable")
    frame = frame.reset_index(drop=True)
    frame.insert(0, "vendor_id", np.arange(1, n + 1, dtype=np.int64))
    return frame


def public_vendors(vendors: pd.DataFrame) -> pd.DataFrame:
    """The vendor table as written to vendors.csv."""
    return vendors[c.VENDOR_COLUMNS].copy()
