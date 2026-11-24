"""Named values shared by the synthetic ERP data generator."""

from datetime import date
from pathlib import Path

import numpy as np

SEED = 20231231

EPOCH = date(2022, 1, 1)
LAST_DATE = date(2023, 12, 31)
LAST_DAY = (LAST_DATE - EPOCH).days
EPOCH_WEEKDAY = EPOCH.weekday()
N_MONTHS = 24
SATURDAY = 5
DAYS_PER_MONTH = 30.4375
UNPAID_DAY = 1_000_000

TICKERS_PATH = Path(__file__).resolve().parents[1] / "reference" / "tickers.csv"

# (name, first date, last date) of the evaluation windows, inclusive.
WINDOWS = (
    ("train", date(2022, 1, 1), date(2023, 6, 30)),
    ("val", date(2023, 7, 1), date(2023, 9, 30)),
    ("test", date(2023, 10, 1), date(2023, 12, 31)),
)
MIN_POSITIVES_PER_WINDOW = 30

COMPANY_COLUMNS = ["company_id", "ticker", "company_name", "sector", "size_tier"]
VENDOR_COLUMNS = [
    "vendor_id",
    "vendor_name",
    "country",
    "category",
    "onboarded_on",
    "payment_terms_days",
]
TRANSACTION_COLUMNS = [
    "txn_id",
    "company_id",
    "vendor_id",
    "invoice_number",
    "invoice_date",
    "due_date",
    "paid_on",
    "amount",
    "payment_method",
]
LABEL_COLUMNS = ["txn_id", "anomaly_type"]

SIZE_TIERS = ("S", "M", "L")
SIZE_TIER_WEIGHTS = (0.4, 0.4, 0.2)
SIZE_TIER_MULTIPLIER = {"S": 0.5, "M": 1.0, "L": 2.5}

# (Faker locale, ISO-2 country, share of vendors)
LOCALES = (
    ("en_US", "US", 0.75),
    ("en_GB", "GB", 0.06),
    ("de_DE", "DE", 0.07),
    ("fr_FR", "FR", 0.06),
    ("en_IN", "IN", 0.06),
)

CATEGORIES = (
    "software",
    "hardware",
    "professional_services",
    "logistics",
    "facilities",
    "marketing",
    "raw_materials",
    "travel",
)
CATEGORY_WEIGHTS = (0.18, 0.10, 0.20, 0.12, 0.10, 0.12, 0.10, 0.08)
CATEGORY_MEDIAN_USD = {
    "software": 3000.0,
    "hardware": 6000.0,
    "professional_services": 8000.0,
    "logistics": 2500.0,
    "facilities": 2000.0,
    "marketing": 4000.0,
    "raw_materials": 9000.0,
    "travel": 800.0,
}
PAYMENT_TERMS = (30, 45, 60, 90)
PAYMENT_TERMS_WEIGHTS = (0.60, 0.15, 0.20, 0.05)
PAYMENT_METHODS = ("ach", "wire", "check", "card")
PAYMENT_METHOD_WEIGHTS = (0.50, 0.15, 0.20, 0.15)
INJECTED_TERMS_DAYS = 30

# Vendor population.
N_OLD_VENDORS = 1380
N_NEW_ORDINARY_VENDORS = 120
INJECTED_PER_MONTH = 12
INJECTED_ONBOARD_SPAN_DAYS = 28
INJECTED_TAIL_DAYS = 28
OLD_FIRST_DAY = (date(2015, 1, 1) - EPOCH).days
NEW_ORDINARY_LAST_DAY = (date(2023, 10, 31) - EPOCH).days
N_INVOICE_FORMATS = 3
ALWAYS_LATE_VENDOR_SHARE = 0.05
MAX_INVOICE_SEQ_START = 5000

# Company-vendor relationships.
MIN_VENDORS_PER_COMPANY = 20
MAX_VENDORS_PER_COMPANY = 80
ZIPF_EXPONENT = 0.8
PAIR_KINDS = ("recurring", "subscription", "annual", "irregular")
PAIR_KIND_WEIGHTS = (0.36, 0.14, 0.04, 0.46)
PAIR_AMOUNT_SIGMA = 0.9
ROW_AMOUNT_SIGMA = 0.35
RECURRING_JITTER = 0.02
MIN_AMOUNT_CENTS = 100
OLD_PAIR_START_SPAN_DAYS = 31
NEW_PAIR_START_SPAN_DAYS = 15
MONTHLY_ROWS = 25
SUBSCRIPTION_ROWS = 32
SUBSCRIPTION_GAP_DAYS = (25, 35)
ANNUAL_MONTHS = 12
DAYS_PER_YEAR = 365
IRREGULAR_RATE_MEDIAN = 1.8
IRREGULAR_RATE_SIGMA = 0.8
PERIOD_END_WEIGHT = 1.3
PERIOD_END_SPAN_DAYS = 3
MONTH_STEP_JITTER_DAYS = 2

# Payment behaviour.
PAID_OFFSET_MEAN = -2.0
PAID_OFFSET_SD = 5.0
LATE_LABEL_DAYS = 15
VENDOR_LATE_DAYS = (10, 30)
DISTRESS_START_MONTHS = (0, 2, 5, 7, 10, 13, 16, 18, 20, 21)
DISTRESS_MONTHS = (2, 4)
DISTRESS_LATE_SHARE = 0.6
DISTRESS_LATE_DAYS = (20, 45)
MIN_PAID_AFTER_INVOICE_DAYS = 1

# Anomaly injection.
APPROVAL_THRESHOLD_USD = 10_000
APPROVAL_THRESHOLD_CENTS = APPROVAL_THRESHOLD_USD * 100
SPIKE_RATE = 0.004
SPIKE_MIN_HISTORY = 3
SPIKE_FACTOR = (8.0, 25.0)
DUPLICATE_RATE = 0.003
DUPLICATE_DELAY_DAYS = (1, 30)
SPLIT_RATE = 0.003
SPLIT_SIZE = (3, 5)
SPLIT_SPAN_DAYS = 3
SPLIT_SHARE = (0.90, 0.999)
SPLIT_FIRST_DAY = 60
NEW_VENDOR_FIRST_INVOICE_DAYS = 10
NEW_VENDOR_PAID_AFTER_DAYS = (5, 15)
NEW_VENDOR_EXTRA_SHARE = 0.5
NEW_VENDOR_FOLLOW_UPS = (1, 3)
NEW_VENDOR_FOLLOW_GAP_DAYS = (20, 50)
NEW_VENDOR_FOLLOW_SIGMA = 0.3
WEEKEND_RATE = 0.005
WEEKEND_MAX_SHIFT_DAYS = 6
LARGE_PERCENTILE = 99
TYPE_NAMES = (
    "amount_spike",
    "duplicate_invoice",
    "late_payment",
    "new_vendor_large",
    "split_payment",
    "weekend_payment",
)

MONTH_START_DAYS = np.array(
    [(date(2022 + m // 12, m % 12 + 1, 1) - EPOCH).days for m in range(N_MONTHS)] + [LAST_DAY + 1],
    dtype=np.int64,
)


def iso_dates(days: np.ndarray) -> np.ndarray:
    """Convert day offsets from the first day of the period to ISO date strings."""
    base = np.datetime64(EPOCH.isoformat())
    return (base + days.astype("timedelta64[D]")).astype(str)
