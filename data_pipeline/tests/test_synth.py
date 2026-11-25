"""Tests of the synthetic ERP data generator: files, constraints, labels, windows, leaks, CLI."""

import csv
import hashlib
import io
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

from data_pipeline.synth import __main__ as cli
from data_pipeline.synth import constants as c
from data_pipeline.synth.checks import check_label_counts, window_counts
from data_pipeline.synth.errors import GenerationError
from data_pipeline.synth.generator import generate
from data_pipeline.synth.output import write_outputs

EXPECTED_SHA256 = {
    "companies.csv": "104cdccb37f464dd456555e71ab20346d7ee4dd4dff9cdffe561e5c18cbe229c",
    "vendors.csv": "8c155e73dc12f5c1fe86a1a58c07259ad70ca9a32dcd5b883a8e0d1b0aac79a8",
    "transactions.csv": "151a9d9766f64fa38dff825a80ec2357cffb2c402a755cac1dbeda65c15fdaa5",
    "labels/transaction_labels.csv": (
        "fb91cda8e4fea87b9c3823c5c8c8bdb5049603ecd29cc8263d226ebf4fa106dc"
    ),
}
HEADERS = {
    "companies.csv": "company_id,ticker,company_name,sector,size_tier",
    "vendors.csv": "vendor_id,vendor_name,country,category,onboarded_on,payment_terms_days",
    "transactions.csv": (
        "txn_id,company_id,vendor_id,invoice_number,invoice_date,due_date,paid_on,"
        "amount,payment_method"
    ),
    "labels/transaction_labels.csv": "txn_id,anomaly_type",
}
SIX_TYPES = {
    "amount_spike",
    "duplicate_invoice",
    "late_payment",
    "new_vendor_large",
    "split_payment",
    "weekend_payment",
}
CATEGORIES = {
    "software",
    "hardware",
    "professional_services",
    "logistics",
    "facilities",
    "marketing",
    "raw_materials",
    "travel",
}
MAX_LEAK_RATIO = 1.5
NON_AMOUNT_TYPES = {"late_payment", "weekend_payment"}
LATE_DAYS = 15
CV_FOLDS = 5
LEAK_MIN_LEAF = 100
LEAK_MAX_DEPTH = 10
APPROVAL_CENTS = 10_000 * 100
SPLIT_LOW_CENTS = 9000 * 100
SPLIT_HIGH_CENTS = 9990 * 100
SPIKE_MIN_FACTOR = 8
FIRST_DAY = "2022-01-01"
LAST_DAY = "2023-12-31"


def window_of(day: str) -> str | None:
    """Name of the evaluation window holding an ISO date, None outside all windows."""
    if day < "2022-01-01" or day > LAST_DAY:
        return None
    if day < "2023-07-01":
        return "train"
    return "val" if day <= "2023-09-30" else "test"


def cents(amounts: pd.Series) -> pd.Series:
    return amounts.str.replace(".", "", regex=False).astype("int64")


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """One generation written to disk, with the files read back as text and as typed frames."""
    out_dir = tmp_path_factory.mktemp("synth_out")
    data = generate()
    manifest = write_outputs(data, out_dir)
    raw = {
        name: pd.read_csv(out_dir / name, dtype=str, keep_default_na=False, encoding="utf-8")
        for name in HEADERS
    }
    tx = raw["transactions.csv"].copy()
    for col in ("txn_id", "company_id", "vendor_id"):
        tx[col] = tx[col].astype("int64")
    tx["cents"] = cents(tx["amount"])
    for col in ("invoice_date", "due_date", "paid_on"):
        tx[col] = pd.to_datetime(tx[col].replace("", pd.NA))
    labels = raw["labels/transaction_labels.csv"].copy()
    labels["txn_id"] = labels["txn_id"].astype("int64")
    return SimpleNamespace(
        data=data,
        out_dir=out_dir,
        manifest=manifest,
        raw=raw,
        tx=tx,
        labels=labels,
        companies=raw["companies.csv"],
        vendors=raw["vendors.csv"],
    )


def labelled(synth, kind: str) -> pd.DataFrame:
    ids = synth.labels.loc[synth.labels["anomaly_type"] == kind, "txn_id"]
    return synth.tx[synth.tx["txn_id"].isin(ids)]


# 1. hashes and manifest


@pytest.mark.parametrize("name", sorted(EXPECTED_SHA256))
def test_written_file_hash_matches_constant(synth, name):
    digest = hashlib.sha256((synth.out_dir / name).read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256[name]


def test_manifest_lists_hashes_rows_seed_and_snapshot(synth):
    on_disk = json.loads((synth.out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == synth.manifest
    assert on_disk["seed"] == 20231231
    assert isinstance(on_disk["data_snapshot"], int)
    assert set(on_disk["files"]) == set(EXPECTED_SHA256)
    for name, entry in on_disk["files"].items():
        assert entry["sha256"] == EXPECTED_SHA256[name]
        assert entry["rows"] == len(synth.raw[name])


# 2. format


@pytest.mark.parametrize("name", sorted(HEADERS))
def test_csv_is_utf8_with_unix_line_endings_and_header(synth, name):
    raw = (synth.out_dir / name).read_bytes()
    text = raw.decode("utf-8")
    assert "\r" not in text
    assert text.endswith("\n")
    assert text.split("\n", 1)[0] == HEADERS[name]


def test_unpaid_is_an_empty_unquoted_field(synth):
    text = (synth.out_dir / "transactions.csv").read_text(encoding="utf-8")
    assert '"' not in text
    rows = list(csv.reader(io.StringIO(text)))
    paid_index = rows[0].index("paid_on")
    assert any(row[paid_index] == "" for row in rows[1:])
    assert all(len(row) == len(rows[0]) for row in rows)


def test_amounts_have_two_decimals_and_are_positive(synth):
    amounts = synth.raw["transactions.csv"]["amount"]
    assert amounts.str.fullmatch(r"\d+\.\d{2}").all()
    assert (synth.tx["cents"] > 0).all()


# 3. constraints


def test_companies_follow_ticker_reference(synth):
    companies = synth.companies
    tickers = pd.read_csv(c.TICKERS_PATH, dtype=str, keep_default_na=False, encoding="utf-8")
    assert len(companies) == 69
    assert companies["company_id"].tolist() == [str(i) for i in range(1, 70)]
    assert companies["ticker"].tolist() == sorted(tickers["ticker"])
    reference = tickers.set_index("ticker")
    for row in companies.itertuples():
        assert row.company_name == reference.at[row.ticker, "company_name"]
        assert row.sector == reference.at[row.ticker, "sector"]
    assert set(companies["size_tier"]) <= {"S", "M", "L"}


def test_vendors_ids_order_and_value_lists(synth):
    vendors = synth.vendors
    assert vendors["vendor_id"].tolist() == [str(i) for i in range(1, len(vendors) + 1)]
    keys = list(zip(vendors["onboarded_on"], vendors["vendor_name"], strict=True))
    assert keys == sorted(keys)
    assert vendors["vendor_name"].is_unique
    assert set(vendors["country"]) <= {"US", "GB", "DE", "FR", "IN"}
    assert set(vendors["category"]) <= CATEGORIES
    assert set(vendors["payment_terms_days"]) <= {"30", "45", "60", "90"}
    assert vendors["onboarded_on"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()


def test_transaction_keys_and_foreign_keys(synth):
    tx = synth.tx
    assert tx["txn_id"].tolist() == list(range(1, len(tx) + 1))
    assert tx["invoice_date"].is_monotonic_increasing
    assert tx["company_id"].isin(range(1, 70)).all()
    assert tx["vendor_id"].isin(synth.vendors["vendor_id"].astype("int64")).all()
    assert not tx.duplicated(["vendor_id", "invoice_number"]).any()


def test_transaction_dates_and_terms(synth):
    tx = synth.tx
    terms = synth.vendors.set_index(synth.vendors["vendor_id"].astype("int64"))[
        "payment_terms_days"
    ].astype("int64")
    gap = (tx["due_date"] - tx["invoice_date"]).dt.days
    assert (gap == tx["vendor_id"].map(terms)).all()
    assert tx["invoice_date"].min() >= pd.Timestamp(FIRST_DAY)
    assert tx["invoice_date"].max() <= pd.Timestamp(LAST_DAY)
    assert tx["paid_on"].max() <= pd.Timestamp(LAST_DAY)


def test_methods_and_only_paid_on_has_empties(synth):
    raw = synth.raw["transactions.csv"]
    assert set(raw["payment_method"]) <= {"ach", "wire", "check", "card"}
    empties = (raw == "").sum()
    assert empties.drop("paid_on").sum() == 0
    assert empties["paid_on"] > 0
    assert (synth.companies == "").sum().sum() == 0
    assert (synth.vendors == "").sum().sum() == 0


# 4. labels


def test_labels_sorted_unique_and_known(synth):
    labels = synth.labels
    assert set(labels["anomaly_type"]) == SIX_TYPES
    keys = list(zip(labels["txn_id"], labels["anomaly_type"], strict=True))
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)
    assert labels["txn_id"].isin(synth.tx["txn_id"]).all()


def test_late_payment_labels_are_exactly_more_than_fifteen_days_late(synth):
    tx = synth.tx
    days = (tx["paid_on"] - tx["due_date"]).dt.days
    expected = set(tx.loc[days > LATE_DAYS, "txn_id"])
    assert set(labelled(synth, "late_payment")["txn_id"]) == expected
    labelled_ids = set(labelled(synth, "late_payment")["txn_id"])
    sixteen = set(tx.loc[days == LATE_DAYS + 1, "txn_id"])
    fifteen = set(tx.loc[days == LATE_DAYS, "txn_id"])
    assert sixteen and sixteen <= labelled_ids
    assert fifteen and not fifteen & labelled_ids


# 5. anomaly shapes


def test_weekend_payments_are_exactly_the_labelled_ones(synth):
    paid = synth.tx[synth.tx["paid_on"].notna()]
    weekend = paid["paid_on"].dt.day_of_week >= c.SATURDAY
    flagged = set(labelled(synth, "weekend_payment")["txn_id"])
    assert flagged
    assert set(paid.loc[weekend, "txn_id"]) == flagged


def test_split_payment_events(synth):
    rows = labelled(synth, "split_payment").sort_values(["company_id", "vendor_id", "invoice_date"])
    gap = rows.groupby(["company_id", "vendor_id"])["invoice_date"].diff().dt.days
    rows = rows.assign(event=((gap.isna()) | (gap > c.SPLIT_SPAN_DAYS)).cumsum())
    sizes = rows.groupby("event").size()
    span = rows.groupby("event")["invoice_date"].agg(lambda s: (s.max() - s.min()).days)
    assert sizes.between(3, 5).all()
    assert (span <= c.SPLIT_SPAN_DAYS).all()
    assert rows["cents"].between(SPLIT_LOW_CENTS, SPLIT_HIGH_CENTS).all()
    assert (rows["cents"] < APPROVAL_CENTS).all()


def test_duplicates_copy_an_earlier_invoice(synth):
    tx = synth.tx
    dups = labelled(synth, "duplicate_invoice")
    originals = tx[~tx["txn_id"].isin(dups["txn_id"])]
    key = ["company_id", "vendor_id", "cents"]
    pairs = dups.merge(originals, on=key, suffixes=("", "_orig"))
    gap = (pairs["invoice_date"] - pairs["invoice_date_orig"]).dt.days
    matched = set(pairs.loc[gap.between(*c.DUPLICATE_DELAY_DAYS), "txn_id"])
    assert len(dups) > 0
    assert matched == set(dups["txn_id"])


def test_new_vendor_large_shape(synth):
    tx = synth.tx
    rows = labelled(synth, "new_vendor_large")
    assert len(rows) > 0
    first = tx.groupby("vendor_id")["txn_id"].min()
    assert (rows["txn_id"] == rows["vendor_id"].map(first)).all()
    onboarded = pd.to_datetime(
        synth.vendors.set_index(synth.vendors["vendor_id"].astype("int64"))["onboarded_on"]
    )
    days = (rows["invoice_date"] - rows["vendor_id"].map(onboarded)).dt.days
    assert days.between(0, 14).all()
    rest = tx[~tx["txn_id"].isin(set(rows["txn_id"]))]
    p99 = rest.groupby("company_id")["cents"].quantile(c.LARGE_PERCENTILE / 100)
    assert (rows["cents"] >= rows["company_id"].map(p99)).all()
    assert rows["paid_on"].notna().all()
    assert (rows["paid_on"] < rows["due_date"]).all()


def test_amount_spikes_are_far_above_pair_history(synth):
    tx = synth.tx
    spikes = labelled(synth, "amount_spike")
    assert len(spikes) > 0
    pairs = tx[
        tx.set_index(["company_id", "vendor_id"]).index.isin(
            spikes.set_index(["company_id", "vendor_id"]).index
        )
    ]
    history = {
        key: g.sort_values("txn_id") for key, g in pairs.groupby(["company_id", "vendor_id"])
    }
    for row in spikes.itertuples():
        group = history[(row.company_id, row.vendor_id)]
        earlier = group.loc[group["txn_id"] < row.txn_id, "cents"]
        assert len(earlier) > 0
        assert row.cents >= SPIKE_MIN_FACTOR * earlier.median(), row.txn_id


# 6. windows


@pytest.mark.parametrize(
    ("day", "window"),
    [
        ("2022-01-01", "train"),
        ("2023-06-30", "train"),
        ("2023-07-01", "val"),
        ("2023-09-30", "val"),
        ("2023-10-01", "test"),
        ("2023-12-31", "test"),
        ("2024-01-01", None),
    ],
)
def test_window_edges(day, window):
    assert window_of(day) == window


def test_window_counts_use_inclusive_edges_on_both_bases():
    days = ["2023-06-30", "2023-07-01", "2023-09-30", "2023-10-01"]
    transactions = pd.DataFrame(
        {"txn_id": [1, 2, 3, 4, 5], "invoice_date": days + [days[0]], "paid_on": days + [""]}
    )
    labels = pd.DataFrame({"txn_id": [1, 2, 3, 4, 5], "anomaly_type": ["amount_spike"] * 5})
    counts = window_counts(transactions, labels)
    got = counts[counts["anomaly_type"] == "amount_spike"].set_index(["window", "basis"])
    assert got.at[("train", "paid_on"), "positives"] == 1
    assert got.at[("val", "paid_on"), "positives"] == 2
    assert got.at[("test", "paid_on"), "positives"] == 1
    assert got.at[("train", "invoice_date"), "positives"] == 2
    assert got.at[("val", "invoice_date"), "positives"] == 2
    assert got.at[("test", "invoice_date"), "positives"] == 1


def test_every_type_has_enough_positives_in_every_window(synth):
    raw_tx = synth.raw["transactions.csv"].assign(txn_id=synth.tx["txn_id"])
    joined = synth.labels.merge(raw_tx[["txn_id", "invoice_date", "paid_on"]], on="txn_id")
    for basis in ("paid_on", "invoice_date"):
        joined["window"] = joined[basis].map(lambda d: window_of(d) if d else None)
        counts = joined.groupby(["anomaly_type", "window"]).size()
        for kind in SIX_TYPES:
            for window in ("train", "val", "test"):
                n = int(counts.get((kind, window), 0))
                assert n >= c.MIN_POSITIVES_PER_WINDOW, (kind, window, basis, n)


def test_check_label_counts_accepts_the_data_and_rejects_a_missing_type(synth):
    check_label_counts(synth.data.transactions, synth.data.labels)
    without = synth.data.labels[synth.data.labels["anomaly_type"] != "weekend_payment"]
    with pytest.raises(GenerationError, match="weekend_payment"):
        check_label_counts(synth.data.transactions, without)


# 7. leak check


def format_features(invoice_numbers: pd.Series, txn_ids: pd.Series) -> pd.DataFrame:
    shape = invoice_numbers.str.replace(r"[A-Za-z]", "A", regex=True)
    shape = shape.str.replace(r"\d", "9", regex=True)
    return pd.DataFrame(
        {
            "txn_id": txn_ids.to_numpy(),
            "length": invoice_numbers.str.len().to_numpy(),
            "digits": invoice_numbers.str.count(r"\d").to_numpy(),
            "letters": invoice_numbers.str.count(r"[A-Za-z]").to_numpy(),
            "others": invoice_numbers.str.count(r"[^A-Za-z0-9]").to_numpy(),
            "ends_letter": invoice_numbers.str[-1]
            .str.fullmatch(r"[A-Za-z]")
            .astype(int)
            .to_numpy(),
            "shape": pd.factorize(shape)[0],
        }
    )


def out_of_fold_auprc(features: pd.DataFrame, target: np.ndarray, groups: pd.Series) -> float:
    """AUPRC of a decision tree scored on folds grouped by invoice month."""
    scores = np.zeros(len(target))
    for train, test in GroupKFold(CV_FOLDS).split(features, target, groups):
        model = DecisionTreeClassifier(
            min_samples_leaf=LEAK_MIN_LEAF, max_depth=LEAK_MAX_DEPTH, random_state=0
        ).fit(features.iloc[train], target[train])
        scores[test] = model.predict_proba(features.iloc[test])[:, 1]
    return float(average_precision_score(target, scores))


def leak_ratios(tx: pd.DataFrame, labels: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Out-of-fold AUPRC and base rate per type, from invoice-number format and id only."""
    features = format_features(tx["invoice_number"], tx["txn_id"])
    groups = tx["invoice_date"].dt.strftime("%Y-%m")
    by_type = {k: set(g["txn_id"]) for k, g in labels.groupby("anomaly_type")}
    by_type["union_non_late"] = set().union(
        *(ids for kind, ids in by_type.items() if kind != "late_payment")
    )
    result = {}
    for kind, ids in by_type.items():
        target = tx["txn_id"].isin(ids).to_numpy().astype(int)
        result[kind] = (out_of_fold_auprc(features, target, groups), target.mean())
    return result


def test_leak_probe_catches_ids_given_to_injected_rows_last(synth):
    tx = synth.tx
    target = tx["txn_id"].isin(set(synth.labels["txn_id"])).to_numpy().astype(int)
    features = format_features(tx["invoice_number"], tx["txn_id"])
    features["txn_id"] = np.where(target == 1, len(tx) + np.arange(len(tx)), features["txn_id"])
    groups = tx["invoice_date"].dt.strftime("%Y-%m")
    assert out_of_fold_auprc(features, target, groups) > MAX_LEAK_RATIO * target.mean()


def test_invoice_format_and_id_do_not_reveal_labels(synth):
    ratios = leak_ratios(synth.tx, synth.labels)
    assert set(ratios) == SIX_TYPES | {"union_non_late"}
    for kind, (auprc, base) in ratios.items():
        assert auprc <= MAX_LEAK_RATIO * base, f"{kind}: AUPRC {auprc:.4f}, base rate {base:.4f}"


# 8. determinism


def test_another_seed_changes_the_transactions(synth):
    other = generate(c.SEED + 1).transactions
    text = other.to_csv(index=False, lineterminator="\n").encode("utf-8")
    assert hashlib.sha256(text).hexdigest() != EXPECTED_SHA256["transactions.csv"]


# 9. command line


def test_cli_writes_all_files_and_prints_one_line(synth, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "generate", lambda: synth.data)
    out = tmp_path / "data"
    assert cli.main(["--out-dir", str(out)]) == 0
    printed = capsys.readouterr().out
    assert len(printed.strip().splitlines()) == 1
    names = set(HEADERS) | {"manifest.json"}
    assert {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} == names
    for name, digest in EXPECTED_SHA256.items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_cli_rejects_a_file_as_output_folder(tmp_path, capsys):
    target = tmp_path / "taken"
    target.write_text("x", encoding="utf-8")
    assert cli.main(["--out-dir", str(target)]) == 1
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1
    assert "Traceback" not in err


def test_cli_reports_a_generation_error_in_one_line(tmp_path, monkeypatch, capsys):
    def fail():
        raise GenerationError("too few positives")

    monkeypatch.setattr(cli, "generate", fail)
    assert cli.main(["--out-dir", str(tmp_path / "data")]) == 1
    captured = capsys.readouterr()
    assert len(captured.err.strip().splitlines()) == 1
    assert "Traceback" not in captured.err
    assert captured.out == ""
    assert not (tmp_path / "data").exists()
