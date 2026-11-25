"""Integration tests of the PostgreSQL layer.

They need PostgreSQL 16 with DATABASE_URL_ADMIN, SF_OWNER_PASSWORD and SF_READER_PASSWORD set.
The tests rebuild the database `sovereignfin_test` from generated synthetic CSV files, then
check the schema, the roles and the named queries against it.
"""

import csv
import os
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import date
from decimal import Decimal

import psycopg
import pytest
from psycopg.conninfo import make_conninfo

from data_pipeline.synth.generator import generate
from data_pipeline.synth.output import write_outputs
from database import queries, rebuild

pytestmark = pytest.mark.integration

TEST_DATABASE = "sovereignfin_test"
CONNECT_TIMEOUT_SECONDS = 10
MODEL = "test-model"
TOP_N = 3
LATE_THRESHOLD = 15
ALL_TABLES = ("companies", "vendors", "transactions", "anomaly_scores")
SCORE_LOW, SCORE_MID, SCORE_HIGH = 0.25, 0.5, 0.75
SNAP_OLD, SNAP_MID, SNAP_NEW = 1, 2, 3
COLUMNS = {
    "company_overview": [
        "ticker",
        "company_name",
        "sector",
        "size_tier",
        "vendor_count",
        "txn_count",
        "total_amount",
        "unpaid_count",
        "first_invoice_date",
        "last_invoice_date",
    ],
    "vendor_payments": [
        "vendor_id",
        "vendor_name",
        "category",
        "country",
        "txn_count",
        "total_amount",
        "avg_days_to_pay",
        "last_paid_on",
    ],
    "late_payments": [
        "ticker",
        "vendor_id",
        "vendor_name",
        "late_count",
        "avg_days_late",
        "max_days_late",
        "late_amount",
    ],
    "top_transactions": [
        "txn_id",
        "ticker",
        "invoice_number",
        "invoice_date",
        "due_date",
        "paid_on",
        "amount",
        "vendor_id",
        "vendor_name",
        "payment_method",
    ],
    "anomaly_summary": [
        "model_version",
        "scored_count",
        "flagged_count",
        "max_score",
        "avg_score",
        "data_snapshot",
        "last_scored_at",
    ],
}
UPSERT = """
INSERT INTO anomaly_scores (txn_id, model_version, score, reasons, data_snapshot)
VALUES (%(txn)s, %(model)s, %(score)s, %(reasons)s, %(snap)s)
ON CONFLICT (txn_id, model_version) DO UPDATE
SET score = EXCLUDED.score, reasons = EXCLUDED.reasons,
    data_snapshot = EXCLUDED.data_snapshot, scored_at = now()
WHERE anomaly_scores.data_snapshot <= EXCLUDED.data_snapshot
"""


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def env():
    missing = [n for n in rebuild.REQUIRED_ENV if not os.environ.get(n)]
    assert not missing, f"Set {', '.join(missing)} to run the PostgreSQL integration tests."
    return {name: os.environ[name] for name in rebuild.REQUIRED_ENV}


@pytest.fixture(scope="module")
def built(env, tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("synth")
    write_outputs(generate(), data_dir)
    assert rebuild.main(["--database", TEST_DATABASE, "--data-dir", str(data_dir)]) == 0
    admin = env["DATABASE_URL_ADMIN"]
    base = {"dbname": TEST_DATABASE, "connect_timeout": CONNECT_TIMEOUT_SECONDS}
    return {
        "reader": make_conninfo(
            admin, user="sf_reader", password=env["SF_READER_PASSWORD"], **base
        ),
        "owner": make_conninfo(admin, user="sf_owner", password=env["SF_OWNER_PASSWORD"], **base),
        "csv": {t: _read_csv(data_dir / f"{t}.csv") for t in rebuild.CSV_TABLES},
    }


@contextmanager
def _connect(conninfo, autocommit=False):
    with psycopg.connect(conninfo, autocommit=autocommit) as conn:
        yield conn
        if not autocommit:
            conn.rollback()


def _reader(built):
    return _connect(built["reader"], autocommit=True)


def _txn_rows(built, ticker=None):
    companies = {r["company_id"]: r["ticker"] for r in built["csv"]["companies"]}
    rows = built["csv"]["transactions"]
    return [r for r in rows if ticker is None or companies[r["company_id"]] == ticker]


def _days(later, earlier):
    return (date.fromisoformat(later) - date.fromisoformat(earlier)).days


@pytest.fixture
def reader_url(built, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", built["reader"])


@pytest.fixture
def scored(built):
    """One committed anomaly score on the first transaction, removed afterwards."""
    txn_id = int(built["csv"]["transactions"][0]["txn_id"])
    with _connect(built["owner"], autocommit=True) as conn:
        conn.execute(
            UPSERT,
            {"txn": txn_id, "model": MODEL, "score": SCORE_HIGH, "reasons": ["r"], "snap": 7},
        )
        try:
            yield txn_id
        finally:
            conn.execute("DELETE FROM anomaly_scores WHERE model_version = %s", (MODEL,))


def test_row_counts_equal_csv_rows(built):
    with _reader(built) as conn:
        for table in rebuild.CSV_TABLES:
            count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            assert count == len(built["csv"][table])
        assert conn.execute("SELECT count(*) FROM anomaly_scores").fetchone()[0] == 0


def test_no_orphan_transactions(built):
    with _reader(built) as conn:
        orphans = conn.execute(
            "SELECT count(*) FROM transactions t "
            "LEFT JOIN companies c USING (company_id) LEFT JOIN vendors v USING (vendor_id) "
            "WHERE c.company_id IS NULL OR v.vendor_id IS NULL"
        ).fetchone()[0]
    assert orphans == 0


def test_all_tables_are_owned_by_sf_owner(built):
    with _reader(built) as conn:
        owners = conn.execute(
            "SELECT tablename, tableowner FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
    assert {t: o for t, o in owners if t in ALL_TABLES} == dict.fromkeys(ALL_TABLES, "sf_owner")


def _first_txn(built):
    return built["csv"]["transactions"][0]


def _insert_txn(conn, row, **override):
    values = {**row, **override}
    conn.execute(
        "INSERT INTO transactions (txn_id, company_id, vendor_id, invoice_number, invoice_date,"
        " due_date, paid_on, amount, payment_method) VALUES (%(txn_id)s, %(company_id)s,"
        " %(vendor_id)s, %(invoice_number)s, %(invoice_date)s, %(due_date)s,"
        " NULLIF(%(paid_on)s, '')::date, %(amount)s, %(payment_method)s)",
        values,
    )


NEW_ID = 10**12
MAX_VENDOR_ID = 2**31 - 1


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"amount": "0"}, psycopg.errors.CheckViolation),
        ({"due_date": "1999-01-01"}, psycopg.errors.CheckViolation),
        ({"company_id": 32000}, psycopg.errors.ForeignKeyViolation),
        ({"same_invoice": True}, psycopg.errors.UniqueViolation),
    ],
    ids=["amount_zero", "due_before_invoice", "unknown_company", "duplicate_invoice"],
)
def test_transaction_constraints_reject_bad_rows(built, override, error):
    row = _first_txn(built)
    same = override.pop("same_invoice", False)
    if "due_date" in override:
        override["invoice_date"] = "2000-01-01"
    fresh = {} if same else {"invoice_number": "bad-row"}
    with _connect(built["owner"]) as conn:
        with pytest.raises(error):
            _insert_txn(conn, row, **{"txn_id": NEW_ID, **fresh, **override})


def test_payment_terms_outside_the_set_are_rejected(built):
    vendor = built["csv"]["vendors"][0]
    with _connect(built["owner"]) as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(
                "INSERT INTO vendors VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    MAX_VENDOR_ID,
                    "Terms Test",
                    vendor["country"],
                    vendor["category"],
                    date(2023, 1, 1),
                    31,
                ),
            )


def test_reader_settings_and_select(built):
    with _reader(built) as conn:
        assert conn.execute("SELECT count(*) FROM companies").fetchone()[0] > 0
        assert conn.execute("SHOW default_transaction_read_only").fetchone()[0] == "on"
        assert conn.execute("SHOW statement_timeout").fetchone()[0] == "5s"


def test_reader_cannot_write_even_in_a_read_write_transaction(built):
    row = _first_txn(built)
    with _reader(built) as conn:
        conn.execute("BEGIN READ WRITE")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            _insert_txn(conn, row, txn_id=NEW_ID, invoice_number="reader-write")
        conn.execute("ROLLBACK")


def test_reader_cannot_create_tables(built):
    with _reader(built) as conn:
        conn.execute("BEGIN READ WRITE")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("CREATE TABLE reader_made (x int)")
        conn.execute("ROLLBACK")


def test_default_privileges_give_reader_select_on_new_tables(built):
    with _connect(built["owner"], autocommit=True) as owner:
        owner.execute("CREATE TABLE extra_loaded (x int)")
        try:
            owner.execute("INSERT INTO extra_loaded VALUES (1)")
            with _reader(built) as reader:
                assert reader.execute("SELECT x FROM extra_loaded").fetchall() == [(1,)]
        finally:
            owner.execute("DROP TABLE extra_loaded")


def _score_state(conn, txn_id):
    row = conn.execute(
        "SELECT score, data_snapshot FROM anomaly_scores WHERE txn_id = %s AND model_version = %s",
        (txn_id, MODEL),
    ).fetchone()
    return row


def _upsert(conn, txn_id, score, snap):
    conn.execute(
        UPSERT, {"txn": txn_id, "model": MODEL, "score": score, "reasons": ["x"], "snap": snap}
    )


def test_upsert_never_replaces_newer_data(built):
    txn_id = int(_first_txn(built)["txn_id"])
    with _connect(built["owner"]) as conn:
        _upsert(conn, txn_id, SCORE_MID, SNAP_MID)
        _upsert(conn, txn_id, SCORE_LOW, SNAP_OLD)
        assert _score_state(conn, txn_id) == (SCORE_MID, SNAP_MID)
        _upsert(conn, txn_id, SCORE_HIGH, SNAP_NEW)
        assert _score_state(conn, txn_id) == (SCORE_HIGH, SNAP_NEW)
        _upsert(conn, txn_id, SCORE_LOW, SNAP_NEW)
        assert _score_state(conn, txn_id) == (SCORE_LOW, SNAP_NEW)


def test_score_above_one_is_rejected(built):
    txn_id = int(_first_txn(built)["txn_id"])
    with _connect(built["owner"]) as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            _upsert(conn, txn_id, 1.5, SNAP_OLD)


def _run(name, **params):
    return queries.run_named(name, params)


def test_company_overview_matches_csv(built, reader_url):
    companies = {r["ticker"]: r["company_id"] for r in built["csv"]["companies"]}
    rows = _run("company_overview", ticker="AAPL")
    assert len(rows) == 1
    assert list(rows[0]) == COLUMNS["company_overview"]
    expected = [r for r in built["csv"]["transactions"] if r["company_id"] == companies["AAPL"]]
    assert rows[0]["txn_count"] == len(expected)
    assert rows[0]["total_amount"] == sum(Decimal(r["amount"]) for r in expected)
    assert isinstance(rows[0]["first_invoice_date"], date)


def test_vendor_payments_columns_and_types(built, reader_url):
    rows = _run("vendor_payments", ticker="AAPL", start_date=None, end_date=None, limit=5)
    assert 0 < len(rows) <= 5
    assert list(rows[0]) == COLUMNS["vendor_payments"]
    assert isinstance(rows[0]["total_amount"], Decimal)
    totals = [r["total_amount"] for r in rows]
    assert totals == sorted(totals, reverse=True)


def test_late_payments_exclude_exactly_the_threshold(built, reader_url):
    paid = [r for r in built["csv"]["transactions"] if r["paid_on"]]
    exact = Counter()
    for r in paid:
        if _days(r["paid_on"], r["due_date"]) == LATE_THRESHOLD:
            exact[r["company_id"]] += 1
    company_id = exact.most_common(1)[0][0]
    ticker = next(c["ticker"] for c in built["csv"]["companies"] if c["company_id"] == company_id)
    expected = defaultdict(int)
    for r in paid:
        if r["company_id"] == company_id and _days(r["paid_on"], r["due_date"]) > LATE_THRESHOLD:
            expected[int(r["vendor_id"])] += 1
    rows = _run(
        "late_payments",
        ticker=ticker,
        start_date=None,
        end_date=None,
        min_days_late=LATE_THRESHOLD,
        limit=queries.MAX_LIMIT,
    )
    assert list(rows[0]) == COLUMNS["late_payments"]
    assert len(rows) == min(len(expected), queries.MAX_LIMIT)
    for row in rows:
        assert row["late_count"] == expected[row["vendor_id"]]
        assert row["max_days_late"] > LATE_THRESHOLD
    assert isinstance(rows[0]["late_amount"], Decimal)


def test_top_transactions_are_the_largest_in_order(built, reader_url):
    rows = _run("top_transactions", ticker=None, start_date=None, end_date=None, limit=TOP_N)
    amounts = sorted((Decimal(r["amount"]) for r in built["csv"]["transactions"]), reverse=True)
    assert [r["amount"] for r in rows] == amounts[:TOP_N]
    assert list(rows[0]) == COLUMNS["top_transactions"]
    assert isinstance(rows[0]["invoice_date"], date)


def test_ticker_none_spans_companies(built, reader_url):
    rows = _run(
        "top_transactions", ticker=None, start_date=None, end_date=None, limit=queries.MAX_LIMIT
    )
    assert len({r["ticker"] for r in rows}) > 1


def test_unknown_ticker_gives_no_rows(built, reader_url):
    assert _run("company_overview", ticker="ZZZZ") == []
    dated = {"ticker": "ZZZZ", "start_date": None, "end_date": None}
    assert _run("vendor_payments", **dated, limit=5) == []
    assert _run("top_transactions", **dated, limit=5) == []
    assert _run("late_payments", **dated, min_days_late=0, limit=5) == []
    assert _run("anomaly_summary", ticker="ZZZZ", min_score=0) == []


def test_anomaly_summary_columns_and_filter(built, reader_url, scored):
    rows = _run("anomaly_summary", ticker=None, min_score=SCORE_MID)
    assert list(rows[0]) == COLUMNS["anomaly_summary"]
    assert rows[0]["model_version"] == MODEL
    assert rows[0]["scored_count"] == 1
    assert rows[0]["flagged_count"] == 1
    assert rows[0]["data_snapshot"] == 7
    assert _run("anomaly_summary", ticker=None, min_score=1)[0]["flagged_count"] == 0
