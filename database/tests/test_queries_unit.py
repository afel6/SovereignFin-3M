"""Unit tests of parameter validation and run_named against a fake connection."""

import threading
import traceback
from datetime import date, datetime
from types import SimpleNamespace

import psycopg
import pytest

from database import queries

WIDE = {"ticker": "AAPL", "start_date": None, "end_date": None, "limit": 5}
FAKE_PASSWORD = "pw" + "-" + "hunter2"
FAKE_URL = "postgresql://sf_reader:" + FAKE_PASSWORD + "@db.example.invalid:5432/sovereignfin"
BAD_ESCAPE_PASSWORD = "pw" + "%zz" + "-hunter2"
WAIT_LIMIT_SECONDS = 60
TINY_LIMIT_SECONDS = 0.05
NULLABLE = ("late_payments", "top_transactions", "anomaly_summary")


def _params(name, **override):
    base = {
        "company_overview": {"ticker": "AAPL"},
        "vendor_payments": dict(WIDE),
        "late_payments": {**WIDE, "min_days_late": 0},
        "top_transactions": dict(WIDE),
        "anomaly_summary": {"ticker": "AAPL", "min_score": 0.5},
    }[name]
    return {**base, **override}


def _bad(name, **override):
    with pytest.raises(queries.QueryParamsError):
        queries.validate_params(name, _params(name, **override))


def test_unknown_name_is_rejected():
    with pytest.raises(queries.UnknownQueryError):
        queries.validate_params("drop_everything", {})


def test_missing_and_unexpected_keys_are_named():
    params = _params("vendor_payments")
    del params["limit"]
    with pytest.raises(queries.QueryParamsError, match="limit"):
        queries.validate_params("vendor_payments", params)
    with pytest.raises(queries.QueryParamsError, match="extra_key"):
        queries.validate_params("vendor_payments", _params("vendor_payments", extra_key=1))


@pytest.mark.parametrize("name", sorted(queries.QUERY_PARAMS))
def test_ticker_none_only_where_allowed(name):
    if name in NULLABLE:
        assert queries.validate_params(name, _params(name, ticker=None))["ticker"] is None
    else:
        _bad(name, ticker=None)
    _bad(name, ticker=5)


@pytest.mark.parametrize("key", ["start_date", "end_date"])
def test_dates_accepted(key):
    for value in (date(2023, 1, 31), "2023-01-31"):
        params = _params("top_transactions", **{key: value})
        assert queries.validate_params("top_transactions", params)[key] == date(2023, 1, 31)


@pytest.mark.parametrize("key", ["start_date", "end_date"])
def test_datetime_passes_through_unchanged(key):
    value = datetime(2023, 1, 31, 1)
    params = _params("top_transactions", **{key: value})
    assert queries.validate_params("top_transactions", params)[key] is value


@pytest.mark.parametrize("value", ["2023-1-31", "2023-02-30", 20230131])
def test_dates_rejected(value):
    _bad("top_transactions", start_date=value)
    _bad("top_transactions", end_date=value)


@pytest.mark.parametrize("value", [1, 500])
def test_limit_bounds_pass(value):
    assert queries.validate_params("top_transactions", _params("top_transactions", limit=value))


@pytest.mark.parametrize("value", [0, 501, True, 1.0])
def test_limit_rejected(value):
    _bad("top_transactions", limit=value)


def test_min_days_late():
    assert queries.validate_params("late_payments", _params("late_payments", min_days_late=0))
    assert queries.validate_params("late_payments", _params("late_payments", min_days_late=-1))
    _bad("late_payments", min_days_late=True)


@pytest.mark.parametrize("value", [0, 1, 0.5])
def test_min_score_accepted(value):
    assert queries.validate_params("anomaly_summary", _params("anomaly_summary", min_score=value))


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan"), float("inf"), True])
def test_min_score_rejected(value):
    _bad("anomaly_summary", min_score=value)


def test_input_mapping_is_not_mutated():
    params = _params("top_transactions", start_date="2023-01-31")
    before = dict(params)
    out = queries.validate_params("top_transactions", params)
    assert params == before
    assert out is not params


class FakeConn:
    """Stand-in for a psycopg connection that records what happens to it."""

    def __init__(self, rows=None, execute_error=None, fetch=None):
        self.rows = [{"a": 1}] if rows is None else rows
        self.execute_error = execute_error
        self.fetch = fetch
        self.events = []
        self.sql = None
        self.params = None
        self.read_only = False
        self.cancelled = threading.Event()

    def execute(self, sql, params):
        self.events.append("execute")
        self.sql, self.params = sql, params
        if self.execute_error:
            raise self.execute_error
        return self

    def fetchall(self):
        self.events.append("fetch_start")
        if self.fetch:
            self.fetch(self)
        self.events.append("fetch_end")
        return self.rows

    def cancel_safe(self):
        self.events.append("cancel")
        self.cancelled.set()

    def close(self):
        self.events.append("close")


@pytest.fixture
def install(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", FAKE_URL)

    def _install(conn=None, connect_error=None):
        def connect(url, **kwargs):
            if connect_error:
                raise connect_error
            return conn

        monkeypatch.setattr(queries.psycopg, "connect", connect)
        return conn

    return _install


def test_sql_text_is_the_file_and_params_are_separate(install):
    conn = install(FakeConn(rows=[{"ticker": "AAPL"}, {"ticker": "MSFT"}]))
    out = queries.run_named("top_transactions", _params("top_transactions"))
    assert out == [{"ticker": "AAPL"}, {"ticker": "MSFT"}]
    assert conn.sql == (queries.QUERIES_DIR / "top_transactions.sql").read_text(encoding="utf-8")
    assert conn.params == {"ticker": "AAPL", "start_date": None, "end_date": None, "limit": 5}
    assert "AAPL" not in conn.sql
    assert conn.read_only is True
    assert conn.events[-1] == "close"


def test_database_url_unset(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(queries.DatabaseUnavailableError):
        queries.run_named("company_overview", {"ticker": "AAPL"})


def test_connect_failure_leaks_nothing(install):
    install(connect_error=psycopg.OperationalError("failed " + FAKE_URL))
    with pytest.raises(queries.DatabaseUnavailableError) as info:
        queries.run_named("company_overview", {"ticker": "AAPL"})
    assert FAKE_PASSWORD not in str(info.value)
    assert "db.example.invalid" not in str(info.value)


def test_query_canceled_is_timeout_and_closes(install):
    conn = install(FakeConn(execute_error=psycopg.errors.QueryCanceled("canceled")))
    with pytest.raises(queries.QueryTimeoutError):
        queries.run_named("company_overview", {"ticker": "AAPL"})
    assert conn.events[-1] == "close"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (psycopg.OperationalError("lost " + FAKE_URL), queries.DatabaseUnavailableError),
        (psycopg.errors.UndefinedTable("boom " + FAKE_PASSWORD), queries.QueryError),
    ],
)
def test_other_database_errors_are_friendly(install, error, expected):
    conn = install(FakeConn(execute_error=error))
    with pytest.raises(expected) as info:
        queries.run_named("company_overview", {"ticker": "AAPL"})
    assert FAKE_PASSWORD not in str(info.value)
    assert conn.events[-1] == "close"


def test_connect_error_traceback_does_not_carry_the_password(monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://sf_reader:" + BAD_ESCAPE_PASSWORD + "@127.0.0.1:1/sovereignfin",
    )
    with pytest.raises(queries.DatabaseUnavailableError) as info:
        queries.run_named("company_overview", {"ticker": "AAPL"})
    text = "".join(traceback.format_exception(info.value))
    assert BAD_ESCAPE_PASSWORD not in text
    assert "hunter2" not in text


def test_execute_error_traceback_does_not_carry_the_password(install):
    conn = install(FakeConn(execute_error=psycopg.OperationalError("lost " + FAKE_URL)))
    with pytest.raises(queries.DatabaseUnavailableError) as info:
        queries.run_named("company_overview", {"ticker": "AAPL"})
    assert FAKE_PASSWORD not in "".join(traceback.format_exception(info.value))
    assert conn.events[-1] == "close"


def test_slow_fetch_times_out_and_timer_cancels_while_fetching(install, monkeypatch):
    monkeypatch.setattr(queries, "STATEMENT_TIMEOUT_SECONDS", TINY_LIMIT_SECONDS)

    def slow(conn):
        assert conn.cancelled.wait(WAIT_LIMIT_SECONDS)

    conn = install(FakeConn(fetch=slow))
    with pytest.raises(queries.QueryTimeoutError):
        queries.run_named("company_overview", {"ticker": "AAPL"})
    assert conn.events == ["execute", "fetch_start", "cancel", "fetch_end", "close"]


def test_elapsed_check_after_fetch_without_cancel(install, monkeypatch):
    ticks = iter([0.0, 1000.0])
    monkeypatch.setattr(queries, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    conn = install(FakeConn())
    with pytest.raises(queries.QueryTimeoutError):
        queries.run_named("company_overview", {"ticker": "AAPL"})
    assert "cancel" not in conn.events
    assert conn.events[-1] == "close"
