"""Run the named, read-only SQL queries under the sf_reader role."""

from __future__ import annotations

import math
import os
import re
import threading
import time
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

QUERIES_DIR = Path(__file__).resolve().parent / "queries"
CONNECT_TIMEOUT_SECONDS = 5
STATEMENT_TIMEOUT_SECONDS = 5
MAX_LIMIT = 500
MIN_SCORE_LOWER = 0.0
MIN_SCORE_UPPER = 1.0
ISO_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
TIMEOUT_MESSAGE = f"The database query took longer than {STATEMENT_TIMEOUT_SECONDS} seconds."
UNAVAILABLE_MESSAGE = "The database is unavailable."

QUERY_PARAMS: dict[str, frozenset[str]] = {
    "company_overview": frozenset({"ticker"}),
    "vendor_payments": frozenset({"ticker", "start_date", "end_date", "limit"}),
    "late_payments": frozenset({"ticker", "start_date", "end_date", "min_days_late", "limit"}),
    "top_transactions": frozenset({"ticker", "start_date", "end_date", "limit"}),
    "anomaly_summary": frozenset({"ticker", "min_score"}),
}
# Queries whose ticker may be None (all companies).
NULLABLE_TICKER = frozenset({"late_payments", "top_transactions", "anomaly_summary"})


class QueryError(Exception):
    """Base class for named-query failures; the message is one friendly line."""


class UnknownQueryError(QueryError):
    """The query name is not one of the named queries."""


class QueryParamsError(QueryError):
    """The parameters do not match the query."""


class QueryTimeoutError(QueryError):
    """The query ran longer than the time limit."""


class DatabaseUnavailableError(QueryError):
    """The database cannot be reached."""


def _ticker(name: str, value: object) -> str | None:
    if value is None and name in NULLABLE_TICKER:
        return None
    if isinstance(value, str):
        return value
    raise QueryParamsError("Parameter ticker must be a string.")


def _date(key: str, value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str) and ISO_DATE.fullmatch(value):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise QueryParamsError(f"Parameter {key} must be a date or YYYY-MM-DD.")


def _int(key: str, value: object, low: int | None, high: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise QueryParamsError(f"Parameter {key} must be an integer.")
    if (low is not None and value < low) or (high is not None and value > high):
        bound = f"at least {low}" if high is None else f"between {low} and {high}"
        raise QueryParamsError(f"Parameter {key} must be {bound}.")
    return value


def _score(value: object) -> float | int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise QueryParamsError("Parameter min_score must be a number.")
    if not math.isfinite(value) or not MIN_SCORE_LOWER <= value <= MIN_SCORE_UPPER:
        raise QueryParamsError("Parameter min_score must be between 0 and 1.")
    return value


def _convert(name: str, params: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in params.items():
        if key == "ticker":
            out[key] = _ticker(name, value)
        elif key in ("start_date", "end_date"):
            out[key] = _date(key, value)
        elif key == "limit":
            out[key] = _int(key, value, 1, MAX_LIMIT)
        elif key == "min_days_late":
            out[key] = _int(key, value, None, None)
        else:
            out[key] = _score(value)
    return out


def validate_params(name: str, params: Mapping[str, object]) -> dict[str, object]:
    """Check the name and parameters; return the converted parameters as a new dict."""
    expected = QUERY_PARAMS.get(name) if isinstance(name, str) else None
    if expected is None:
        raise UnknownQueryError("Unknown query name.")
    keys = set(params)
    missing, extra = sorted(expected - keys), sorted(map(str, keys - expected))
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if extra:
            parts.append("unexpected " + ", ".join(extra))
        raise QueryParamsError("Query parameters do not match: " + "; ".join(parts) + ".")
    return _convert(name, params)


def _cancel(conn: psycopg.Connection) -> None:
    # Runs on the timer thread; a failed cancel is still caught by the elapsed-time check.
    try:
        conn.cancel_safe()
    except psycopg.Error:
        return


def _execute(conn: psycopg.Connection, sql_text: str, params: dict[str, object]) -> list[dict]:
    started = time.monotonic()
    timer = threading.Timer(STATEMENT_TIMEOUT_SECONDS, _cancel, args=(conn,))
    timer.start()
    try:
        rows = conn.execute(sql_text, params).fetchall()
    finally:
        timer.cancel()
    if time.monotonic() - started > STATEMENT_TIMEOUT_SECONDS:
        raise QueryTimeoutError(TIMEOUT_MESSAGE)
    return [dict(row) for row in rows]


def run_named(name: str, params: Mapping[str, object]) -> list[dict]:
    """Run a named query with validated parameters and return its rows as dicts."""
    values = validate_params(name, params)
    sql_text = (QUERIES_DIR / f"{name}.sql").read_text(encoding="utf-8")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise DatabaseUnavailableError("DATABASE_URL is not set.")
    try:
        conn = psycopg.connect(url, connect_timeout=CONNECT_TIMEOUT_SECONDS, row_factory=dict_row)
    except psycopg.Error:
        # The driver's text can echo the connection string, password included.
        raise DatabaseUnavailableError(UNAVAILABLE_MESSAGE) from None
    try:
        conn.read_only = True
        return _execute(conn, sql_text, values)
    except psycopg.errors.QueryCanceled as exc:
        raise QueryTimeoutError(TIMEOUT_MESSAGE) from exc
    except psycopg.OperationalError:
        raise DatabaseUnavailableError(UNAVAILABLE_MESSAGE) from None
    except psycopg.Error as exc:
        raise QueryError("The database query failed.") from exc
    finally:
        conn.close()
