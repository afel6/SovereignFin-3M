"""Static checks of the SQL files: placeholders, schema contract, grants and CSV columns."""

import re

import pytest

from data_pipeline.synth import constants
from database import queries, rebuild

PLACEHOLDER = re.compile(r"%\((\w+)\)s")
COMMENT = re.compile(r"--[^\n]*")
EXPECTED_NAMES = {
    "company_overview",
    "vendor_payments",
    "late_payments",
    "top_transactions",
    "anomaly_summary",
}


def _query_text(name: str) -> str:
    return (queries.QUERIES_DIR / f"{name}.sql").read_text(encoding="utf-8")


def _init_text() -> str:
    return rebuild.INIT_SQL.read_text(encoding="utf-8")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def test_query_names_are_exactly_the_five_named_queries():
    assert set(queries.QUERY_PARAMS) == EXPECTED_NAMES
    files = {p.stem for p in queries.QUERIES_DIR.glob("*.sql")}
    assert files == EXPECTED_NAMES


@pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
def test_placeholders_match_declared_params(name):
    assert set(PLACEHOLDER.findall(_query_text(name))) == set(queries.QUERY_PARAMS[name])


@pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
def test_no_other_format_markers(name):
    leftover = PLACEHOLDER.sub("", _query_text(name))
    assert "%" not in leftover
    assert "{" not in leftover
    assert "?" not in leftover
    assert "$1" not in leftover


@pytest.mark.parametrize(
    "fragment",
    [
        "score real NOT NULL CHECK (score BETWEEN 0 AND 1)",
        "reasons text[] NOT NULL",
        "data_snapshot bigint NOT NULL",
        "scored_at timestamptz NOT NULL DEFAULT now()",
        "PRIMARY KEY (txn_id, model_version)",
        "txn_id bigint REFERENCES transactions",
        "UNIQUE (vendor_id, invoice_number)",
        "CHECK (due_date >= invoice_date)",
        "CHECK (amount > 0)",
        "payment_terms_days IN (30, 45, 60, 90)",
        "GRANT SELECT ON ALL TABLES IN SCHEMA public TO sf_reader",
        "ALTER DEFAULT PRIVILEGES FOR ROLE sf_owner IN SCHEMA public"
        " GRANT SELECT ON TABLES TO sf_reader",
        "default_transaction_read_only = on",
        "statement_timeout = '5s'",
    ],
)
def test_init_sql_contains_contract_fragment(fragment):
    assert fragment in _squash(_init_text())


def test_init_sql_has_no_password_and_no_write_grant_to_reader():
    text = COMMENT.sub("", _init_text())
    assert not re.search(r"\bPASSWORD\b", text, re.IGNORECASE)
    for line in text.upper().splitlines():
        if "GRANT" in line and "SF_READER" in line:
            assert not re.search(r"\b(INSERT|UPDATE|DELETE|ALL PRIVILEGES)\b", line)


@pytest.mark.parametrize(
    ("table", "columns"),
    [
        ("companies", constants.COMPANY_COLUMNS),
        ("vendors", constants.VENDOR_COLUMNS),
        ("transactions", constants.TRANSACTION_COLUMNS),
    ],
)
def test_generator_columns_equal_copy_columns(table, columns):
    assert list(rebuild.CSV_TABLES[table]) == columns
