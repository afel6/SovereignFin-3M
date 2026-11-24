"""Live service check: the database server is PostgreSQL 16."""

import os

import pytest

CONNECT_TIMEOUT_SECONDS = 10
EXPECTED_MAJOR_VERSION = 16
VERSION_NUM_DIVISOR = 10000


@pytest.mark.integration
def test_postgres_major_version_is_16():
    """A server running another PostgreSQL major version fails the check."""
    url = os.environ.get("DATABASE_URL_ADMIN")
    assert url, "DATABASE_URL_ADMIN is not set; export it to point at the local PostgreSQL."
    import psycopg

    with psycopg.connect(url, connect_timeout=CONNECT_TIMEOUT_SECONDS) as conn:
        row = conn.execute("SHOW server_version_num").fetchone()
    assert row is not None
    assert int(row[0]) // VERSION_NUM_DIVISOR == EXPECTED_MAJOR_VERSION
