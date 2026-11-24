"""Rebuild the SovereignFin database: roles, schema and the synthetic ERP data.

Usage: python -m database.rebuild [--database NAME] [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

REPO_ROOT = Path(__file__).resolve().parent.parent
INIT_SQL = Path(__file__).resolve().parent / "init.sql"
DEFAULT_DATABASE = "sovereignfin"
CONNECT_TIMEOUT_SECONDS = 10
COPY_CHUNK_BYTES = 1 << 20
OWNER_ROLE = "sf_owner"
READER_ROLE = "sf_reader"
SYNTH_COMMAND = "python -m data_pipeline.synth --out-dir data"
REQUIRED_ENV = ("DATABASE_URL_ADMIN", "SF_OWNER_PASSWORD", "SF_READER_PASSWORD")
CONNECT_FAILED = "rebuild: cannot connect to PostgreSQL (is `docker compose up -d` running?)"

# CSV-loaded tables in foreign-key order with their CSV columns.
CSV_TABLES: dict[str, tuple[str, ...]] = {
    "companies": ("company_id", "ticker", "company_name", "sector", "size_tier"),
    "vendors": (
        "vendor_id",
        "vendor_name",
        "country",
        "category",
        "onboarded_on",
        "payment_terms_days",
    ),
    "transactions": (
        "txn_id",
        "company_id",
        "vendor_id",
        "invoice_number",
        "invoice_date",
        "due_date",
        "paid_on",
        "amount",
        "payment_method",
    ),
}
COUNTED_TABLES = (*CSV_TABLES, "anomaly_scores")


class RebuildError(Exception):
    """An expected failure; the message is the one line to show."""


def _connect(url: str) -> psycopg.Connection:
    return psycopg.connect(url, connect_timeout=CONNECT_TIMEOUT_SECONDS, autocommit=True)


def _open(url: str) -> psycopg.Connection:
    try:
        return _connect(url)
    except psycopg.Error as exc:
        raise RebuildError(CONNECT_FAILED) from exc


def _read_env() -> dict[str, str]:
    values = {}
    for name in REQUIRED_ENV:
        value = os.environ.get(name, "")
        if not value:
            raise RebuildError(f"rebuild: set {name} in the environment (see .env.example).")
        values[name] = value
    return values


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m database.rebuild")
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    return parser.parse_args(argv)


def _check_csv(path: Path, columns: tuple[str, ...]) -> None:
    if path.is_dir():
        raise RebuildError(f"rebuild: {path.name} is a directory, expected a CSV file.")
    try:
        with path.open("rb") as handle:
            header = handle.readline()
    except FileNotFoundError as exc:
        raise RebuildError(f"rebuild: {path.name} not found; run `{SYNTH_COMMAND}`.") from exc
    except OSError as exc:
        raise RebuildError(f"rebuild: cannot read {path.name}.") from exc
    expected = ",".join(columns)
    try:
        line = header.decode("utf-8").removesuffix("\n")
    except UnicodeDecodeError:
        line = ""
    if line != expected:
        raise RebuildError(f"rebuild: {path.name} has an unexpected header (want {expected}).")


def _role_ddl(role: str, password: str, exists: bool) -> sql.Composed:
    verb = "ALTER" if exists else "CREATE"
    return sql.SQL("{} ROLE {} WITH LOGIN PASSWORD {}").format(
        sql.SQL(verb), sql.Identifier(role), sql.Literal(password)
    )


def _prepare_cluster(admin: psycopg.Connection, env: dict[str, str], database: str) -> None:
    passwords = {OWNER_ROLE: env["SF_OWNER_PASSWORD"], READER_ROLE: env["SF_READER_PASSWORD"]}
    for role, password in passwords.items():
        found = admin.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,)).fetchone()
        admin.execute(_role_ddl(role, password, found is not None))
    found = admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,)).fetchone()
    if found is None:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))


def _copy_csv(cur: psycopg.Cursor, table: str, path: Path) -> None:
    columns = sql.SQL(", ").join(sql.Identifier(c) for c in CSV_TABLES[table])
    statement = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT csv, HEADER true)").format(
        sql.Identifier(table), columns
    )
    with path.open("rb") as handle, cur.copy(statement) as copy:
        while chunk := handle.read(COPY_CHUNK_BYTES):
            copy.write(chunk)


def _load(url: str, data_dir: Path) -> dict[str, int]:
    init_text = INIT_SQL.read_text(encoding="utf-8")
    counts: dict[str, int] = {}
    with psycopg.connect(url, connect_timeout=CONNECT_TIMEOUT_SECONDS) as conn:
        cur = conn.cursor()
        cur.execute(init_text)
        cur.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(OWNER_ROLE)))
        for table in CSV_TABLES:
            _copy_csv(cur, table, data_dir / f"{table}.csv")
        cur.execute("RESET ROLE")
        for table in COUNTED_TABLES:
            cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table)))
            counts[table] = cur.fetchone()[0]
    return counts


def _load_target(admin_url: str, database: str, data_dir: Path) -> dict[str, int]:
    try:
        return _load(make_conninfo(admin_url, dbname=database), data_dir)
    except psycopg.OperationalError as exc:
        if exc.sqlstate is None:
            raise RebuildError(CONNECT_FAILED) from exc
        raise


def _run(args: argparse.Namespace) -> str:
    env = _read_env()
    for table, columns in CSV_TABLES.items():
        _check_csv(args.data_dir / f"{table}.csv", columns)
    admin_url = env["DATABASE_URL_ADMIN"]
    with _open(admin_url) as admin:
        _prepare_cluster(admin, env, args.database)
    counts = _load_target(admin_url, args.database, args.data_dir)
    summary = ", ".join(f"{table} {n}" for table, n in counts.items())
    return f"rebuild: {args.database} ready: {summary}"


def main(argv: list[str] | None = None) -> int:
    """Run the rebuild; print one line and return the exit code."""
    try:
        print(_run(_parse_args(argv)))
    except RebuildError as exc:
        print(exc, file=sys.stderr)
        return 1
    except psycopg.Error as exc:
        state = exc.sqlstate or "unknown"
        print(
            f"rebuild: database error ({state}): {exc.diag.message_primary}",
            file=sys.stderr,
        )
        return 1
    except OSError:
        print("rebuild: cannot read a data file.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
