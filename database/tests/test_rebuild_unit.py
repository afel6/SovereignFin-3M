"""Unit tests of the rebuild command: argument checks, CSV checks and error messages."""

import psycopg
import pytest
from psycopg import sql

from database import rebuild

OWNER_PASSWORD = "owner" + "-" + "pw1"
READER_PASSWORD = "reader" + "-" + "pw2"
ADMIN_PASSWORD = "admin" + "-" + "pw3"
ADMIN_URL = "postgresql://postgres:" + ADMIN_PASSWORD + "@db.example.invalid:5432/postgres"
SECRETS = (OWNER_PASSWORD, READER_PASSWORD, ADMIN_PASSWORD, "db.example.invalid")
ENV = {
    "DATABASE_URL_ADMIN": ADMIN_URL,
    "SF_OWNER_PASSWORD": OWNER_PASSWORD,
    "SF_READER_PASSWORD": READER_PASSWORD,
}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)


@pytest.fixture
def data_dir(tmp_path):
    for table, columns in rebuild.CSV_TABLES.items():
        header = (",".join(columns) + "\n").encode("utf-8")
        (tmp_path / f"{table}.csv").write_bytes(header)
    return tmp_path


class FakeAdmin:
    """Admin connection stand-in that records the statements it receives."""

    def __init__(self):
        self.statements = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self.statements.append((query, params))
        return self

    def fetchone(self):
        return None


def _run(capsys, data_dir, *extra):
    code = rebuild.main(["--data-dir", str(data_dir), *extra])
    out = capsys.readouterr()
    return code, out.out + out.err


def _assert_one_clean_line(text):
    assert len(text.strip().splitlines()) == 1
    for secret in SECRETS:
        assert secret not in text


@pytest.mark.parametrize("name", rebuild.REQUIRED_ENV)
@pytest.mark.parametrize("value", [None, ""])
def test_missing_env_names_only_the_variable(monkeypatch, capsys, data_dir, name, value):
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    code, text = _run(capsys, data_dir)
    assert code == 1
    assert name in text
    _assert_one_clean_line(text)


def test_missing_csv(capsys, data_dir):
    (data_dir / "vendors.csv").unlink()
    code, text = _run(capsys, data_dir)
    assert code == 1
    assert "vendors.csv" in text
    _assert_one_clean_line(text)


def test_csv_path_is_directory(capsys, data_dir):
    (data_dir / "companies.csv").unlink()
    (data_dir / "companies.csv").mkdir()
    code, text = _run(capsys, data_dir)
    assert code == 1
    assert "companies.csv" in text
    _assert_one_clean_line(text)


def test_wrong_header(capsys, data_dir):
    (data_dir / "transactions.csv").write_bytes(b"a,b,c\n")
    code, text = _run(capsys, data_dir)
    assert code == 1
    assert "transactions.csv" in text
    _assert_one_clean_line(text)


def test_connect_failure_hides_url_and_passwords(monkeypatch, capsys, data_dir):
    def refuse(url):
        raise psycopg.OperationalError("could not connect to " + ADMIN_URL)

    monkeypatch.setattr(rebuild, "_connect", refuse)
    code, text = _run(capsys, data_dir)
    assert code == 1
    assert text.strip() == rebuild.CONNECT_FAILED
    _assert_one_clean_line(text)


def test_role_ddl_quotes_identifier_and_literal():
    query = rebuild._role_ddl("sf_owner", "it's" + "-pw", exists=False)
    assert isinstance(query, sql.Composed)
    kinds = {type(part) for part in query}
    assert sql.Identifier in kinds
    assert sql.Literal in kinds
    rendered = query.as_string()
    assert '"sf_owner"' in rendered
    assert "'it''s-pw'" in rendered
    assert rebuild._role_ddl("sf_owner", "x", exists=True).as_string().startswith("ALTER ROLE")


def test_successful_run_uses_identifiers_and_prints_no_secret(monkeypatch, capsys, data_dir):
    admin = FakeAdmin()
    loaded = {}

    def fake_load(url, database, directory):
        loaded.update(url=url, database=database)
        return dict.fromkeys(rebuild.COUNTED_TABLES, 0)

    monkeypatch.setattr(rebuild, "_connect", lambda url: admin)
    monkeypatch.setattr(rebuild, "_load_target", fake_load)
    code, text = _run(capsys, data_dir, "--database", "sovereignfin_test")
    assert code == 0
    assert text.strip() == (
        "rebuild: sovereignfin_test ready: companies 0, vendors 0, transactions 0, anomaly_scores 0"
    )
    assert loaded["database"] == "sovereignfin_test"
    create = [q for q, _ in admin.statements if isinstance(q, sql.Composed)]
    database_ddl = [q for q in create if sql.Identifier("sovereignfin_test") in list(q)]
    assert len(database_ddl) == 1
    assert "CREATE DATABASE" in database_ddl[0].as_string()
    role_ddl = [q for q in create if "ROLE" in q.as_string()]
    assert len(role_ddl) == 2
    for query in role_ddl:
        assert any(isinstance(part, sql.Literal) for part in query)
    for query, params in admin.statements:
        if isinstance(query, str):
            for password in (OWNER_PASSWORD, READER_PASSWORD):
                assert password not in query
                assert password not in (params or ())
    for secret in SECRETS:
        assert secret not in text


def test_database_error_prints_one_line_without_url(monkeypatch, capsys, data_dir):
    class Failing(FakeAdmin):
        def execute(self, query, params=None):
            raise psycopg.errors.InsufficientPrivilege("denied for " + ADMIN_URL)

    monkeypatch.setattr(rebuild, "_connect", lambda url: Failing())
    code, text = _run(capsys, data_dir)
    assert code == 1
    assert text.startswith("rebuild: database error")
    _assert_one_clean_line(text)
