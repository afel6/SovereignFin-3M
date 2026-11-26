import pytest
import sqlglot
from sqlglot import exp

from database.charts import UnsafeSQLError, ensure_read_only


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; DROP TABLE vendors",
        "SELECT * FROM vendors; SELECT * FROM payments",
        "SELECT 1 /* x */; DROP TABLE vendors",
        "SELECT 1 -- x\n; DELETE FROM vendors",
        "/* SELECT 1 */ DROP TABLE vendors",
        "sElEcT 1; dRoP tAbLe vendors",
        "SELECT 1；DROP TABLE vendors",
        "DROP TABLE vendors",
        "DELETE FROM vendors",
        "UPDATE vendors SET name = 'x'",
        "INSERT INTO vendors (name) VALUES ('x')",
        "TRUNCATE vendors",
        "ALTER TABLE vendors ADD COLUMN x int",
        "CREATE TABLE t (id int)",
        "GRANT ALL ON vendors TO PUBLIC",
        "COPY vendors TO PROGRAM 'cat /etc/passwd'",
        "CALL do_something()",
        "DO $$ BEGIN PERFORM 1; END $$",
        "SELECT pg_sleep(10)",
        "SELECT pg_catalog.pg_sleep(10)",
        "SELECT PG_SLEEP(10)",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT pg_read_binary_file('/etc/passwd')",
        "SELECT lo_import('/etc/passwd')",
        "SELECT lo_export(1, '/tmp/x')",
        "SELECT * FROM dblink('host=x', 'SELECT 1') AS t(a int)",
        "SELECT dblink_exec('host=x', 'DROP TABLE vendors')",
        "SELECT set_config('log_statement', 'all', false)",
        "SELECT current_setting('server_version')",
        "SELECT * FROM vendors WHERE id = (SELECT length(pg_read_file('/etc/passwd')))",
        "SELECT id FROM vendors UNION SELECT usename FROM pg_catalog.pg_user",
        "SELECT id FROM vendors UNION SELECT table_name FROM information_schema.tables",
        "WITH x AS (DELETE FROM vendors RETURNING *) SELECT * FROM x",
        "SELECT * INTO newt FROM vendors",
        "SELECT * FROM secrets",
        "SELECT * FROM vendors WHERE id IN (SELECT id FROM secrets)",
        "SELECT * FROM other.vendors",
        "SELECT * FROM vendors FOR UPDATE",
        "SELECT * FROM vendors FOR SHARE",
        "SELECT pg_\u017fleep(10)",
        "SELECT \u017fet_config('x', 'y', false)",
        "SELECT current_sett\u0131ng('x')",
        "SELECT dbl\u0131nk_exec('x', 'y')",
        "SELECT pg_read_\ufb01le('x')",
        "WITH x AS (DELETE FROM vendors RETURNING id) SELECT 1",
        "WITH t AS (SELECT * FROM secrets) SELECT * FROM t",
        "SELECT * FROM vendors JOIN secrets ON true",
        "SELECT 'abc",
        "SELECT 1 /* unterminated",
        "",
        "   ",
        "SELEC * FROM vendors",
        "SELECT * FROM (",
    ],
)
def test_rejects_attacks(sql, allowed):
    with pytest.raises(UnsafeSQLError) as error:
        ensure_read_only(sql, allowed)
    assert isinstance(error.value.reason, str) and error.value.reason
    assert "\n" not in error.value.reason and "Traceback" not in error.value.reason
    assert not sql.strip() or sql not in error.value.reason


@pytest.mark.parametrize(
    ("sql", "limit"),
    [
        ("SELECT * FROM vendors", 100),
        ("SELECT * FROM vendors LIMIT 10", 10),
        ("SELECT * FROM vendors LIMIT 100", 100),
        ("SELECT * FROM vendors LIMIT 101", 100),
        ("SELECT * FROM vendors LIMIT ALL", 100),
        ("SELECT * FROM vendors LIMIT 1 + 1", 100),
        ("WITH t AS (SELECT * FROM vendors) SELECT * FROM t", 100),
        ("SELECT v.id FROM vendors v JOIN payments p ON p.vendor_id = v.id", 100),
        ("SELECT * FROM public.vendors", 100),
        ("SELECT lower(name), count(*), now() FROM vendors GROUP BY 1", 100),
    ],
)
def test_valid_queries(sql, limit, allowed):
    out = ensure_read_only(sql, allowed, 100)
    assert ensure_read_only(out, allowed, 100) == out
    tree = sqlglot.parse_one(out, read="postgres")
    assert int(tree.args["limit"].expression.this) == limit


@pytest.mark.parametrize(
    ("sql", "limit"),
    [
        ("SELECT id FROM vendors UNION SELECT id FROM payments", 100),
        ("SELECT id FROM vendors UNION SELECT id FROM payments LIMIT 10", 10),
    ],
)
def test_union_limit_is_on_the_root(sql, limit, allowed):
    out = ensure_read_only(sql, allowed, 100)
    assert ensure_read_only(out, allowed, 100) == out
    tree = sqlglot.parse_one(out, read="postgres")
    assert isinstance(tree, exp.Union)
    assert int(tree.args["limit"].expression.this) == limit
    assert not list(tree.this.find_all(exp.Limit))
    assert not list(tree.expression.find_all(exp.Limit))


def test_default_limit(allowed):
    tree = sqlglot.parse_one(ensure_read_only("SELECT * FROM vendors", allowed), read="postgres")
    assert int(tree.args["limit"].expression.this) == 500


@pytest.mark.parametrize("max_rows", [0, -1, True, "100"])
def test_rejects_invalid_max_rows(allowed, max_rows):
    with pytest.raises(ValueError, match="max_rows"):
        ensure_read_only("SELECT 1", allowed, max_rows)


@pytest.mark.parametrize("allowed_tables", [{"vendors"}, frozenset({""}), frozenset({1})])
def test_rejects_invalid_allowed_tables(allowed_tables):
    with pytest.raises(ValueError, match="allowed_tables"):
        ensure_read_only("SELECT 1", allowed_tables)


def test_cte_checks_underlying_tables(allowed):
    tree = sqlglot.parse_one(
        ensure_read_only("WITH t AS (SELECT * FROM vendors) SELECT * FROM t", allowed),
        read="postgres",
    )
    assert {table.name for table in tree.find_all(exp.Table)} == {"vendors", "t"}
