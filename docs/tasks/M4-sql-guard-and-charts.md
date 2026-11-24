# Task M4: SQL guard and query-result-to-chart JSON

Paste this whole file into your coding agent (Claude Code, Codex, Cursor or Copilot) as the first
message. It holds everything the task needs.

## Role and goal

You are a Python backend engineer in the SovereignFin-3M repository, a local-first financial
question-answering engine built as a university project. You work for a contributor (your user),
who reviews your work and opens the pull request.

Goal: implement two pure functions in `database/charts/`. `ensure_read_only` checks that a SQL
string is one safe read-only query and returns it with a row limit. `dataframe_to_chart` turns a
query result (a pandas DataFrame) into a JSON-safe `ChartSpec`. The application itself runs only
named queries under a read-only role; the guard is an extra layer.

Work in this order: Setup, Tests to write first, code, Definition of done, Pull request.

## Setup

Requires Python 3.13 and uv (https://docs.astral.sh/uv/). `uv python install 3.13` installs Python
if it is missing.

```bash
git clone https://github.com/afel6/SovereignFin-3M.git
cd SovereignFin-3M
git pull
git switch -c m4/sql-guard-charts
uv sync --group charts --group dev
```

If you cloned earlier, run `git switch main` and `git pull` before you create the branch. Then read
`AGENTS.md` at the repository root, when it exists, and follow it. Run `git pull` on `main` again
at the start of each work session and merge `main` into your branch, so new shared files arrive.
`orchestrator/api_contract.py`, which holds `ChartSpec`, arrives on `main` around 2026-11-25; pull
before you run the chart tests.

## Allowed paths

You create and change files only under these paths:

- `database/charts/`

Every other path is off-limits: other folders, every `api.py`, `orchestrator/api_contract.py`,
`pyproject.toml`, `uv.lock`, `.github/`, `AGENTS.md`, `CLAUDE.md`. A CI check rejects a pull request
that changes any file outside the allowed paths. When you need a new dependency or a contract
change, stop and ask the contributor to open an issue.

## Files to commit

- `database/charts/__init__.py` — exports `ensure_read_only`, `UnsafeSQLError` and `dataframe_to_chart`
- `database/charts/sql_guard.py` — `UnsafeSQLError` and `ensure_read_only`
- `database/charts/sql_to_chart.py` — `dataframe_to_chart`
- `database/charts/tests/` — the tests (`test_*.py`)

## Runtime outputs (never commit)

- None. This task writes no files at run time.

## Contract

These blocks are copied verbatim from `docs/contracts.md` in the repository. Build to them exactly
and leave them unchanged. When one looks wrong, stop and ask the contributor to open an issue.

```contract sql-guard
Module: database/charts/sql_guard.py

def ensure_read_only(sql: str, allowed_tables: frozenset[str], max_rows: int = 500) -> str
  - parse with sqlglot, postgres dialect
  - exactly one statement
  - SELECT or WITH ... SELECT only
  - no DML, DDL, COPY, CALL or DO anywhere in the tree
  - every referenced table is in allowed_tables
  - deny the functions pg_sleep, pg_read_*, lo_*, dblink*, set_config, current_setting
  - append LIMIT max_rows when there is none; clamp a larger LIMIT to max_rows
  - return the SQL to run; otherwise raise UnsafeSQLError(reason)
```

```contract chart-conversion
Module: database/charts/sql_to_chart.py

def dataframe_to_chart(df: pd.DataFrame, chart_type: str | None = None, x: str | None = None,
                       y: list[str] | None = None, title: str = "") -> ChartSpec
ChartSpec is imported from orchestrator.api_contract (see "Chart spec").
```

```contract chart-spec
ChartSpec (orchestrator/api_contract.py; import it from there, never redefine it):
  chart_type: "bar" | "line" | "pie" | "table"
  title: str
  x_field: str | null
  y_fields: list[str]
  data: list[dict[str, str | int | float | null]]   at most 500 rows
```

```contract named-queries
database/queries/*.sql: named, parameterized queries (psycopg %(name)s placeholders):
  company_overview, vendor_payments, late_payments, top_transactions, anomaly_summary
They run only by name, under sf_reader, with statement_timeout 5 s.
Application SQL safety = the read-only role + named queries only. The SQL guard below is an
extra layer; the application does not depend on it.
```

## Guard requirements

These define the part of the guard that the contract leaves to you. In every test, `allowed_tables`
is `frozenset({"vendors", "payments"})` unless the case says otherwise.

- `UnsafeSQLError(Exception)` lives in `sql_guard.py` and carries `reason: str`, a short sentence
  that names the rule that failed and contains no SQL text and no stack trace.
- Empty or whitespace-only input, and every sqlglot parse error, raise `UnsafeSQLError`. No other
  exception type leaves `ensure_read_only` for any string input.
- CTE names are not tables: `WITH t AS (SELECT * FROM vendors) SELECT * FROM t` passes with
  `allowed_tables={"vendors"}`. A table inside a CTE body, a subquery, a `WHERE ... IN (SELECT ...)`
  or a join is checked like any other table.
- `SELECT ... INTO name` creates a table; treat it as DDL and reject it.
- Schema-qualified names pass only when the schema is `public` and the table name is allowed:
  `public.vendors` passes, `pg_catalog.pg_user`, `information_schema.tables` and `other.vendors`
  fail. State this rule in the `ensure_read_only` docstring.
- Function names are compared case-insensitively and without a schema prefix: `pg_sleep`,
  `PG_SLEEP` and `pg_catalog.pg_sleep` are the same denied function. The deny list is `pg_sleep`,
  `pg_read_*`, `lo_*`, `dblink*`, `set_config`, `current_setting`, and it applies anywhere in the
  tree: select list, `WHERE`, subqueries, `FROM` (table functions) and CTEs. Other functions such
  as `count`, `lower` and `now` pass.
- Limit rules, applied to the whole statement: no `LIMIT` gives `LIMIT max_rows`; an integer
  literal at or below `max_rows` stays; an integer literal above `max_rows`, `LIMIT ALL` and any
  non-literal expression give `max_rows`. For a `UNION` the limit sits on the whole union, not on
  its last branch.
- The returned string is valid PostgreSQL that parses again with sqlglot.

## Chart requirements

- `dataframe_to_chart` returns `orchestrator.api_contract.ChartSpec`. Import it; leave its
  definition there.
- Chart type when `chart_type` is `None`:
  - a date or datetime `x` column with at least one numeric `y` column gives `"line"`;
  - exactly one text column plus at least one numeric column gives `"bar"`;
  - anything else gives `"table"`.
  "Date or datetime" covers pandas datetime columns and object columns of `datetime.date` or
  `datetime.datetime` values.
- An explicit `chart_type` outside `bar`, `line`, `pie`, `table`, an `x` that names a missing
  column, or a `y` entry that names a missing column raises `ValueError`. `"pie"` takes exactly one
  `y` column; any other count raises `ValueError`.
- `y` defaults to the numeric columns. `x` defaults to the first non-numeric column, and to `None`
  for `"table"`. `ChartSpec.x_field` and `y_fields` carry the chosen names.
- `data` is the list of rows (a dict per row, keyed by column name, holding every column of the
  DataFrame), limited to the first 500 rows.
- Every value is JSON-safe: numpy integers become `int`; numpy floats become `float`; NaN, NaT and
  `None` become `None`; `datetime.date`, `datetime.datetime` and `pandas.Timestamp` become ISO 8601
  strings (`isoformat()`); `Decimal` becomes `float`; anything else becomes `str`.
- `title` is passed through; it defaults to `""`.

## Data

- No data files. Unit tests build small pandas DataFrames and SQL strings inside the test files.
- The chart tests import `orchestrator.api_contract`; pull `main` once that file has landed.
- Tests check a returned SQL string by parsing it again with `sqlglot.parse_one(sql,
  read="postgres")` and reading the limit value and the table names from the tree. They never
  compare whole SQL strings.

## Tests to write first

Write these in `database/charts/tests/` before the code, run them, see them fail, then write the
code until they pass. Use `test_sql_guard.py` for the guard and `test_sql_to_chart.py` for the
charts, and share helpers in one `conftest.py`.

Attack strings (parametrized in one test, each must raise `UnsafeSQLError`, and the error's
`reason` is a non-empty `str`); at least these 41:

```text
SELECT 1; DROP TABLE vendors
SELECT * FROM vendors; SELECT * FROM payments
SELECT 1 /* x */; DROP TABLE vendors
SELECT 1 -- x<newline>; DELETE FROM vendors
/* SELECT 1 */ DROP TABLE vendors
sElEcT 1; dRoP tAbLe vendors
SELECT 1；DROP TABLE vendors            (fullwidth semicolon U+FF1B)
DROP TABLE vendors
DELETE FROM vendors
UPDATE vendors SET name = 'x'
INSERT INTO vendors (name) VALUES ('x')
TRUNCATE vendors
ALTER TABLE vendors ADD COLUMN x int
CREATE TABLE t (id int)
GRANT ALL ON vendors TO PUBLIC
COPY vendors TO PROGRAM 'cat /etc/passwd'
CALL do_something()
DO $$ BEGIN PERFORM 1; END $$
SELECT pg_sleep(10)
SELECT pg_catalog.pg_sleep(10)
SELECT PG_SLEEP(10)
SELECT pg_read_file('/etc/passwd')
SELECT pg_read_binary_file('/etc/passwd')
SELECT lo_import('/etc/passwd')
SELECT lo_export(1, '/tmp/x')
SELECT * FROM dblink('host=x', 'SELECT 1') AS t(a int)
SELECT dblink_exec('host=x', 'DROP TABLE vendors')
SELECT set_config('log_statement', 'all', false)
SELECT current_setting('server_version')
SELECT * FROM vendors WHERE id = (SELECT length(pg_read_file('/etc/passwd')))
SELECT id FROM vendors UNION SELECT usename FROM pg_catalog.pg_user
SELECT id FROM vendors UNION SELECT table_name FROM information_schema.tables
WITH x AS (DELETE FROM vendors RETURNING *) SELECT * FROM x
SELECT * INTO newt FROM vendors
SELECT * FROM secrets
SELECT * FROM vendors WHERE id IN (SELECT id FROM secrets)
SELECT * FROM other.vendors
(empty string)
(three spaces)
SELEC * FROM vendors
SELECT * FROM (
```

Passing cases and the checks on the returned SQL (`max_rows=100` where a number is shown):

| input | expected | catches |
|---|---|---|
| `SELECT * FROM vendors` | tables `{vendors}`, limit 100 | no limit appended |
| `SELECT * FROM vendors LIMIT 10` | limit 10 | clamping every limit to `max_rows` |
| `SELECT * FROM vendors LIMIT 100` | limit 100 | an off-by-one at the boundary |
| `SELECT * FROM vendors LIMIT 101` | limit 100 | keeping a larger limit |
| `SELECT * FROM vendors LIMIT ALL` | limit 100 | treating `ALL` as a literal |
| `SELECT * FROM vendors LIMIT 1 + 1` | limit 100 | trusting a non-literal expression |
| `SELECT id FROM vendors UNION SELECT id FROM payments` | the root node is the union with limit 100; neither branch has a limit | appending to the last branch only |
| `SELECT id FROM vendors UNION SELECT id FROM payments LIMIT 10` | union limit 10 | limiting only the last branch |
| `WITH t AS (SELECT * FROM vendors) SELECT * FROM t`, allowed `{"vendors"}` | passes; tables outside CTE names are `{vendors}` | treating the CTE name `t` as a disallowed table |
| `SELECT v.id FROM vendors v JOIN payments p ON p.vendor_id = v.id` | passes | rejecting joins or aliases |
| `SELECT * FROM public.vendors` | passes | rejecting the `public` schema |
| `SELECT lower(name), count(*), now() FROM vendors GROUP BY 1` | passes | a deny list that is too broad |

Also: the input `SELECT * FROM vendors` with `max_rows=500` (the default) gives limit 500; a call
never raises an exception other than `UnsafeSQLError` for any attack string above.

`dataframe_to_chart` (frames are built in the test):

| input | expected | catches |
|---|---|---|
| `day` = datetimes 2024-01-01 and 2024-01-02, `amount` = 1.5, 2.5 | type `line`, `x_field="day"`, `y_fields=["amount"]`, data `[{"day": "2024-01-01T00:00:00", "amount": 1.5}, {"day": "2024-01-02T00:00:00", "amount": 2.5}]` | timestamps left as objects |
| the same with `day` an object column of `datetime.date` | type `line`, first `day` value `"2024-01-01"` | detecting only pandas datetimes |
| `vendor` = `["a", "b"]`, `total` = `[10, 20]`, `n` = `[1, 2]` | type `bar`, `x_field="vendor"`, `y_fields=["total", "n"]` | a single `y` default |
| `a` = `["x"]`, `b` = `["y"]`, `n` = `[1]` | type `table` | accepting two text columns as `bar` |
| `a` = `[1]`, `b` = `[2]` | type `table`, `x_field=None`, `y_fields=["a", "b"]` | an invented `x` |
| `day` datetimes only, no numeric column | type `table` | `line` without a numeric `y` |
| the bar frame with `chart_type="table"` | type `table` | ignoring an explicit type |
| `vendor` text and one numeric column, `chart_type="pie"` | type `pie`, one `y` field | rejecting valid `pie` |
| the bar frame (two numeric columns), `chart_type="pie"` | `ValueError` | allowing several `y` for `pie` |
| the bar frame, `chart_type="pie"`, `y=["total"]` | type `pie`, `y_fields=["total"]` | ignoring `y` |
| `chart_type="scatter"` | `ValueError` | passing unknown types through |
| `x="nope"`, then `y=["nope"]` | `ValueError` each | a `KeyError` or a silent default |
| 501 rows, `n` = 0..500 | `len(data) == 500`, first `n` is 0, last `n` is 499 | no limit, or the last 500 rows |
| `v` = `[1.0, float("nan")]` | second `v` is `None` | `NaN` in the JSON |
| a datetime column holding `pd.NaT` | that value is `None` | `"NaT"` as a string |
| `n` = `np.int64` values | `type(value) is int`, and `chart.model_dump_json()` succeeds | numpy scalars in the payload |
| `v` = `np.float64(2.5)` | `type(value) is float` | |
| `Decimal("1.50")` | `1.5`, a `float` | `Decimal` or a string in the payload |
| `uuid.UUID(int=1)` | `"00000000-0000-0000-0000-000000000001"` | an object that is not JSON |
| `title="Payments"` | `chart.title == "Payments"`; default is `""` | dropping the title |
| result of any case above | an instance of `orchestrator.api_contract.ChartSpec` | redefining a look-alike class |

This task has no PostgreSQL test and no downloaded data, so it has no `integration` or
`needs_data` marker. Use no `pytest.skip`, `skipif`, `importorskip` or `xfail` anywhere.

## Rules

- Tests first, with pytest. Unit tests make no network calls.
- `uv run ruff check .` and `uv run ruff format .` are clean.
- Windows-safe paths: `pathlib.Path`, `encoding="utf-8"` on every text read and write.
- Secrets stay in your local `.env`, which is never committed. This task needs none.
- `UnsafeSQLError.reason` is one friendly line; it leaves out the rejected SQL and any stack trace.
- Conventional commit messages, for example `feat(charts): add ensure_read_only`.

## Definition of done

- `uv run pytest database/charts` passes.
- The repository gate passes: `uv run python scripts/gate.py`.
- `git status` shows changes only under the allowed paths; no skipped or xfail tests; no secrets.

## Pull request

1. Push the branch: `git push -u origin m4/sql-guard-charts`. Without write access to the repository,
   fork it on GitHub, push to your fork and open the pull request from there.
2. Open a draft pull request to `main` by 2026-11-30, for example
   `gh pr create --draft --base main --title "feat(m4): SQL guard and query-result-to-chart JSON"`.
3. Fill in the pull request template: task id M4; what changed; what the agent generated and what
   the contributor checked by hand (for example: read `sql_guard.py` line by line, tried extra
   attack strings, checked the chart JSON in a browser); how to run it; the checklist.
4. Keep each pull request to one concern and under about 400 changed lines; for example, the guard
   first and the chart conversion second.
5. Mark it ready for review by 2026-12-07. CI must be green, and the code owner's approval is
   required before it merges.
