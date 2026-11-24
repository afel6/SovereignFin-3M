# Task M2: Clean the 10-K dataset and load it into PostgreSQL

Paste this whole file into your coding agent (Claude Code, Codex, Cursor or Copilot) as the first
message. It holds everything the task needs.

## Role and goal

You are a Python data engineer in the SovereignFin-3M repository, a local-first financial
question-answering engine built as a university project. You work for a contributor (your user),
who reviews your work and opens the pull request.

Goal: implement the cleaning step that turns the raw 10-K question/answer dataset into the cleaned
dataset defined in the Contract section, exactly, and a loader that puts the cleaned rows into a
PostgreSQL table `tenk_snippets`.

Work in this order: Setup, Tests to write first, code, Definition of done, Pull request.

## Setup

Requires Python 3.13 and uv (https://docs.astral.sh/uv/). `uv python install 3.13` installs Python
if it is missing.

```bash
git clone https://github.com/afel6/SovereignFin-3M.git
cd SovereignFin-3M
git pull
git switch -c m2/data-cleaning
uv sync --group data --group dev
```

If you cloned earlier, run `git switch main` and `git pull` before you create the branch. Then read
`AGENTS.md` at the repository root, when it exists, and follow it. Run `git pull` on `main` again
at the start of each work session and merge `main` into your branch, so new shared files arrive.

## Allowed paths

You create and change files only under these paths:

- `data_pipeline/clean/`
- `database/tenk/`

Every other path is off-limits: other folders, every `api.py`, `orchestrator/api_contract.py`,
`pyproject.toml`, `uv.lock`, `.github/`, `AGENTS.md`, `CLAUDE.md`. A CI check rejects a pull request
that changes any file outside the allowed paths. When you need a new dependency or a contract
change, stop and ask the contributor to open an issue.

## Files to commit

- `data_pipeline/clean/__init__.py` — exports `normalize_text` and `clean`
- `data_pipeline/clean/__main__.py` — the cleaning CLI
- `data_pipeline/clean/normalize.py` — `normalize_text`
- `data_pipeline/clean/clean.py` — `clean`: drop rules, schema, derived columns
- `data_pipeline/clean/load_data.py` — the PostgreSQL loader and its CLI
- `data_pipeline/clean/tests/` — the tests (`test_*.py`)
- `database/tenk/tenk_snippets.sql` — the table definition

## Runtime outputs (never commit)

- `data/processed/` and any other folder under `data/` — git-ignored; the cleaned files are rebuilt
  by the commands in the Data section.

## Contract

These blocks are copied verbatim from `docs/contracts.md` in the repository. Build to them exactly
and leave them unchanged. When one looks wrong, stop and ask the contributor to open an issue.

```contract raw-input
File:     data/raw/financial_qa_10k.parquet
Source:   Hugging Face dataset virattt/financial-qa-10K, train file
sha256:   4b71e8cb8d0ce7a01eb117d3ef55afa994ab47571728ac8b2fa052e061d0a066
Rows:     7000
Columns:  exactly {question, answer, context, ticker, filing}
raw_index = the 0-based row position in this file.
Download: uv run python -m data_pipeline.download   (fetches the file and checks the sha256)
```

```contract cleaned-output-schema
File: data/processed/cleaned_10k_reports.parquet
Sorted by raw_index. No nulls. Columns in this order:

column            type   rule
row_id            str    f"{raw_index:04d}-" + first 12 hex characters of sha256 of the UTF-8 bytes of
                         the RAW question, answer, context, ticker, filing joined by U+001F
raw_index         int32
ticker            str    raw value unchanged (keeps BRK-A)
company_name      str    joined from data_pipeline/reference/tickers.csv on ticker
filing            str    raw value unchanged; derive no fiscal year from it
question          str    N(raw question)
answer            str    N(raw answer)
context           str    N(raw context)
context_id        str    first 16 hex characters of sha256 of the UTF-8 bytes of the cleaned context
qgroup_id         str    row_id of the lowest-raw_index kept row with the same ticker and the same
                         casefolded cleaned question
is_table          bool   the cleaned context contains at least 2 "|" characters
mentions_company  bool   the casefolded question contains a casefolded alias of its own ticker, where
                         the match is not preceded or followed by [a-z0-9]; aliases are names, never
                         1-letter tickers

N is the normalization in "Text normalization".
The parquet file is written with an explicit pyarrow schema (pa.string(), pa.int32(), pa.bool_()).
Logical types are compared, not pandas dtypes.
data_pipeline/reference/tickers.csv has the columns ticker,company_name,sector,aliases, with aliases
separated by "|".
```

```contract text-normalization
N(s) applies these steps in this order:
1. Delete every code point whose Unicode category is Cc or Cf and that is not White_Space
   (this removes U+0099, U+200B, U+00AD, U+FEFF).
2. Map every Pd (dash punctuation) character and U+2212 to "-"; map U+2018 and U+2019 to "'";
   map U+201C and U+201D to '"'. Keep these characters as they are: • ® ™ é ¥ € § α ı
3. Remove the tag pattern: "<", an optional "/", one ASCII letter or ".", then zero or more
   characters other than "<" and ">", then ">". Repeat until no match is left
   (raw row 5828 contains "</...>").
4. Collapse every run of these White_Space code points to one space, then strip leading and
   trailing spaces: U+0009-U+000D, U+0020, U+0085, U+00A0, U+1680, U+2000-U+200A, U+2028, U+2029,
   U+202F, U+205F, U+3000. Use this explicit list, not str.isspace() or \s: those also match
   U+001C-U+001F, which step 1 already deleted.
5. Apply Unicode NFC. Not NFKC.
No case folding, no "?" added, no table reformatting, no truncation.
```

```contract drop-rules
A raw row is dropped by the first rule that matches, checked in this order:
1. empty_field       question, answer or context is empty after N (raw rows 1727, 1733, 2731)
2. non_english_text  any raw field contains a code point in U+0400-U+052F (raw rows 2253, 6471)
3. exact_duplicate   same ticker, filing, question, answer and context (after N) as a kept row
                     with a lower raw_index
Every other row is kept: paraphrase duplicates, repeated contexts, the second JNJ batch, tables
and long contexts. About 6988 rows are kept.

Drop log: data/processed/drop_log.csv with the columns raw_index, row_id, reason, duplicate_of
(duplicate_of = the row_id of the kept row that an exact_duplicate row repeats).
```

```contract cleaning-interface
Package data_pipeline.clean (folder data_pipeline/clean/) exposes:

def normalize_text(s: str) -> str
    N(s), exactly as in "Text normalization".
def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]
    Takes the raw frame; returns (cleaned, drop_log) as in "Cleaned output schema" and "Drop rules".
CLI: python -m data_pipeline.clean --raw PATH --out-dir DIR
    Writes DIR/cleaned_10k_reports.parquet and DIR/drop_log.csv.

The package must not import data_pipeline.contract (checked on the syntax tree) and makes no
network calls.
```

```contract validator-checks
Command: uv run python -m data_pipeline.contract.validate --cmd "python -m data_pipeline.clean"
It runs the given command on the pinned raw file into a temporary directory and checks:
V1   raw input: sha256 equals the pin; 7000 rows; columns exactly as listed; every filing
     matches ^\d{4}_10K$
V2   determinism: two runs give identical frames and identical drop logs
V3   schema: exact column list, order and types; no nulls; sorted by raw_index; row_id is unique
     and matches ^\d{4}-[0-9a-f]{12}$
V4   provenance: every row_id is recomputed from the raw row at its raw_index
V5   partition: kept raw_index values plus drop-log raw_index values equal {0..6999}, with no
     overlap; every drop reason is recomputed; duplicate_of points to a kept row with identical
     normalized content
V6   text: every cleaned field equals N(raw field) exactly, with no tolerance
V7   derived columns: company_name, context_id, qgroup_id, is_table and mentions_company are
     recomputed exactly
V8   properties: no Cc/Cf; no Pd other than "-"; none of U+2212, U+2018, U+2019, U+201C, U+201D,
     U+0400-U+052F; no double space; no leading or trailing space; no tag pattern; stable under NFC
V9   normalize_text: idempotent on every raw field, and equal to the reference N on further
     edge-case strings
V10  the kept row count lies in a sanity band around the expected count
V11  changes stay inside the task's allowed paths
Import ban: the cleaning package does not import data_pipeline.contract.
```

```contract database-roles
sf_owner   (DATABASE_URL_OWNER)  offline jobs only: schema setup, CSV loader, 10-K snippet loader,
                                 anomaly_scores writer
sf_reader  (DATABASE_URL)        all query-time code; SELECT grants only; read-only transactions by
                                 default; statement_timeout 5 s set on the role; default privileges
                                 give it SELECT on tables sf_owner creates later
Role passwords come from .env, never from SQL files.
```

```contract database-variables
# query time, read-only role
DATABASE_URL=postgresql://sf_reader:...@127.0.0.1:5432/sovereignfin
# offline loaders and writers only
DATABASE_URL_OWNER=postgresql://sf_owner:...@127.0.0.1:5432/sovereignfin
# database rebuild only
DATABASE_URL_ADMIN=postgresql://postgres:...@127.0.0.1:5432/postgres
POSTGRES_PASSWORD=
SF_OWNER_PASSWORD=
SF_READER_PASSWORD=
```

## Loader requirements

These define the part of the task that the contract leaves to you.

- `database/tenk/tenk_snippets.sql` creates the table with `CREATE TABLE IF NOT EXISTS
  tenk_snippets`: `row_id text PRIMARY KEY`, then every other cleaned column in schema order, all
  `NOT NULL`, typed `raw_index integer`, `is_table boolean`, `mentions_company boolean`, the rest
  `text`.
- `load_data.py` exposes `load(parquet_path: Path, db_url: str) -> int`, which applies
  `tenk_snippets.sql`, upserts every row of the parquet file in one transaction and returns the
  number of rows written.
- The upsert is `INSERT ... ON CONFLICT (row_id) DO UPDATE SET <each other column> =
  EXCLUDED.<column>`, with values passed as parameters (psycopg `executemany`, or `COPY` into a
  temporary table followed by one `INSERT ... SELECT ... ON CONFLICT`). Loading the same file twice
  leaves the table unchanged.
- CLI: `uv run python -m data_pipeline.clean.load_data --parquet PATH`. It reads the URL from
  `DATABASE_URL_OWNER` (the read-only `DATABASE_URL` cannot write). When the variable is missing it
  prints `DATABASE_URL_OWNER is not set.` and exits with code 1. A connection or SQL error prints
  one friendly line without the URL or password and exits with code 1.

## Data

- Real data, once the data layer is on `main` (planned for 2026-11-27):
  `uv run python -m data_pipeline.download` fetches the raw file into `data/raw/`, and
  `uv run python -m data_pipeline.contract.build_reference --out-dir data/processed` writes the
  reference cleaned output. Run your own CLI into another folder, for example
  `uv run python -m data_pipeline.clean --raw data/raw/financial_qa_10k.parquet --out-dir data/m2_out`,
  and compare the two.
- Until then, unit tests build small pandas DataFrames inside the test file, with the five raw
  columns.
- `clean` reads `data_pipeline/reference/tickers.csv` through one small helper function (for
  example `load_tickers()`); unit tests replace it with `monkeypatch` and a two-row DataFrame, so
  they run before that file lands.
- PostgreSQL: copy `.env.example` to `.env`, fill in the passwords, then `docker compose up -d` (Docker Desktop must be running)
  (PostgreSQL on 127.0.0.1:5432) and `uv run python -m database.rebuild` (creates the roles and
  tables; lands with the data layer).

## Tests to write first

Write these in `data_pipeline/clean/tests/` before the code, run them, see them fail, then write
the code until they pass. Every expected value below follows from the Contract section.

`normalize_text`:

| input | expected | catches |
|---|---|---|
| `"a​b"` | `"ab"` | treating U+200B as a space |
| `"﻿Revenue"` | `"Revenue"` | keeping the BOM |
| `"x\u0099y"` | `"xy"` | keeping C1 control characters |
| `"co­op"` | `"coop"` | mapping the soft hyphen to `-` |
| `"a\u0085b"` | `"a b"` | deleting every Cc character, including White_Space ones |
| `"a\x1fb"` | `"ab"` | using `str.isspace()` for "White_Space" |
| `"a\tb"` | `"a b"` | |
| `"− 5"` | `"- 5"` | |
| `"a–b—c‐d"` | `"a-b-c-d"` | mapping only some dashes |
| `"‘q’ “d”"` | `"'q' \"d\""` | |
| `"• ® ™ é ¥ € § α ı"` | unchanged | NFKC or ASCII folding |
| `"<b>bold</b>"` | `"bold"` | |
| `"<<a>/b>"` | `""` | removing tags in a single pass |
| `"a </...> b"` | `"a b"` | |
| `"<br/>"` | `""` | |
| `"x < y > z"` | unchanged | removing every `<...>` span |
| `"1 <2 and 3> 4"` | unchanged | allowing a digit after `<` |
| `" \t a \n b 　"` | `"a b"` | |
| `"a  b"` | `"a b"` | |
| `"é"` | `"é"` | skipping NFC |
| `"ﬁ x²"` | unchanged | applying NFKC |
| `"What is REVENUE"` | unchanged | case folding or adding `?` |

Also assert `normalize_text(normalize_text(s)) == normalize_text(s)` for every input above.

`clean` (build one raw DataFrame of about ten rows; compute expected `row_id` and `context_id` in
the test with `hashlib.sha256` from the rules, never by calling your own code):

- `row_id` uses the RAW fields: two rows that differ only by a no-break space get different
  `row_id`s, and the later one is dropped as `exact_duplicate` with `duplicate_of` = the earlier
  `row_id`.
- `context_id` uses the cleaned context.
- An answer of `"​"` or `"<b></b>"` is dropped as `empty_field`.
- A row with Cyrillic text in any field is dropped as `non_english_text`; a row that is both empty
  and Cyrillic is dropped as `empty_field` (first rule wins).
- A paraphrased question for the same ticker is kept.
- `qgroup_id`: two kept rows of one ticker whose questions differ only in case share the
  lower row's `row_id`; the same question under another ticker gets its own.
- `is_table`: a context with two `|` is `True`; with one `|` it is `False`.
- `mentions_company` with alias `Apple` for ticker `AAPL`: `"How did apple's revenue change?"` is
  `True`; `"Pineapple exports"` and `"Apple2 sales"` are `False`; another company's alias is
  `False`.
- The cleaned frame has exactly the schema columns in order, no nulls, sorted by `raw_index`,
  `raw_index` as int32; the drop log has the columns `raw_index, row_id, reason, duplicate_of`.
- Two runs of `clean` on the same input give equal frames.

CLI and rules:

- Running the CLI on a small raw parquet file in `tmp_path` writes both output files; reading the
  parquet schema with `pyarrow.parquet.read_schema` shows string, int32 and bool types.
- An `ast` scan of every `.py` file under `data_pipeline/clean/` finds no import of
  `data_pipeline.contract`.

Integration tests (marker `integration`; they need PostgreSQL and `DATABASE_URL_OWNER`):

- Loading a 3-row parquet file twice leaves exactly 3 rows; changing one answer and loading again
  updates that row and still leaves 3 rows.
- A connection with `DATABASE_URL` (read-only role) can `SELECT` from `tenk_snippets`.

Data tests (marker `needs_data`; they need the downloaded raw file):

- The CLI on the real raw file equals the reference output of `build_reference`.
- Raw rows 1727, 1733 and 2731 are dropped as `empty_field`; 2253 and 6471 as `non_english_text`.

Mark these tests with `@pytest.mark.integration` or `@pytest.mark.needs_data`. Both are
deselected by default. CI's integration job runs the `integration` tests against a PostgreSQL
service; `needs_data` tests run only on your machine. Run both locally with
`uv run pytest -m "integration or needs_data" data_pipeline/clean`. Use no `pytest.skip`,
`skipif`, `importorskip` or `xfail` anywhere.

## Rules

- Tests first, with pytest. Unit tests make no network calls.
- `uv run ruff check .` and `uv run ruff format .` are clean.
- Windows-safe paths: `pathlib.Path`, `encoding="utf-8"` on every text read and write, CSV
  written with `lineterminator="\n"`.
- SQL uses parameters only (`%s` or `%(name)s` placeholders); values are never formatted into SQL
  text.
- Secrets stay in your local `.env`, which is never committed. Never print or log a database URL.
- CLI errors are one friendly line and a non-zero exit code; no stack trace for expected failures.
- Conventional commit messages, for example `feat(clean): add normalize_text`.

## Definition of done

- `uv run pytest data_pipeline/clean` passes.
- The integration and data tests pass locally:
  `uv run pytest -m "integration or needs_data" data_pipeline/clean`.
- The validator passes:
  `uv run python -m data_pipeline.contract.validate --cmd "python -m data_pipeline.clean"`.
- The repository gate passes: `uv run python scripts/gate.py`.
- `git status` shows changes only under the allowed paths; no skipped or xfail tests; no secrets.

## Pull request

1. Push the branch: `git push -u origin m2/data-cleaning`. Without write access to the repository,
   fork it on GitHub, push to your fork and open the pull request from there.
2. Open a draft pull request to `main` by 2026-11-30, for example
   `gh pr create --draft --base main --title "feat(m2): clean the 10-K dataset and load it into PostgreSQL"`.
3. Fill in the pull request template: task id M2; what changed; what the agent generated and what
   the contributor checked by hand (for example: read `normalize.py` line by line, ran the
   validator, compared row counts); how to run it; the checklist.
   CI does not run `needs_data` tests: run them with `uv run pytest -m needs_data data_pipeline/clean`
   and say in the body that you ran them and that they passed.
4. Keep each pull request to one concern and under about 400 changed lines; for example, cleaning
   first and the loader second.
5. Mark it ready for review by 2026-12-07. CI must be green, and the code owner's approval is
   required before it merges.
