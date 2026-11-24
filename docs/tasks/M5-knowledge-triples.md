# Task M5: Extract knowledge triples from the 10-K dataset

Paste this whole file into your coding agent (Claude Code, Codex, Cursor or Copilot) as the first
message. It holds everything the task needs.

## Role and goal

You are a Python data engineer in the SovereignFin-3M repository, a local-first financial
question-answering engine built as a university project. You work for a contributor (your user),
who reviews your work and opens the pull request.

Goal: implement an extractor that turns the cleaned 10-K contexts into knowledge-graph triples
(`Triple` objects) and writes them to `data/kg/triples_10k.parquet`. Every triple must be grounded:
its tail appears verbatim in the context it came from.

Work in this order: Setup, Tests to write first, code, Definition of done, Pull request.

## Setup

Requires Python 3.13 and uv (https://docs.astral.sh/uv/). `uv python install 3.13` installs Python
if it is missing.

```bash
git clone https://github.com/afel6/SovereignFin-3M.git
cd SovereignFin-3M
git pull
git switch -c m5/knowledge-triples
uv sync --group kg --group dev
```

If you cloned earlier, run `git switch main` and `git pull` before you create the branch. Then read
`AGENTS.md` at the repository root, when it exists, and follow it. Run `git pull` on `main` again
at the start of each work session and merge `main` into your branch, so new shared files arrive.

## Allowed paths

You create and change files only under these paths:

- `atlas_kv/contrib/`

Every other path is off-limits: other folders, every `api.py`, `orchestrator/api_contract.py`,
`pyproject.toml`, `uv.lock`, `.github/`, `AGENTS.md`, `CLAUDE.md`. A CI check rejects a pull request
that changes any file outside the allowed paths. When you need a new dependency or a contract
change, stop and ask the contributor to open an issue.

## Files to commit

- `atlas_kv/contrib/__init__.py` — marks the package
- `atlas_kv/contrib/extract_triples.py` — extraction, grounding filter and CLI
- `atlas_kv/contrib/samples/triples_sample.csv` — at most 200 extracted rows, columns = the `Triple` fields, UTF-8, `"\n"` line endings
- `atlas_kv/contrib/tests/` — the tests (`test_*.py`)

## Runtime outputs (never commit)

- `data/kg/triples_10k.parquet` and any other folder under `data/` — git-ignored; rebuilt by the
  CLI in the Extraction requirements section.

## Contract

These blocks are copied verbatim from `docs/contracts.md` in the repository. Build to them exactly
and leave them unchanged. When one looks wrong, stop and ask the contributor to open an issue.

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

```contract knowledge-api
Module: atlas_kv/api.py

class Triple(BaseModel):
    head: str
    relation: Relation    # enum in atlas_kv/schema.py, see "Triple vocabulary"
    tail: str
    ticker: str | None
    source: Literal["synthetic_db", "tenk", "reference"]
    source_id: str
    confidence: float     # 0..1
    extractor: str

class Fact(BaseModel):
    head: str
    relation: str
    tail: str
    source: str
    score: float

def lookup(query: str, tickers: list[str] | None = None, k: int = 8) -> list[Fact]
```

```contract triple-vocabulary
Relation enum (atlas_kv/schema.py), non-numeric relations only:
  operates_in_sector, headquartered_in, has_segment, offers_product, competes_with, faces_risk,
  uses_technology, has_subsidiary, listed_on, supplied_by, pays_late_to, shares_vendor_with
Rules:
  The tail is a non-numeric string.
  For source="tenk": source_id is the row_id of a cleaned row; the casefolded tail appears
  verbatim in the casefolded cleaned context of that row; the head is that row's company_name.
Files (one row per Triple, columns = the Triple fields, relation stored as its string value):
  data/kg/core_triples.parquet   built from the synthetic database and tickers.csv
  data/kg/triples_10k.parquet    extracted from the cleaned 10-K dataset (task M5)
```

```contract llm-client
Module: orchestrator/llm.py. Any folder may import orchestrator.llm and
orchestrator.api_contract; no other orchestrator module.

def complete_text(system: str, user: str, *, tier: Literal["fast", "strong"] = "fast",
                  mode: Literal["cloud", "local"] = "cloud", max_tokens: int = 1024) -> str

def complete_json(system: str, user: str, schema: type[T], *,
                  tier: Literal["fast", "strong"] = "fast",
                  mode: Literal["cloud", "local"] = "cloud") -> T
    # T is a pydantic model. Uses the provider's JSON mode or guided JSON when supported,
    # otherwise prompt + parse; validates with pydantic; 1 repair retry.

Errors: LLMError with kind one of "not_configured", "rate_limited", "timeout", "bad_output", "http".
Tests: SF_LLM_PROVIDER=stub selects StubLLM, which returns scripted responses and makes no
network calls.
Tiers: SF_LLM_FAST_MODEL (plan, route, extract); SF_LLM_STRONG_MODEL (synthesis).
```

```contract llm-variables
# nvidia (default) | openai_compatible | stub
SF_LLM_PROVIDER=nvidia
# free key from the NVIDIA build API; used when SF_LLM_PROVIDER=nvidia
NVIDIA_API_KEY=
# any OpenAI-compatible endpoint; used when SF_LLM_PROVIDER=openai_compatible
SF_LLM_BASE_URL=
SF_LLM_API_KEY=
# empty = local mode is not configured
SF_LLM_LOCAL_MODEL=
# empty = the client's default model for the provider
SF_LLM_FAST_MODEL=
SF_LLM_STRONG_MODEL=
OLLAMA_HOST=http://127.0.0.1:11434
```

## Extraction requirements

These define the part of the task that the contract leaves to you.

- Import `Triple` from `atlas_kv.api` and `Relation` from `atlas_kv.schema`. Never redefine
  either. Both arrive on `main` around 2026-11-25; until then, use temporary stand-ins only in
  your local working copy and never commit them.
- Extract with the LLM through `orchestrator.llm.complete_json`, passing a pydantic schema for the
  model's output, or with rules, or with both. Keep `relation` a plain `str` in the model's output
  schema, so the grounding filter rejects and counts an unknown relation instead of a validation
  error losing it.
- The `extractor` field names the method, for example `"llm:<model name>"` or `"rules:v1"`.
- The LLM call is a parameter of the extraction function, with the signature of `complete_json`,
  so tests pass a fake. The extraction function takes the cleaned DataFrame and returns the
  accepted `Triple` list, the rejection counts and the per-row error counts.
- Fixed fields of every triple: `source="tenk"`, `source_id` = the row's `row_id`, `head` = that
  row's `company_name` (replace whatever head the model returned), `ticker` = the row's `ticker`.
- The grounding filter applies to every candidate, whatever produced it. A candidate is accepted
  only when all of these hold:
  1. `relation` is a value of `Relation`.
  2. `tail` is non-empty after stripping; the stored tail is the stripped text.
  3. `tail` contains no digit (the strict reading of "non-numeric").
  4. The casefolded tail appears verbatim in the casefolded cleaned `context` of that row.
  5. `confidence` is within 0 to 1, both ends included.
- Count every rejected candidate by reason (use the names `unknown_relation`, `empty_tail`,
  `numeric_tail`, `tail_not_in_context`, `confidence_out_of_range`) and print the counts at the
  end of the run.
- Most 10-K facts fit the first nine relations of the vocabulary. `supplied_by`, `pays_late_to`
  and `shares_vendor_with` come from the synthetic database; a 10-K run rarely produces them.
- CLI:
  `uv run python -m atlas_kv.contrib.extract_triples --cleaned data/processed/cleaned_10k_reports.parquet --out data/kg/triples_10k.parquet [--limit N] [--delay-seconds S] [--sample-out atlas_kv/contrib/samples/triples_sample.csv]`.
  `--limit N` processes the first N rows; `--sample-out` writes at most 200 of the extracted rows
  to that CSV.
- The delay between LLM calls defaults to a named constant (for example
  `DEFAULT_DELAY_SECONDS`). Set it from the provider's published free-tier rate limit and note
  that limit in a comment next to the constant.
- `main(argv)` returns the exit code. An `LLMError` of kind `"not_configured"` stops the run with
  one friendly line (no stack trace, no key value) and exit code 1. Any other `LLMError` kind for
  one row is counted by kind and the run continues; the final summary line reports accepted
  triples, rejected candidates by reason and errors by kind.
- The parquet output has one row per `Triple`, columns = the `Triple` fields, `relation` stored as
  its string value, and reads back into `Triple` objects.

## Data

- Real data, once the data layer is on `main` (planned for 2026-11-27):
  `uv run python -m data_pipeline.download` fetches the raw file into `data/raw/`, and
  `uv run python -m data_pipeline.contract.build_reference --out-dir data/processed` writes
  `data/processed/cleaned_10k_reports.parquet`.
- Until then, unit tests build small pandas DataFrames inside the test file, with the cleaned
  columns of the schema above (at least `row_id`, `ticker`, `company_name`, `context`).
- LLM key: each contributor uses their own free `NVIDIA_API_KEY` (from https://build.nvidia.com)
  in a local `.env`, only for the live run. Never commit it, print it or log it. Tests set
  `SF_LLM_PROVIDER=stub` with `monkeypatch.setenv` and inject fake LLM functions; no test reaches
  the network. Never write a realistic-looking fake key anywhere (CI runs a secret scan); build
  fake key strings in tests by concatenation, for example `"fake" + "-" + "key"`.

## Tests to write first

Write these in `atlas_kv/contrib/tests/` before the code, run them, see them fail, then write the
code until they pass. Every expected value below follows from the Contract and Extraction
requirements sections. Build a one-row cleaned DataFrame with `company_name="Acme Corp"`,
`ticker="ACME"` and the `context` `"Acme sells cloud computing services in Texas."`; the fake LLM
returns one candidate per case.

Grounding filter (one candidate each, `relation="offers_product"` and `confidence=0.9` unless
stated):

| tail / change | expected | catches |
|---|---|---|
| `"cloud computing services"` | accepted | |
| `"Cloud Computing"` | accepted, tail stored as `"Cloud Computing"` | comparing without casefolding |
| `"  Texas  "` | accepted, stored tail `"Texas"` | storing unstripped tails |
| `"quantum networks"` (not in context) | rejected, `tail_not_in_context` | trusting the model |
| `"$4.2 billion"` | rejected, `numeric_tail` | checking only digits-only tails |
| `"2023"` | rejected, `numeric_tail` | allowing year-like tails |
| `"   "` | rejected, `empty_tail` | keeping blank tails |
| `relation="sells_stuff"` | rejected, `unknown_relation` | silently accepting free-text relations |
| `confidence=1.5` | rejected, `confidence_out_of_range` | not checking the range |
| `confidence=-0.1` | rejected, `confidence_out_of_range` | checking only the upper bound |
| `confidence=0` and `confidence=1` | both accepted | an exclusive range |
| `head="Wrong Inc"` | accepted, head `"Acme Corp"` | keeping the model's head |

Also assert on an accepted triple: `source == "tenk"`, `source_id` equals the row's `row_id`,
`ticker == "ACME"` and `extractor` is non-empty. In a run with one accepted and two rejected
candidates, the rejection counts sum to 2.

Run behaviour:

- A fake LLM that raises `LLMError("rate_limited")` for the second of three rows: the run still
  returns the triples of rows one and three, and the error counts show `rate_limited: 1`.
  This catches a loop that stops at the first error.
- A fake LLM that raises `LLMError("not_configured")`: `main([...])` on a small cleaned parquet
  file in `tmp_path` returns 1, and `capsys` shows exactly one non-empty line with no traceback
  text. This catches a crash with a stack trace.
- `--limit 2` on a three-row file calls the fake LLM twice.
- `--delay-seconds 0` makes no pause: inject or monkeypatch the sleep function, and never wait for
  real.

Outputs:

- The output parquet written by `main` reads back, row by row, into valid `Triple` objects equal
  to the accepted ones, with `relation` stored as a string.
- The committed `atlas_kv/contrib/samples/triples_sample.csv` has exactly the `Triple` field names
  as header columns and at most 200 data rows. Read it with `encoding="utf-8"` and check that the
  bytes contain no `"\r"`.

Data tests (marker `needs_data`; they need the cleaned parquet from the Data section and a live
key, so they run only on your machine): the CLI with `--limit 20` on the real cleaned file writes a
parquet file whose every tail passes the grounding filter against its row's context.

Mark these tests with `@pytest.mark.integration` or `@pytest.mark.needs_data`. Both are
deselected by default. CI's integration job runs the `integration` tests with a PostgreSQL service
only: no downloaded data and no other services. `needs_data` tests run only on your machine. Run
both locally with `uv run pytest -m "integration or needs_data" atlas_kv/contrib`. Use no
`pytest.skip`, `skipif`, `importorskip` or `xfail` anywhere.

### Live run

This is a step on the contributor's machine, not a test. With your own `NVIDIA_API_KEY` in `.env`,
run the CLI on the real cleaned data, with `--sample-out` set to
`atlas_kv/contrib/samples/triples_sample.csv`, and commit that sample. Pick 20 sample rows at
random, check each by hand against its context, and report the precision (correct rows out of 20)
in the pull request body.

## Rules

- Tests first, with pytest. Unit tests make no network calls.
- `uv run ruff check .` and `uv run ruff format .` are clean.
- Windows-safe paths: `pathlib.Path`, `encoding="utf-8"` on every text read and write, CSV
  written with `lineterminator="\n"`.
- Secrets stay in your local `.env`, which is never committed. Never print or log an API key.
- CLI errors are one friendly line and a non-zero exit code; no stack trace for expected failures.
- Conventional commit messages, for example `feat(kg): add the 10-K triple extractor`.

## Definition of done

- `uv run pytest atlas_kv/contrib` passes.
- The integration and data tests pass locally:
  `uv run pytest -m "integration or needs_data" atlas_kv/contrib`.
- The live run is done, the sample is committed, and the precision of 20 hand-checked rows is in
  the pull request body.
- The repository gate passes: `uv run python scripts/gate.py`.
- `git status` shows changes only under the allowed paths; no skipped or xfail tests; no secrets.

## Pull request

1. Push the branch: `git push -u origin m5/knowledge-triples`. Without write access to the
   repository, fork it on GitHub, push to your fork and open the pull request from there.
2. Open a draft pull request to `main` by 2026-11-30, for example
   `gh pr create --draft --base main --title "feat(m5): extract knowledge triples from the 10-K dataset"`.
3. Fill in the pull request template: task id M5; what changed; what the agent generated and what
   the contributor checked by hand (for example: read the grounding filter line by line, checked
   20 sample rows against their contexts); how to run it; the checklist.
   CI does not run `needs_data` tests: run them with `uv run pytest -m needs_data atlas_kv/contrib`
   and say in the body that you ran them and that they passed.
4. Keep each pull request to one concern and under about 400 changed lines; for example, the
   extractor and filter first and the CLI and sample second.
5. Mark it ready for review by 2026-12-07. CI must be green, and the code owner's approval is
   required before it merges.
