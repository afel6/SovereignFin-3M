# Task M8: Benchmark against plain RAG, and the presentation

Paste this whole file into your coding agent (Claude Code, Codex, Cursor or Copilot) as the first
message. It holds everything the task needs.

## Role and goal

You are a Python evaluation engineer in the SovereignFin-3M repository, a local-first financial
question-answering engine built as a university project. You work for a contributor (your user),
who reviews your work and opens the pull request.

Goal: measure the full system (`POST /ask`) against a plain retrieval-augmented baseline that uses
the same language model, on 30 questions with known answers; write the comparison, two charts and
a short slide deck that presents the method, the results and the limits.

Work in this order: Setup, Tests to write first, code, Definition of done, Pull request.

## Setup

Requires Python 3.13 and uv (https://docs.astral.sh/uv/). `uv python install 3.13` installs Python
if it is missing.

```bash
git clone https://github.com/afel6/SovereignFin-3M.git
cd SovereignFin-3M
git pull
git switch -c m8/benchmark
uv sync --group eval --group api --group dev
```

The `eval` group includes the retrieval packages the baseline needs. If you cloned earlier, run
`git switch main` and `git pull` before you create the branch. Then read `AGENTS.md` at the
repository root, when it exists, and follow it. Run `git pull` on `main` again at the start of each
work session and merge `main` into your branch, so new shared files arrive.

## Allowed paths

You create and change files only under these paths:

- `evaluation/benchmark/`
- `docs/presentation/`

The folder `docs/presentation/core/` belongs to the maintainer and stays untouched.

Every other path is off-limits: other folders, every `api.py`, `orchestrator/api_contract.py`,
`pyproject.toml`, `uv.lock`, `.github/`, `AGENTS.md`, `CLAUDE.md`. A CI check rejects a pull request
that changes any file outside the allowed paths. When you need a new dependency or a contract
change, stop and ask the contributor to open an issue.

## Files to commit

- `evaluation/benchmark/__init__.py` — empty package marker
- `evaluation/benchmark/questions.csv` — the 30 questions with gold answers
- `evaluation/benchmark/metrics.py` — `numeric_match`, `token_f1`, latency percentiles
- `evaluation/benchmark/run_benchmark.py` — the benchmark runner, the baseline and its CLI
- `evaluation/benchmark/results/results.csv` — one row per question and system
- `evaluation/benchmark/results/comparison.md` — the written comparison
- `evaluation/benchmark/results/latency.png` — latency chart
- `evaluation/benchmark/results/accuracy.png` — accuracy chart
- `evaluation/benchmark/tests/` — the tests (`test_*.py`)
- `docs/presentation/benchmark-slides.md` — the slide deck in Marp Markdown (or, instead of this
  file, `docs/presentation/benchmark-slides.pptx` built with python-pptx; commit one of the two)

## Runtime outputs (never commit)

- None. Every output is committed under `evaluation/benchmark/results/`. Do not commit anything
  under `data/`.

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

```contract rag-engine-api
Module: rag_engine/api.py

class Passage(BaseModel):
    context_id: str
    ticker: str
    filing: str
    text: str
    score: float          # higher = more relevant
    row_ids: list[str]    # row_ids of the cleaned rows that share this context_id

def search(query: str, tickers: list[str] | None = None, k: int = 5) -> list[Passage]
def build_index(cleaned_parquet: Path) -> int
    # offline; one point per unique context_id; never index the answer column;
    # returns the number of points written

Alternative implementations (rag_engine/contrib/rag_search.py) implement the same search and
build_index signatures and return rag_engine.api.Passage.
```

```contract ask-request
POST /ask
Body AskRequest:
  question: str          1..2000 characters after stripping leading and trailing whitespace
  mode: "local" | "cloud" = "cloud"
  thread_id: str | None  at most 64 characters from [A-Za-z0-9_-]
An invalid body returns 422 (FastAPI default error body).
```

```contract ask-response
200 AskResponse:
  answer: str
  outcome: "answered" | "not_found" | "refused" | "clarify"
  citations: list[Citation]
  chart: ChartSpec | null
  route: list["rag" | "sql" | "anomaly" | "kg"]
  latency_ms: int
  mode: "local" | "cloud"
  warnings: list[str]

Citation:
  id: str
  source: "rag" | "sql" | "anomaly" | "kg"
  ticker: str | null
  text: str              at most 500 characters

outcome meanings: answered = an answer backed by citations; not_found = the evidence does not
contain the answer; refused = the request is declined; clarify = the answer asks the user to make
the question more specific.
IDs are strings in JSON.
```

```contract timeouts
LLM: 20 s per attempt; at most 2 retries (3 attempts) on 429, 5xx or timeout, with backoff
     1 s, then 4 s.
Qdrant: 5 s.  Postgres: statement timeout 5 s.
/ask is capped at 90 s in total; past that the server returns 503 {"detail": "LLM unavailable"}.
Worst case is about 3 LLM calls x 3 attempts x 20 s.
Target: /ask p95 at most 15 s on the reference laptop, excluding 429 backoff.
```

```contract server-and-mock
The server binds 127.0.0.1:8000.
CORS allows http://localhost:5173 and http://localhost:8501 only.
Mock server: same models and status codes, canned answers chosen by keyword in the question:
  "revenue"     -> route rag, with citations
  "late"        -> route sql, with a bar chart
  "anomal"      -> route anomaly, with a table chart
  mode "local"  -> 503 {"detail": "Local mode is not configured on this machine."}
Run it: uv run uvicorn orchestrator.mock_server:app --port 8000
```

```contract llm-client
Module: orchestrator/llm.py. Any folder may import orchestrator.llm and
orchestrator.api_contract (both import only the standard library, pydantic and httpx). The engine
folders rag_engine, rdl_engine and atlas_kv import no other orchestrator module.

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

## Questions

`questions.csv` is UTF-8, written with `\n` line ends, with 30 rows and the columns `id`,
`category`, `question`, `gold_answer`, `gold_number`, `ticker`, `source`:

- Categories: 12 `numeric`, 10 `text`, 4 `sql`, 4 `anomaly`. `id` is unique.
- `gold_number` is a plain number (no `$`, no commas, no scale word) on `numeric` rows and empty on
  all others. `gold_answer` is a short sentence on every row.
- For `numeric` and `text` rows, `ticker` is the company's ticker and `source` is the `row_id` of
  the cleaned row the gold answer comes from. The gold answer is the cleaned `answer` text, or a
  short sentence that restates its figure.
- For `sql` and `anomaly` rows, `source` is a short description of the database fact, for example
  the vendor and the count that the query returns; `ticker` is empty.
- Read the gold values from the cleaned data and, for the database rows, from a read-only query
  against the synthetic database (the `DATABASE_URL` role). Never invent a gold value, and never
  write to the database.
- Ask questions that one passage or one query answers. Avoid questions that depend on a filing
  year; the dataset has one filing year per company and no page numbers. A 10-K citation is a
  `context_id` plus `row_ids`.

## Runner requirements

These define the part of the task that the contract leaves to you.

### Systems

For each question, in file order, the runner runs two systems:

- `system`: `POST /ask` with `{"question": ..., "mode": "cloud"}` through an injected
  `httpx.Client`, with a per-request timeout named as a constant (100 s, above the server's 90 s
  cap). The answer is the `answer` field of a 200 reply.
- `baseline`: the search function, called as `search(question, tickers=None, k=5)`; the
  text of the top 5 passages goes into one prompt (fixed system prompt, a constant: answer only
  from the numbered passages, say so when they do not contain the answer); then
  `orchestrator.llm.complete_text(system_prompt, user_prompt, tier="strong")`. This is the same LLM
  tier the server uses for synthesis. The search function defaults to
  `rag_engine.contrib.rag_search:search`; `--baseline-search module:function` selects any function
  with the same signature. It is loaded with `importlib`; a value without a colon, or a function
  that cannot be imported, stops the run before the first question with one friendly line and exit
  code 1.

The baseline has no database access, so it cannot answer `sql` and `anomaly` questions. Run it on
those rows anyway, and say so in `comparison.md`.

Record latency in milliseconds with `time.perf_counter()`: for `system` the whole POST, for
`baseline` the search plus the LLM call. A timeout, a connection error, a non-200 status or an
`orchestrator.llm.LLMError` is recorded in `error` as one short line (for example `timeout`,
`http 503`, `llm not_configured`), the answer is empty, and the run goes on. Errors are never
swallowed silently and never carry a key.

`--delay-seconds` pauses between LLM calls so the free tier is not exceeded: the runner calls the
injected `sleep(delay)` before every request and every baseline call except the very first. The
default is a named constant, `DEFAULT_DELAY_SECONDS`. Set it from the provider's published
free-tier rate limit (see build.nvidia.com): at least 60 divided by the requests per minute, and
write the limit and its source in a comment. When you cannot find the published limit, stop and ask
the contributor.

### Metrics

`metrics.py` is pure Python except for `numpy`:

- `numeric_match(answer: str, gold: float) -> bool`. Parse every number in the answer: strip `$`
  and `,`, accept a trailing `%`, and scale by a following word `thousand`, `million` or `billion`
  (case-insensitive). A trailing full stop after a number is not part of it. The match is true when
  any parsed value `x` satisfies `|x - g| <= 0.01 * |g|`; when `g` is 0 the match is exact
  (`x == 0`). No number in the answer gives false.
- `token_f1(prediction: str, gold: str) -> float`, SQuAD style: lowercase, strip punctuation, drop
  the words `a`, `an` and `the`, split on whitespace, then F1 of the multiset overlap. Both empty
  gives 1.0; exactly one empty gives 0.0.
- `latency_percentiles(values: list[float]) -> tuple[float, float]` returns p50 and p95 with
  `numpy.percentile` (default linear interpolation).

### Outputs

- `results.csv` has the columns, in this order, `id, category, system_name, answer, latency_ms,
  numeric_match, token_f1, error`. `system_name` is `system` or `baseline`; there are two rows per
  question. `numeric_match` is `1` or `0` on `numeric` rows and empty on the others; `token_f1` is
  computed on every row against `gold_answer`; a row with an error has an empty answer, `0`,
  and `0.0`.
- `comparison.md` holds: a header with the target (`mock server` or `real system`) and the run
  date; the model names read from `SF_LLM_STRONG_MODEL` and `SF_LLM_FAST_MODEL` (`default` when the
  variable is empty); a table of system against baseline with the numeric match rate (numeric
  rows), the mean token F1 (text rows), latency p50 and p95 in ms, and the error count; a
  per-category table (rows, mean token F1 and, for `numeric`, the match rate, for each system); and
  the note that the baseline has no database access.
- `latency.png`: latency p50 and p95 of both systems, as grouped bars. `accuracy.png`: numeric match
  rate and mean text token F1 of both systems, as grouped bars. Use matplotlib with the `Agg`
  backend, a title, axis labels in plain words and a legend. When matplotlib is missing from the
  `eval` group, stop and ask the contributor to open an issue.

CLI:

```bash
uv run python -m evaluation.benchmark.run_benchmark --questions evaluation/benchmark/questions.csv \
  --out-dir evaluation/benchmark/results [--base-url URL] [--baseline-search module:function] \
  [--delay-seconds S] [--timeout-seconds 100] [--target mock|real]
```

The base URL comes from `--base-url`, else the environment variable `SF_API_BASE`, else
`http://127.0.0.1:8000`. `--target` defaults to `real`. Create the output folder, and write every
file with `encoding="utf-8"` and `\n` line ends.

### Slides

`docs/presentation/benchmark-slides.md` (Marp front matter `marp: true`, slides separated by `---`)
has 6 to 10 slides, in this order: the problem; the system overview; the method (questions, the
baseline, the metrics); the results with the two charts (relative links to
`../../evaluation/benchmark/results/`); the limits, stated plainly (one filing year of short
snippets, no page numbers, a synthetic database, a free-tier LLM, 30 questions); the next steps.
Every number on a slide is copied from `comparison.md`. The first version says that its results
come from the mock server.

## Data

- Real data, once the data layer is on `main` (planned for 2026-11-27):
  `uv run python -m data_pipeline.download` fetches the raw file, and
  `uv run python -m data_pipeline.contract.build_reference --out-dir data/processed` writes
  `data/processed/cleaned_10k_reports.parquet`, the source of the 10-K gold answers.
- Synthetic database: copy `.env.example` to `.env`, fill in the passwords, `docker compose up -d` (Docker Desktop must be running),
  then `uv run --env-file .env python -m database.rebuild` (planned for 2026-11-27). Use it for a read-only query
  to check the `sql` and `anomaly` facts.
- Mock server: `uv run uvicorn orchestrator.mock_server:app --port 8000`, on `main` with
  `orchestrator/api_contract.py` and `orchestrator/llm.py` (planned for 2026-11-25). It answers
  by keyword, so its numbers are not a result about quality; the first version of the results
  comes from it and says so.
- Tests never start a server. The runner takes an injected `httpx.Client`; tests pass
  `fastapi.testclient.TestClient(orchestrator.mock_server.app)` (it is an `httpx.Client`) or an
  `httpx.Client` built with `httpx.MockTransport`.
- LLM key: for a live run you use your own free `NVIDIA_API_KEY` (from build.nvidia.com) in your
  local `.env`. It is never committed, printed or logged. Tests set `SF_LLM_PROVIDER=stub` with
  `monkeypatch.setenv` and inject fake functions; no test reaches the network. Never write a
  realistic-looking fake key anywhere (CI scans for them); build fake key strings in tests by
  concatenation.

## Tests to write first

Write these in `evaluation/benchmark/tests/` before the code, run them, see them fail, then write
the code until they pass. Every expected value below follows from the Contract and Runner
requirements sections.

`numeric_match(answer, gold)`:

| answer | gold | expected | catches |
|---|---|---|---|
| `"Revenue was $1.2 billion."` | `1200000000` | `True` | ignoring the scale word |
| `"about 1.21 billion"` | `1200000000` | `True` | an exact-only comparison |
| `"1.25 billion"` | `1200000000` | `False` | a tolerance above 1% |
| `"$1,200 million"` | `1200000000` | `True` | not stripping the comma |
| `"1.2 million"` | `1200000000` | `False` | ignoring the scale word's size |
| `"margin of 12%"` | `12` | `True` | not accepting `%` |
| `"it was 12."` | `12` | `True` | reading `12.` as not a number |
| `"revenue is unknown"` | `12` | `False` | |
| `"0"` | `0` | `True` | |
| `"0.5"` | `0` | `False` | a relative tolerance on a zero gold |
| `"3 and 12 and 40"` | `12` | `True` | looking only at the first number |

`token_f1(prediction, gold)`:

| prediction | gold | expected | catches |
|---|---|---|---|
| `"net income rose"` | `"net income rose"` | `1.0` | |
| `"alpha beta"` | `"gamma delta"` | `0.0` | |
| `"the cat sat"` | `"cat sat down"` | `0.8` (approximately) | counting `the` as a token, or set overlap |
| `"Cat, SAT!"` | `"cat sat"` | `1.0` | keeping case or punctuation |
| `"a b b"` | `"b"` | `2/3` (precision 1/2, recall 1; the `a` is dropped) | dropping nothing, or set overlap |
| `""` | `""` | `1.0` | |
| `""` | `"x"` | `0.0` | |

`latency_percentiles`:

- `[10, 20, 30, 40, 50]` gives p50 `30.0` and p95 `48.0` (linear interpolation).
- `[100]` gives `(100.0, 100.0)`.

`questions.csv` (read the real file):

- 30 rows; the category counts are exactly 12 `numeric`, 10 `text`, 4 `sql`, 4 `anomaly`.
- `id` values are unique; every `gold_answer` and `source` is non-empty.
- `gold_number` parses as a float on every `numeric` row and is empty on every other row.
- Every question is 1 to 2000 characters after stripping.

`run_benchmark` end to end (a small questions file in `tmp_path` with three rows, one `numeric`, one
`text`, one `sql`):

- Use `TestClient(orchestrator.mock_server.app)` for the system, a fake search function that
  returns objects with `text` and `context_id`, a fake `complete_text` that returns a fixed
  sentence, a recording sleep function, and `monkeypatch.setenv("SF_LLM_PROVIDER", "stub")`,
  `SF_LLM_STRONG_MODEL` and `SF_LLM_FAST_MODEL` set to two test names.
- `results.csv` has exactly the eight columns in order and 6 rows (two per question), with
  `system_name` values `system` and `baseline`, and an empty `error` on every row.
- The recording sleep function is called 5 times, every time with the delay, so it is never called
  before the first request (6 calls would catch a sleep before the first).
- The fake complete function receives `tier="strong"` and a prompt that contains the text of every
  fake passage, at most 5 of them when the fake search returns 7.
- `comparison.md` contains the two model names, the word `mock` (target `mock`) and the date given
  to the function; both PNG files exist and start with the PNG signature `\x89PNG\r\n\x1a\n`.
- A `httpx.MockTransport` that answers 503 with `{"detail": "LLM unavailable"}` gives `system` rows
  with `error` `http 503`, an empty answer, `numeric_match` `0` on the numeric row, and the run
  still writes all 6 rows. A fake `complete_text` that raises `LLMError` gives `baseline` rows with
  an error and the run goes on.
- A `--baseline-search` value `nocolon` makes the CLI exit with code 1 and print one line.
- Loading `json:dumps` through the loader returns the callable `json.dumps`.

Data tests (marker `needs_data`; they need the downloaded data and the cleaned parquet file):

- For every `numeric` and `text` row of the real `questions.csv`, `source` is a `row_id` in
  `data/processed/cleaned_10k_reports.parquet`, the `ticker` equals that row's ticker, and for
  `numeric` rows the `gold_number` appears, after removing commas, in the cleaned answer text of
  that row (or in its gold answer text).

Mark these tests with `@pytest.mark.needs_data`. It is deselected by default and runs only on your
machine; CI's integration job runs only `integration` tests, with a PostgreSQL service, and the
other tests here need none. Run it locally with `uv run pytest -m "integration or needs_data"
evaluation/benchmark`. Use no `pytest.skip`, `skipif`, `importorskip` or `xfail` anywhere. A time
limit in a test catches a hang only: set it far above the usual time and never assert a duration;
the sleep function is injected, so tests never wait.

## Rules

- Tests first, with pytest. Tests make no network calls and start no server.
- `uv run ruff check .` and `uv run ruff format .` are clean.
- Windows-safe paths: `pathlib.Path`, `encoding="utf-8"` on every text read and write, CSV
  written with `lineterminator="\n"`.
- SQL uses parameters only (`%s` or `%(name)s` placeholders), through the read-only role, in a
  read-only transaction. No SQL text lives in the committed code; the database facts live in
  `questions.csv` as values.
- Secrets stay in your local `.env`, which is never committed. Never print or log a key or a
  database URL; error lines carry no URL and no key.
- CLI errors are one friendly line and a non-zero exit code; no stack trace for expected failures.
- Every number with a meaning has a name (timeouts, delay, number of passages, tolerance).
- Conventional commit messages, for example `feat(benchmark): add numeric_match`.

## Definition of done

- `uv run pytest evaluation/benchmark` passes.
- The data tests pass locally: `uv run pytest -m "integration or needs_data" evaluation/benchmark`.
- `results.csv`, `comparison.md`, `latency.png` and `accuracy.png` come from a run against the mock
  server, and the slides use their numbers and say that the results come from the mock server.
- The repository gate passes: `uv run python scripts/gate.py`.
- `git status` shows changes only under the allowed paths; `docs/presentation/core/` is untouched;
  no skipped or xfail tests; no secrets; nothing under `data/`.

## Pull request

1. Push the branch: `git push -u origin m8/benchmark`. Without write access to the repository,
   fork it on GitHub, push to your fork and open the pull request from there.
2. Open a draft pull request to `main` by 2026-11-30, for example
   `gh pr create --draft --base main --title "feat(m8): add the benchmark against plain RAG and the slides"`.
3. Fill in the pull request template: task id M8; what changed; what the agent generated and what
   the contributor checked by hand (for example: checked every gold value against the cleaned data
   or the database, read `metrics.py` line by line, compared slide numbers with `comparison.md`);
   how to run it; the checklist.
   CI does not run `needs_data` tests: run them with `uv run pytest -m needs_data evaluation/benchmark`
   and say in the body that you ran them and that they passed.
4. Keep each pull request to one concern and under about 400 changed lines; for example, the
   metrics and the questions first, the runner second, the slides third. The PNG files count as
   one line each.
5. Mark it ready for review by 2026-12-07, with the results from the mock server. CI must be green,
   and the code owner's approval is required before it merges.
6. The final run against the real system is on 2026-12-15. Run the CLI again with `--target real`
   and your own `NVIDIA_API_KEY`, then open a small follow-up pull request the same day that
   updates the files under `evaluation/benchmark/results/` and the numbers in the slides.
