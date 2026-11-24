# Task M3: Baseline vector search ("plain RAG")

Paste this whole file into your coding agent (Claude Code, Codex, Cursor or Copilot) as the first
message. It holds everything the task needs.

## Role and goal

You are a Python retrieval engineer in the SovereignFin-3M repository, a local-first financial
question-answering engine built as a university project. You work for a contributor (your user),
who reviews your work and opens the pull request.

Goal: implement a plain dense-vector baseline for the `rag-engine-api` in the Contract section:
`build_index` indexes the cleaned 10-K contexts, and `search` returns the best-matching passages.
The maintainer compares this baseline with other retrieval methods, so it stays simple.

Work in this order: Setup, Tests to write first, code, Definition of done, Pull request.

## Setup

Requires Python 3.13 and uv (https://docs.astral.sh/uv/). `uv python install 3.13` installs Python
if it is missing.

```bash
git clone https://github.com/afel6/SovereignFin-3M.git
cd SovereignFin-3M
git pull
git switch -c m3/baseline-search
uv sync --group rag --group dev
```

If you cloned earlier, run `git switch main` and `git pull` before you create the branch. Then read
`AGENTS.md` at the repository root, when it exists, and follow it. Run `git pull` on `main` again
at the start of each work session and merge `main` into your branch, so new shared files arrive.

## Allowed paths

You create and change files only under these paths:

- `rag_engine/contrib/`

Every other path is off-limits: other folders, every `api.py`, `orchestrator/api_contract.py`,
`pyproject.toml`, `uv.lock`, `.github/`, `AGENTS.md`, `CLAUDE.md`. A CI check rejects a pull request
that changes any file outside the allowed paths. When you need a new dependency or a contract
change, stop and ask the contributor to open an issue.

## Files to commit

- `rag_engine/contrib/__init__.py` — exports `search` and `build_index`
- `rag_engine/contrib/rag_search.py` — `build_index`, `search`, the embedding function and the index folder
- `rag_engine/contrib/tests/` — the tests (`test_*.py`)

## Runtime outputs (never commit)

- `data/contrib_index/` — the Qdrant local-mode folder; git-ignored and rebuilt by `build_index`.
- The fastembed model cache — downloaded on first use; stays outside the repository.

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

## Search requirements

These define the part of the task that the contract leaves to you.

- Import `Passage` from `rag_engine.api`. Leave its definition there.
- Method: dense vectors only. Embed with fastembed, model `BAAI/bge-small-en-v1.5` (384
  dimensions, cosine distance). Store and query with `qdrant-client` in local mode,
  `QdrantClient(path=...)`, under `data/contrib_index/`. Close the client after each call, because
  local mode locks the folder.
- Use only what the `rag` dependency group installs. That excludes a Qdrant server, hybrid or BM25
  search, and ChromaDB. A new dependency needs an issue first.
- Module-level names in `rag_search.py`: `INDEX_DIR` (a `Path`, default `data/contrib_index`),
  `COLLECTION = "tenk_contexts"`, `VECTOR_SIZE = 384`, and a function `_embed(texts: list[str]) ->
  list[list[float]]` that wraps fastembed. Code reads these names when it runs, so tests replace
  them with `monkeypatch`.
- `build_index(cleaned_parquet)` reads the cleaned parquet file and writes one point per unique
  `context_id`. It embeds the cleaned `context` text and nothing else. The `question` and `answer`
  columns are never embedded and never stored. It recreates the collection on each call and returns
  the number of points written, which equals the number of unique `context_id` values.
- Point id: `int(context_id, 16)`. A `context_id` is 16 hex characters, so the id is a 64-bit
  unsigned integer.
- Payload: `context_id`, `ticker`, `filing`, `text` (the cleaned context), `row_ids` (every
  `row_id` that shares this `context_id`, sorted ascending). `ticker` and `filing` come from the row
  with the lowest `raw_index` among those rows.
- `search(query, tickers=None, k=5)` embeds the query, keeps only points whose `ticker` is in
  `tickers` when `tickers` is given, and returns at most `k` `Passage` objects sorted by `score`
  descending. `Passage.text` is the payload `text`; `score` is the cosine similarity Qdrant
  reports.
- The first run of the real model downloads it from the internet. That online step happens only on
  the contributor's machine, in the `needs_data` test. Unit tests never download a model and never
  touch the network.

## Data

- Real data, once the data layer is on `main` (planned for 2026-11-27):
  `uv run python -m data_pipeline.download` fetches the raw file into `data/raw/`, and
  `uv run python -m data_pipeline.contract.build_reference --out-dir data/processed` writes
  `data/processed/cleaned_10k_reports.parquet`, the input of `build_index`.
- Until then, unit tests build small pandas DataFrames inside the test file, with the cleaned
  columns, and write them to a parquet file in `tmp_path`.
- Unit tests point `INDEX_DIR` at `tmp_path / "index"` and replace `_embed` with a deterministic
  fake: for each text, a 384-dimension vector that counts lowercased whitespace-separated tokens at
  index `int(sha256(token).hexdigest()[:8], 16) % 384`, scaled to unit length. Texts that share
  words then score higher than texts that share none. Pick the words in the fixture so that no two
  of them land on the same index; the test can assert this.

## Tests to write first

Write these in `rag_engine/contrib/tests/` before the code, run them, see them fail, then write
the code until they pass. Put the fixtures (the frame, the fake embedder, the `monkeypatch` setup)
in one `conftest.py` and reuse them.

Fixture frame, 6 rows (other cleaned columns hold any valid value):

| raw_index | row_id | ticker | filing | context_id | context | answer |
|---|---|---|---|---|---|---|
| 0 | `0000-aaaaaaaaaaaa` | AAPL | 2022_10K | `a000000000000001` | `Apple sells iphones and wearable devices worldwide` | `zebracrossing is the answer` |
| 1 | `0001-bbbbbbbbbbbb` | AAPL | 2022_10K | `a000000000000001` | same as row 0 | `second answer` |
| 2 | `0002-cccccccccccc` | AAPL | 2022_10K | `b000000000000002` | `Apple services revenue includes advertising and cloud` | `third answer` |
| 3 | `0003-dddddddddddd` | MSFT | 2022_10K | `ffffffffffffffff` | `Microsoft sells cloud software and azure subscriptions` | `fourth answer` |
| 4 | `0004-eeeeeeeeeeee` | MSFT | 2022_10K | `ffffffffffffffff` | same as row 3 | `fifth answer` |
| 5 | `0005-ffffffffffff` | MSFT | 2022_10K | `d000000000000004` | `Microsoft gaming revenue grew with xbox consoles` | `sixth answer` |

Use the `context_id` column as it is; the fixture ids are not hashes.

| case | input | expected | catches |
|---|---|---|---|
| duplicate contexts collapse | `build_index` on the fixture | returns `4` | one point per row, or per ticker |
| point count in the store | count points in `COLLECTION` after the build | `4` | the return value disagreeing with the store |
| row ids grouped | `search("iphones", k=5)` | the top passage has `context_id == "a000000000000001"` and `row_ids == ["0000-aaaaaaaaaaaa", "0001-bbbbbbbbbbbb"]` | a single `row_id` per point |
| row ids sorted | the same build from the fixture with its rows in reversed order | the same `row_ids` list; for context `ffffffffffffffff`, `["0003-dddddddddddd", "0004-eeeeeeeeeeee"]` | keeping the order of arrival |
| ticker and filing from the lowest raw_index | two rows with one context: `raw_index` 7 (`BBB`, `2022_10K`), `raw_index` 2 (`AAA`, `2021_10K`), given in this order | the passage has ticker `AAA` and filing `2021_10K` | taking the first or last row seen |
| point id | after the build, retrieve the point with id `int("ffffffffffffffff", 16)` from the store | one point, payload `context_id == "ffffffffffffffff"` | a random or sequential id; a signed 64-bit overflow |
| answer never indexed | after the build, read every payload from the store; run `search("zebracrossing", k=5)` | the string `zebracrossing` is in no payload value and in no `Passage.text` | embedding or storing the answer column |
| payload fields | read one payload | exactly the keys `context_id`, `ticker`, `filing`, `text`, `row_ids`; `text` equals the context | extra columns such as `question` or `answer` in the payload |
| ticker filter | `search("revenue", tickers=["MSFT"], k=5)` | 2 passages, all with ticker `MSFT` | filtering after the top-k cut, or ignoring the filter |
| no filter | `search("revenue", k=5)` | 4 passages, tickers include both `AAPL` and `MSFT` | a default ticker filter |
| unknown ticker | `search("revenue", tickers=["ZZZZ"])` | `[]` | an error on an empty result |
| k respected | `search("cloud", k=2)` | exactly 2 passages | ignoring `k`; `search("cloud", k=10)` returns 4, not an error |
| scores descending | `search("azure subscriptions cloud", k=4)` | scores are non-increasing and the first passage has `context_id == "ffffffffffffffff"` | unsorted output |
| rebuild is idempotent | call `build_index` twice on the same file | both calls return `4`, and the store holds 4 points | appending points on a second run |
| return type | any search result | each item is an instance of `rag_engine.api.Passage` | redefining a look-alike class |

Data tests (marker `needs_data`; they need the downloaded raw file, the built cleaned file and the
real model):

- Build the real index into `data/contrib_index/` with the real embedder. Draw 100 rows with
  `df.sample(100, random_state=20231231)`. For each row, search with its `question`,
  `tickers=[row ticker]` and `k=5`. At least 70 of the 100 searches return the row's own
  `context_id` among the 5 passages. Print the count when the test runs.
- Run the same 100 questions without `tickers` and print that count as well. The test asserts
  nothing about it; the pull request body reports both numbers.

Mark the data test with `@pytest.mark.needs_data`. It is deselected by default and runs only on
your machine; this task has no PostgreSQL test, so CI's integration job has nothing to run here.
Run it locally with `uv run pytest -m "integration or needs_data" rag_engine/contrib`. Use no
`pytest.skip`, `skipif`, `importorskip` or `xfail` anywhere.

## Rules

- Tests first, with pytest. Unit tests make no network calls.
- `uv run ruff check .` and `uv run ruff format .` are clean.
- Windows-safe paths: `pathlib.Path`, `encoding="utf-8"` on every text read and write.
- Secrets stay in your local `.env`, which is never committed. This task needs none.
- Errors are one friendly line and a non-zero exit code for any CLI; a library function raises a
  clear exception with a short message and no stack trace in the message text.
- Conventional commit messages, for example `feat(rag): add baseline vector search`.

## Definition of done

- `uv run pytest rag_engine/contrib` passes.
- The data test passes locally:
  `uv run pytest -m "integration or needs_data" rag_engine/contrib`.
- The repository gate passes: `uv run python scripts/gate.py`.
- `git status` shows changes only under the allowed paths; no skipped or xfail tests; no secrets.

## Pull request

1. Push the branch: `git push -u origin m3/baseline-search`. Without write access to the repository,
   fork it on GitHub, push to your fork and open the pull request from there.
2. Open a draft pull request to `main` by 2026-11-30, for example
   `gh pr create --draft --base main --title "feat(m3): baseline vector search with fastembed and Qdrant"`.
3. Fill in the pull request template: task id M3; what changed; what the agent generated and what
   the contributor checked by hand (for example: read `rag_search.py` line by line, ran the data
   test, confirmed no answer text is in the index); how to run it; the checklist. Report the hit
   count with the ticker filter and the hit count without it (out of 100).
   CI does not run `needs_data` tests: run them with `uv run pytest -m needs_data rag_engine/contrib`
   and say in the body that you ran them and that they passed.
4. Keep each pull request to one concern and under about 400 changed lines.
5. Mark it ready for review by 2026-12-07. CI must be green, and the code owner's approval is
   required before it merges.
