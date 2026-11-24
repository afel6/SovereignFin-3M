# Contracts

These are the fixed interfaces of SovereignFin-3M: data files, Python APIs, the HTTP API, the
database tables and the environment variables. All code in this repository builds to them exactly.
To propose a change, open an issue; the maintainer updates this file.

Each contract is a fenced block tagged `contract <id>`, where `<id>` is also the anchor of the
heading above it (for example `docs/contracts.md#raw-input`). Task briefs in `docs/tasks/` copy
whole blocks from this file, and `docs/tests/test_contract_excerpts.py` checks that every copy
matches this file character for character.

## 1. Cleaned 10-K dataset

### Raw input

```contract raw-input
File:     data/raw/financial_qa_10k.parquet
Source:   Hugging Face dataset virattt/financial-qa-10K, train file
sha256:   4b71e8cb8d0ce7a01eb117d3ef55afa994ab47571728ac8b2fa052e061d0a066
Rows:     7000
Columns:  exactly {question, answer, context, ticker, filing}
raw_index = the 0-based row position in this file.
Download: uv run python -m data_pipeline.download   (fetches the file and checks the sha256)
```

### Cleaned output schema

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

### Text normalization

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

### Drop rules

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

### Cleaning interface

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

### Validator checks

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

## 2. Engine APIs

Each engine folder exposes one `api.py` with pydantic v2 models. Its functions are the only public
surface.

### RAG engine API

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

### RDL engine API

```contract rdl-engine-api
Module: rdl_engine/api.py

class AnomalyHit(BaseModel):
    transaction_id: str
    ticker: str
    vendor_id: int
    vendor_name: str
    amount: float
    invoice_date: date
    score: float
    reasons: list[str]
    model_version: str

def top_anomalies(ticker: str | None = None, k: int = 10) -> list[AnomalyHit]
    # reads the anomaly_scores table
def train_and_score(db_url: str) -> TrainReport
    # offline batch
```

### Knowledge API

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

## 3. HTTP API

Models live in `orchestrator/api_contract.py`. The real server is `orchestrator/app.py`; the mock
server is `orchestrator/mock_server.py`.

### Ask request

```contract ask-request
POST /ask
Body AskRequest:
  question: str          1..2000 characters after stripping leading and trailing whitespace
  mode: "local" | "cloud" = "cloud"
  thread_id: str | None  at most 64 characters from [A-Za-z0-9_-]
An invalid body returns 422 (FastAPI default error body).
```

### Ask response

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

### Chart spec

```contract chart-spec
ChartSpec (orchestrator/api_contract.py; import it from there, never redefine it):
  chart_type: "bar" | "line" | "pie" | "table"
  title: str
  x_field: str | null
  y_fields: list[str]
  data: list[dict[str, str | int | float | null]]   at most 500 rows
```

### Modes and errors

```contract modes-and-errors
mode "cloud": the backend is chosen by SF_LLM_PROVIDER, one of nvidia (default),
  openai_compatible, stub. Cloud mode is configured if and only if:
  SF_LLM_PROVIDER=nvidia and NVIDIA_API_KEY is set, or
  SF_LLM_PROVIDER=openai_compatible and SF_LLM_BASE_URL is set, or
  SF_LLM_PROVIDER=stub.
mode "local": always the Ollama preset (http://127.0.0.1:11434/v1); configured if and only if
  SF_LLM_LOCAL_MODEL is set.

Not configured -> 503 with exactly one of these bodies:
  local:                     {"detail": "Local mode is not configured on this machine."}
  cloud, nvidia:             {"detail": "Cloud mode is not configured: set NVIDIA_API_KEY."}
  cloud, openai_compatible:  {"detail": "Cloud mode is not configured: set SF_LLM_BASE_URL and SF_LLM_API_KEY."}
Any other LLM error -> 503 {"detail": "LLM unavailable"}
A dependency that is down -> 503 {"detail": "<component> unavailable"}
Invalid body -> 422 (FastAPI default)
Responses never contain key values.
```

### Timeouts

```contract timeouts
LLM: 20 s per attempt; at most 2 retries (3 attempts) on 429, 5xx or timeout, with backoff
     1 s, then 4 s.
Qdrant: 5 s.  Postgres: statement timeout 5 s.
/ask is capped at 90 s in total; past that the server returns 503 {"detail": "LLM unavailable"}.
Worst case is about 3 LLM calls x 3 attempts x 20 s.
Target: /ask p95 at most 15 s on the reference laptop, excluding 429 backoff.
```

### Health endpoint

```contract health-endpoint
GET /health -> 200
{"status": "ok" | "degraded",
 "components": {"llm": "up" | "down", "qdrant": "up" | "down", "postgres": "up" | "down"}}
```

### Server and mock

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

## 3b. LLM client

### LLM client

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

## 4. Knowledge-graph triples

### Triple vocabulary

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

## 5. SQL

### Database roles

```contract database-roles
sf_owner   (DATABASE_URL_OWNER)  offline jobs only: schema setup, CSV loader, 10-K snippet loader,
                                 anomaly_scores writer
sf_reader  (DATABASE_URL)        all query-time code; SELECT grants only; read-only transactions by
                                 default; statement_timeout 5 s set on the role; default privileges
                                 give it SELECT on tables sf_owner creates later
Role passwords come from .env, never from SQL files.
```

### Anomaly scores table

```contract anomaly-scores-table
-- created by database/init.sql; written only by sf_owner
CREATE TABLE anomaly_scores (
    txn_id         bigint      REFERENCES transactions,
    model_version  text,
    score          real        NOT NULL CHECK (score BETWEEN 0 AND 1),
    reasons        text[]      NOT NULL,
    data_snapshot  bigint      NOT NULL,  -- generator seed-run epoch or a monotonically increasing build number
    scored_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (txn_id, model_version)
);
-- upsert: never replaces a row scored from newer data
INSERT INTO anomaly_scores (...) VALUES (...)
ON CONFLICT (txn_id, model_version) DO UPDATE SET ...
WHERE anomaly_scores.data_snapshot <= EXCLUDED.data_snapshot;
```

### Named queries

```contract named-queries
database/queries/*.sql: named, parameterized queries (psycopg %(name)s placeholders):
  company_overview, vendor_payments, late_payments, top_transactions, anomaly_summary
They run only by name, under sf_reader, with statement_timeout 5 s.
Application SQL safety = the read-only role + named queries only. The SQL guard below is an
extra layer; the application does not depend on it.
```

### SQL guard

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

### Chart conversion

```contract chart-conversion
Module: database/charts/sql_to_chart.py

def dataframe_to_chart(df: pd.DataFrame, chart_type: str | None = None, x: str | None = None,
                       y: list[str] | None = None, title: str = "") -> ChartSpec
ChartSpec is imported from orchestrator.api_contract (see "Chart spec").
```

## 6. Environment variables

Values come from a local `.env` file, which is never committed. `.env.example` lists the names.
Keys are read only from the environment, never logged and never sent anywhere except the
configured LLM provider host.

### Database variables

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

### LLM variables

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

### Service variables

```contract service-variables
QDRANT_URL=http://127.0.0.1:6333
SF_API_BASE=http://127.0.0.1:8000
# HF_HUB_OFFLINE=1 is set by the query-time server process only; the download and model
# pre-fetch commands run online, on purpose.
```
