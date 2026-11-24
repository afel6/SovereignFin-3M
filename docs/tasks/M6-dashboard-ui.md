# Task M6: Build the dashboard UI

Paste this whole file into your coding agent (Claude Code, Codex, Cursor or Copilot) as the first
message. It holds everything the task needs.

## Role and goal

You are a Python front-end engineer in the SovereignFin-3M repository, a local-first financial
question-answering engine built as a university project. You work for a contributor (your user),
who reviews your work and opens the pull request.

Goal: build a dashboard that talks to the HTTP API in the Contract section: a chat for questions
and cited answers, a chart panel, a Local/Cloud toggle and a health indicator. It must work against
the mock server alone.

Work in this order: Setup, Tests to write first, code, Definition of done, Pull request.

## Setup

Requires Python 3.13 and uv (https://docs.astral.sh/uv/). `uv python install 3.13` installs Python
if it is missing.

```bash
git clone https://github.com/afel6/SovereignFin-3M.git
cd SovereignFin-3M
git pull
git switch -c m6/dashboard-ui
uv sync --group ui --group dev
```

The `ui` group includes the `api` group, so the mock server runs from the same environment.

If you cloned earlier, run `git switch main` and `git pull` before you create the branch. Then read
`AGENTS.md` at the repository root, when it exists, and follow it. Run `git pull` on `main` again
at the start of each work session and merge `main` into your branch, so new shared files arrive.

## Allowed paths

You create and change files only under these paths:

- `frontend/`

Every other path is off-limits: other folders, every `api.py`, `orchestrator/api_contract.py`,
`pyproject.toml`, `uv.lock`, `.github/`, `AGENTS.md`, `CLAUDE.md`. A CI check rejects a pull request
that changes any file outside the allowed paths. When you need a new dependency or a contract
change, stop and ask the contributor to open an issue.

## Files to commit

The details of this brief target Streamlit (recommended). React with Vite and TypeScript is an
alternative, with its own `frontend/package.json` and Vitest tests, all under `frontend/`; use it
only when the contributor asks for it, and map the requirements below onto it.

- `frontend/app.py` — the Streamlit app: chat, chart panel, toggle, health indicator
- `frontend/api_client.py` — functions that call the API with an injected `httpx.Client`
- `frontend/tests/` — the tests (`test_*.py`)

## Runtime outputs (never commit)

None.

## Contract

These blocks are copied verbatim from `docs/contracts.md` in the repository. Build to them exactly
and leave them unchanged. When one looks wrong, stop and ask the contributor to open an issue.

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

```contract chart-spec
ChartSpec (orchestrator/api_contract.py; import it from there, never redefine it):
  chart_type: "bar" | "line" | "pie" | "table"
  title: str
  x_field: str | null
  y_fields: list[str]
  data: list[dict[str, str | int | float | null]]   at most 500 rows
```

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

```contract timeouts
LLM: 20 s per attempt; at most 2 retries (3 attempts) on 429, 5xx or timeout, with backoff
     1 s, then 4 s.
Qdrant: 5 s.  Postgres: statement timeout 5 s.
/ask is capped at 90 s in total; past that the server returns 503 {"detail": "LLM unavailable"}.
Worst case is about 3 LLM calls x 3 attempts x 20 s.
Target: /ask p95 at most 15 s on the reference laptop, excluding 429 backoff.
```

```contract health-endpoint
GET /health -> 200
{"status": "ok" | "degraded",
 "components": {"llm": "up" | "down", "qdrant": "up" | "down", "postgres": "up" | "down"}}
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

```contract service-variables
QDRANT_URL=http://127.0.0.1:6333
SF_API_BASE=http://127.0.0.1:8000
# HF_HUB_OFFLINE=1 is set by the query-time server process only; the download and model
# pre-fetch commands run online, on purpose.
```

## Dashboard requirements

These define the part of the task that the contract leaves to you.

1. Chat
   - A question input with a visible label, and a way to send it by keyboard.
   - The answer text, and the outcome in words: `answered` shows "Answered", `not_found` shows
     "Not found", `refused` shows "Refused", `clarify` shows "Needs clarification".
   - The citations as a list; each entry shows its source, its ticker (or a dash when null) and its
     text. A 10-K citation identifies a passage, not a page; show no page number.
   - Any `warnings` as plain lines under the answer.
2. Chart panel
   - Render `ChartSpec` of type `bar`, `line` and `pie` with Altair (it ships with Streamlit) and
     `table` with `st.dataframe`.
   - Show the `title` above every chart, and a table view of the same `data` (an expander or tab)
     for every chart type.
   - Show no chart panel when `chart` is null.
3. Mode toggle and health
   - A Local/Cloud toggle with a text label, default Cloud, that sends `mode` as `"local"` or
     `"cloud"` in the request body.
   - A 503 response shows its `detail` text unchanged, for example the Local "not configured"
     message.
   - A health indicator from `GET /health` shows the `status` and each component as text, for
     example "llm: up" and "status: degraded". Colour may add to the text, never replace it.
4. API access
   - The API base is read from the environment variable `SF_API_BASE` and defaults to
     `http://127.0.0.1:8000`.
   - The client timeout is 100 seconds, above the server's 90 second cap, kept in a named constant.
   - Connection errors, timeouts, 422 and 503 each show one friendly line in the UI, and the UI
     never shows a stack trace. A 503 shows the server's `detail`; the other three use fixed
     wording of your choice.
   - Every function in `api_client.py` takes an `httpx.Client` as a parameter, so tests inject one.
5. Accessibility basics
   - Every input has a visible label; the toggle has a text label.
   - Status never relies on colour alone.
   - Every chart has a title and a table alternative.
   - Everything works by keyboard.

## Data

- Run the mock server with
  `uv run uvicorn orchestrator.mock_server:app --port 8000`, then the app with
  `uv run streamlit run frontend/app.py` (port 8501, which the server's CORS list allows). The mock
  server, `orchestrator/api_contract.py` and `orchestrator/llm.py` land on `main` around
  2026-11-25; until then, tests build `AskResponse`-shaped dictionaries inside the test file and
  answer with `httpx.MockTransport`.
- Tests never start a server. Code takes an injected `httpx.Client`. Tests pass
  `fastapi.testclient.TestClient(orchestrator.mock_server.app)`, which is an `httpx.Client`, or an
  `httpx.Client` with `httpx.MockTransport`.
- The UI makes no LLM calls itself. Tests set `SF_LLM_PROVIDER=stub` with `monkeypatch.setenv`
  wherever a code path could reach the LLM client. Never write a realistic-looking fake key
  anywhere (CI runs a secret scan); build fake key strings in tests by concatenation.

## Tests to write first

Write these in `frontend/tests/` before the code, run them, see them fail, then write the code
until they pass. Every expected value below follows from the Contract and Dashboard requirements
sections.

`api_client` against `TestClient(orchestrator.mock_server.app)`:

| call | expected | catches |
|---|---|---|
| ask `"What was revenue in 2023?"`, mode `"cloud"` | `outcome` and `citations` present, `citations` non-empty, route contains `"rag"` | ignoring the keyword rule of the mock |
| ask `"Which invoices are late?"`, mode `"cloud"` | `chart.chart_type == "bar"`, route contains `"sql"` | dropping the `chart` field |
| ask any question with mode `"local"` | a 503 error carrying exactly `"Local mode is not configured on this machine."` | rewording the server's `detail` |
| health | `status` in `{"ok", "degraded"}` and the three components `llm`, `qdrant`, `postgres` | reading the wrong keys |

Request body, with an `httpx.MockTransport` handler that records the request:

- Asking `question="q"`, `mode="local"`, `thread_id="t-1"` sends the JSON body
  `{"question": "q", "mode": "local", "thread_id": "t-1"}`. This catches a hardcoded mode.
- Without a thread id the body carries `thread_id` as null or omits it; both validate against the
  contract.

Error mapping with `httpx.MockTransport` (each case returns one friendly line, no exception
escapes):

| transport behaviour | expected | catches |
|---|---|---|
| the handler raises `httpx.ConnectError` | one friendly line, no traceback text | crashing when the API is down |
| the handler raises `httpx.ReadTimeout` | one friendly line | treating a timeout as a crash |
| status 422 with the FastAPI error body | one friendly line | printing the raw validation body |
| status 503 with `{"detail": "LLM unavailable"}` | the text `"LLM unavailable"` unchanged | replacing the server's `detail` |

Also assert the timeout constant equals 100 and the default API base is `http://127.0.0.1:8000`
when `SF_API_BASE` is unset (use `monkeypatch.delenv`), and that setting `SF_API_BASE` changes it.

Streamlit `AppTest` (`streamlit.testing.v1.AppTest.from_file("frontend/app.py")`) with the client
replaced by one that uses `httpx.MockTransport`; run the app and read the element tree:

- A response with two citations renders both, each with its source, ticker and text.
- Each of `bar`, `line`, `pie` and `table` charts renders without an exception
  (`at.exception` is empty), with the title shown and a table view present.
- A response with `chart` null shows no chart panel.
- Each outcome value shows its words: "Answered", "Not found", "Refused", "Needs clarification".
- Switching the toggle to Local makes the recorded request carry `"mode": "local"`; the default
  request carries `"cloud"`.
- A health response with `"status": "degraded"` shows the text "degraded" and every component name
  with its state.
- A 503 shows the `detail` text in the UI and no exception.

Mark any test that needs PostgreSQL with `@pytest.mark.integration` and any that needs downloaded
data with `@pytest.mark.needs_data`; this task needs neither. Both markers are deselected by
default. CI's integration job runs the `integration` tests with a PostgreSQL service only: no
downloaded data and no other services. Run both locally with
`uv run pytest -m "integration or needs_data" frontend`. Use no `pytest.skip`, `skipif`,
`importorskip` or `xfail` anywhere.

## Rules

- Tests first, with pytest. Unit tests make no network calls and start no server.
- `uv run ruff check .` and `uv run ruff format .` are clean.
- Windows-safe paths: `pathlib.Path`, `encoding="utf-8"` on every text read and write.
- Secrets stay in your local `.env`, which is never committed. Never print or log a key.
- UI errors are one friendly line; no stack trace shows in the UI.
- Conventional commit messages, for example `feat(frontend): add the chat panel`.

## Definition of done

- `uv run pytest frontend` passes.
- The app runs against the mock server and shows an answer with citations, a bar chart, the Local
  503 message and the health text: `uv run uvicorn orchestrator.mock_server:app --port 8000` and
  `uv run streamlit run frontend/app.py`.
- The repository gate passes: `uv run python scripts/gate.py`.
- `git status` shows changes only under the allowed paths; no skipped or xfail tests; no secrets.

## Pull request

1. Push the branch: `git push -u origin m6/dashboard-ui`. Without write access to the repository,
   fork it on GitHub, push to your fork and open the pull request from there.
2. Open a draft pull request to `main` by 2026-11-30, for example
   `gh pr create --draft --base main --title "feat(m6): add the dashboard UI"`.
3. Fill in the pull request template: task id M6; what changed; what the agent generated and what
   the contributor checked by hand (for example: used the app by keyboard only, checked each chart
   type against the mock server); how to run it; the checklist.
4. Keep each pull request to one concern and under about 400 changed lines; for example, the API
   client and chat first and the chart panel second.
5. Mark it ready for review by 2026-12-07. CI must be green, and the code owner's approval is
   required before it merges.
