# AGENTS.md

Repository rules for coding agents. Read this file before any change.

## All agents

### Project

SovereignFin-3M is a local-first financial question-answering engine (university project).
A FastAPI endpoint (`/ask`) runs a LangGraph supervisor that routes a question to retrieval over SEC 10-K question/answer snippets (Qdrant), named read-only SQL over a synthetic ERP database (PostgreSQL), anomaly scores from a graph neural network, and knowledge-graph fact retrieval, then returns a cited answer with an optional chart.
The LLM is a free hosted OpenAI-compatible API (NVIDIA build API by default); everything else runs locally.

### Setup and commands

- Use Python 3.13 and uv. Run every command through `uv run`.
- Install with `uv sync --group <group>`. The `dev` group is installed by default; `uv sync --all-groups` installs everything.
- Run `uv run python scripts/gate.py` before every commit. It checks the lock file, runs `ruff check`, `ruff format --check` and pytest with coverage.
- Lint with `uv run ruff check .` and format with `uv run ruff format .`.

### Tests

- Write the failing pytest test first, then the code.
- Mark tests that need PostgreSQL with `integration` and tests that need downloaded data or models with `needs_data`. Both are deselected by default; run them with `uv run pytest -m "integration or needs_data" <path>`.
- Express every condition through a marker or a fixture; the repository has no `skip`, `skipif`, `importorskip` or `xfail`.
- Keep tests offline: set `SF_LLM_PROVIDER=stub` for the LLM, and use `httpx.MockTransport` or FastAPI's `TestClient` for HTTP.

### Contracts

- `docs/contracts.md` is the source of truth for data schemas, Python signatures, the HTTP API, SQL tables and environment variables. Build to it exactly.

### Secrets

- Keep keys only in a local `.env` (git-ignored). `.env.example` lists names only.
- Keep keys and database URLs out of prints, logs and error messages.
- Build fake key strings in tests by concatenation, because CI's secret scan rejects key-shaped strings.

### Code

- Make code Windows-safe: use `pathlib`, pass `encoding="utf-8"` on every text read and write, write `"\n"` line endings, and keep absolute paths and backslashes out of code. Write scripts in Python, not shell.
- Write SQL with parameters only (psycopg `%s` or `%(name)s`). Pass values as parameters and keep them out of SQL text.
- Use the read-only `DATABASE_URL` in query-time code.
- Make CLIs print one friendly line and exit non-zero on an expected failure.

### Commits

- Use conventional commits: `feat(scope): ...`, `fix`, `test`, `docs`, `refactor`, `chore`.

## Contributor branches (mN/*)

Applies when `git branch --show-current` starts with `m2/` through `m8/`.

- Read your task brief, `docs/tasks/MN-*.md` (N is the number in your branch name), and follow it.
- Change only files under the paths in the brief's "Allowed paths" list. CI rejects a pull request that touches any other path.
- Treat the contracts in `docs/contracts.md` as fixed. When one blocks you, stop and ask the contributor to open an issue that proposes the change.
- Leave these files to the maintainer: every `api.py`, `orchestrator/api_contract.py`, `pyproject.toml`, `uv.lock`, `.github/`, `scripts/`, `docker-compose.yml`, `.env.example`, `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`. When you need a dependency, ask the contributor to open an issue.
- Put tests inside your allowed path, for example `data_pipeline/clean/tests/`.
- Keep each pull request to one concern and under about 400 changed lines.
- State in the pull request body what the agent generated, what the contributor verified by hand, and how to run it. The template asks for this.
- Before the pull request, run `uv run pytest <your path>`, then the gate.
- Keep your branch current: `git switch main && git pull`, then merge `main` into your branch.
