# Contributing to SovereignFin-3M

## Setup

1. Install uv from https://docs.astral.sh/uv/.
2. Install Python: `uv python install 3.13`.
3. Clone the repository and enter it:

   ```bash
   git clone https://github.com/afel6/SovereignFin-3M.git
   cd SovereignFin-3M
   ```

4. Install the dependency group your work needs: `uv sync --group <group>` (`uv sync --all-groups` installs everything).
5. Copy `.env.example` to `.env` and fill in the values. `.env` is git-ignored; keep it that way.
6. Start PostgreSQL and Qdrant on 127.0.0.1: `docker compose up -d` (Docker Desktop must be running).

## Branches

- Task contributors: name the branch `mN/<topic>`, for example `m2/data-cleaning`. Your brief in [docs/tasks/](docs/tasks/) gives the number and the allowed paths.
- Other changes: name the branch `<type>/<topic>`, for example `fix/health-timeout`.

## Gate

Run this before every commit and before you open a pull request:

```bash
uv run python scripts/gate.py
```

It checks the lock file, runs `ruff check`, `ruff format --check` and pytest with coverage.

## Tests

Write the failing test first. Tests that need PostgreSQL carry the marker `integration`; tests that need downloaded data or models carry `needs_data`.
Both are deselected by default: `uv run pytest -m "integration or needs_data" <path>`.
CI's integration job runs the `integration` tests against a PostgreSQL service. CI never runs
`needs_data` tests: run them locally with `uv run pytest -m needs_data <path>` and say in the
pull request body that you ran them.

## Pull requests

- `main` is protected. Every change arrives by pull request.
- CI must pass: lint, tests, allowed-path check and secret scan.
- The code owner listed in [.github/CODEOWNERS](.github/CODEOWNERS) must approve. A new push dismisses earlier approvals, and every conversation must be resolved.
- Open a draft pull request early.
- Keep each pull request to one concern and under about 400 changed lines.
- Send contract changes and new dependencies through an issue first.
- Coding agents are welcome. The pull request states what the agent generated and what you verified by hand.

## Reference

- [AGENTS.md](AGENTS.md): repository rules for you and your coding agent.
- [docs/contracts.md](docs/contracts.md): the fixed contracts.
- [docs/tasks/](docs/tasks/): task briefs.
