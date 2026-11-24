# Copilot instructions

Read and follow `AGENTS.md` at the repository root before any change. The five rules that matter most:

1. Run `uv run python scripts/gate.py` before every commit; it must pass.
2. Write the failing test first, then the code.
3. Treat `docs/contracts.md` as fixed: data schemas, signatures, the HTTP API, SQL tables and environment variables. Ask for an issue when a contract blocks you.
4. On a contributor branch (`m2/` to `m8/`), change only the files under the "Allowed paths" of your brief in `docs/tasks/`.
5. Keep secrets, keys and `.env` files out of commits, logs and tests.
