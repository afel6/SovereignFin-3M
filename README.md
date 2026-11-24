# SovereignFin-3M

Local-first financial intelligence engine (university team project). A LangGraph supervisor routes analyst questions to:

- **Agentic RAG** over SEC 10-K question/answer snippets (Qdrant)
- **Relational Deep Learning** (PyTorch Geometric) that flags anomalous vendor transactions in a synthetic PostgreSQL database
- **Knowledge-graph retrieval**: facts from a knowledge graph added to the prompt, plus an experimental AtlasKV study

The LLM is a free hosted OpenAI-compatible API by default (NVIDIA build API); a local mode (Ollama) is planned. Embeddings, Qdrant, PostgreSQL and the graph model run locally. Data stays on the machine except the question and the retrieved snippets, which are sent to the configured LLM provider.

## Repository layout

| Folder | Contents |
|---|---|
| `data_pipeline/` | Dataset download, cleaning, synthetic data generation |
| `database/` | PostgreSQL schema (`init.sql`) and queries |
| `rag_engine/` | Retrieval and answer generation over 10-K snippets |
| `rdl_engine/` | Graph neural network for transaction anomaly detection |
| `atlas_kv/` | Knowledge-graph triples, fact retrieval and the experimental AtlasKV study |
| `orchestrator/` | LangGraph supervisor and tools |
| `frontend/` | User interface |
| `evaluation/` | Metrics and benchmark harness |
| `docs/` | Contracts, task briefs and presentation |
| `scripts/` | The local gate (`gate.py`) |

## Quick start

```bash
uv sync --group api
cp .env.example .env   # fill in passwords; NVIDIA_API_KEY only for live LLM calls
docker compose up -d   # Docker Desktop must be running
uv run uvicorn orchestrator.mock_server:app --port 8000   # canned answers matching the HTTP contract while the engines are built
uv run python scripts/gate.py
```

## Contributing

`main` is protected: all changes arrive through pull requests and need an approving review from the code owner.

- [CONTRIBUTING.md](CONTRIBUTING.md): setup, branches, gate and pull request flow
- [AGENTS.md](AGENTS.md): repository rules for coding agents
- [docs/contracts.md](docs/contracts.md): the fixed contracts
- [docs/tasks/](docs/tasks/): task briefs
