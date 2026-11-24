# SovereignFin-3M

Offline financial intelligence engine (university team project). A LangGraph supervisor routes analyst questions to:

- **Agentic RAG** over SEC 10-K question/answer snippets (Qdrant)
- **Relational Deep Learning** (PyTorch Geometric) that flags anomalous vendor transactions in a synthetic PostgreSQL database
- **AtlasKV-style knowledge-graph injection** (proof of concept)

Everything runs locally at inference time; no data leaves the machine.

## Repository layout

| Folder | Contents |
|---|---|
| `data_pipeline/` | Dataset download, cleaning, synthetic data generation |
| `database/` | PostgreSQL schema (`init.sql`) and queries |
| `rag_engine/` | Retrieval and answer generation over 10-K snippets |
| `rdl_engine/` | Graph neural network for transaction anomaly detection |
| `atlas_kv/` | Knowledge-graph encoding and injection |
| `orchestrator/` | LangGraph supervisor and tools |
| `frontend/` | User interface |
| `evaluation/` | Metrics and benchmark harness |

## Contributing

`main` is protected: all changes arrive through pull requests and need an approving review from the code owner. Setup instructions, task briefs and the contribution guide are coming in the next commits.
