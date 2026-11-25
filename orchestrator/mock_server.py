"""Mock HTTP API with canned answers, for developing clients without a backend.

It serves the same models and status codes as the real server. The answer is picked
by keyword in the question; mode "local" always returns 503.

Run it:
    uv run uvicorn orchestrator.mock_server:app --host 127.0.0.1 --port 8000
"""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import TypeAdapter

from orchestrator.api_contract import (
    AskRequest,
    AskResponse,
    ErrorDetail,
    HealthComponents,
    HealthResponse,
)
from orchestrator.llm import LOCAL_NOT_CONFIGURED

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "mock_responses.json"
KEYWORDS = ("revenue", "late", "anomal")
NO_MATCH_KEY = "no_match"
REQUIRED_KEYS = frozenset((*KEYWORDS, NO_MATCH_KEY))
ALLOWED_ORIGINS = ["http://localhost:5173", "http://localhost:8501"]
MS_PER_SECOND = 1000
STATUS_LOCAL_UNAVAILABLE = 503


def _load_responses() -> dict[str, AskResponse]:
    """Read and validate the canned responses; fail clearly on a missing key."""
    raw = FIXTURE_PATH.read_text(encoding="utf-8")
    responses = TypeAdapter(dict[str, AskResponse]).validate_json(raw)
    if set(responses) != REQUIRED_KEYS:
        raise RuntimeError(
            f"{FIXTURE_PATH.name} must have exactly these keys: {sorted(REQUIRED_KEYS)}; "
            f"found: {sorted(responses)}"
        )
    return responses


RESPONSES = _load_responses()

app = FastAPI(title="SovereignFin-3M mock API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)


def _pick_key(question: str) -> str:
    """Return the first keyword found in the question, or the no-match key."""
    folded = question.casefold()
    return next((word for word in KEYWORDS if word in folded), NO_MATCH_KEY)


@app.post("/ask", response_model=AskResponse, responses={503: {"model": ErrorDetail}})
def ask(request: AskRequest) -> AskResponse:
    """Return the canned answer that matches the question."""
    start = time.perf_counter()
    if request.mode == "local":
        raise HTTPException(status_code=STATUS_LOCAL_UNAVAILABLE, detail=LOCAL_NOT_CONFIGURED)
    latency_ms = max(0, int((time.perf_counter() - start) * MS_PER_SECOND))
    return RESPONSES[_pick_key(request.question)].model_copy(
        update={"mode": request.mode, "latency_ms": latency_ms}
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report every component as up."""
    return HealthResponse(
        status="ok",
        components=HealthComponents(llm="up", qdrant="up", postgres="up"),
    )
