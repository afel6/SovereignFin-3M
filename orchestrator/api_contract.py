"""HTTP request and response models for the question-answering API."""

from typing import Annotated, Literal

from pydantic import AllowInfNan, BaseModel, ConfigDict, Field, Strict, StringConstraints

MAX_QUESTION_CHARS = 2000
MAX_THREAD_ID_CHARS = 64
MAX_CITATION_CHARS = 500
MAX_CHART_ROWS = 500
THREAD_ID_PATTERN = rf"^[A-Za-z0-9_-]{{0,{MAX_THREAD_ID_CHARS}}}$"

Mode = Literal["local", "cloud"]
Source = Literal["rag", "sql", "anomaly", "kg"]
Outcome = Literal["answered", "not_found", "refused", "clarify"]
ComponentStatus = Literal["up", "down"]
ChartValue = (
    Annotated[str, Strict()]
    | Annotated[int, Strict()]
    | Annotated[float, Strict(), AllowInfNan(False)]
    | None
)


class _ResponseModel(BaseModel):
    """Base for response models: immutable."""

    model_config = ConfigDict(frozen=True)


class AskRequest(BaseModel):
    """Body of POST /ask; unknown keys are ignored."""

    model_config = ConfigDict(frozen=True)

    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS),
    ]
    mode: Mode = "cloud"
    thread_id: Annotated[str, StringConstraints(pattern=THREAD_ID_PATTERN)] | None = None


class Citation(_ResponseModel):
    """One piece of evidence behind an answer."""

    id: str
    source: Source
    ticker: str | None
    text: Annotated[str, Field(max_length=MAX_CITATION_CHARS)]


class ChartSpec(_ResponseModel):
    """Chart data for the UI; import it from here, never redefine it."""

    chart_type: Literal["bar", "line", "pie", "table"]
    title: str
    x_field: str | None
    y_fields: list[str]
    data: Annotated[list[dict[str, ChartValue]], Field(max_length=MAX_CHART_ROWS)]


class AskResponse(_ResponseModel):
    """Reply of POST /ask."""

    answer: str
    outcome: Outcome
    citations: list[Citation]
    chart: ChartSpec | None
    route: list[Source]
    latency_ms: int
    mode: Mode
    warnings: list[str]


class HealthComponents(_ResponseModel):
    """Status of each backing service."""

    llm: ComponentStatus
    qdrant: ComponentStatus
    postgres: ComponentStatus


class HealthResponse(_ResponseModel):
    """Reply of GET /health."""

    status: Literal["ok", "degraded"]
    components: HealthComponents


class ErrorDetail(_ResponseModel):
    """Body of an error reply."""

    detail: str
