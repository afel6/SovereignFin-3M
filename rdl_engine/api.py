"""Public interface of the anomaly engine: scored transactions and offline training."""

import datetime

from pydantic import BaseModel

TOP_ANOMALIES_NOT_IMPLEMENTED = "rdl_engine.top_anomalies is not implemented yet (slice S3)."
TRAIN_NOT_IMPLEMENTED = "rdl_engine.train_and_score is not implemented yet (slice S3)."


class AnomalyHit(BaseModel):
    """A transaction flagged as anomalous."""

    transaction_id: str
    ticker: str
    vendor_id: int
    vendor_name: str
    amount: float
    invoice_date: datetime.date
    score: float
    reasons: list[str]
    model_version: str


class TrainReport(BaseModel):
    """Summary of one training and scoring run."""

    model_version: str
    auprc: float
    n_train: int
    n_test: int
    notes: str = ""


def top_anomalies(ticker: str | None = None, k: int = 10) -> list[AnomalyHit]:
    """Return the k highest-scoring anomalies from the anomaly_scores table."""
    raise NotImplementedError(TOP_ANOMALIES_NOT_IMPLEMENTED)


def train_and_score(db_url: str) -> TrainReport:
    """Train the model and score all transactions (offline batch)."""
    raise NotImplementedError(TRAIN_NOT_IMPLEMENTED)
