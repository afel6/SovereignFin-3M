"""Contract tests for rdl_engine/api.py (docs/contracts.md, RDL engine API)."""

import datetime
import inspect
import typing

import pytest
from pydantic import ValidationError

from rdl_engine import api

HIT = {
    "transaction_id": "t1",
    "ticker": "ABC",
    "vendor_id": 7,
    "vendor_name": "Vendor",
    "amount": 12.5,
    "invoice_date": "2023-01-05",
    "score": 0.9,
    "reasons": ["late"],
    "model_version": "v1",
}


def test_anomaly_hit_fields_exact():
    """Fails if a field is renamed, added, dropped or reordered."""
    assert list(api.AnomalyHit.model_fields) == list(HIT)


def test_anomaly_hit_date_parsed():
    """Fails if invoice_date stays a string."""
    hit = api.AnomalyHit(**HIT)
    assert hit.invoice_date == datetime.date(2023, 1, 5)
    assert type(hit.invoice_date) is datetime.date


def test_anomaly_hit_bad_date_refused():
    """Fails if invoice_date is not validated."""
    with pytest.raises(ValidationError):
        api.AnomalyHit(**{**HIT, "invoice_date": "not a date"})


def test_anomaly_hit_vendor_id_int():
    """Fails if vendor_id is typed as a string."""
    assert typing.get_type_hints(api.AnomalyHit)["vendor_id"] is int


def test_train_report_fields_and_default():
    """Fails if fields differ or notes has no empty-string default."""
    assert list(api.TrainReport.model_fields) == [
        "model_version",
        "auprc",
        "n_train",
        "n_test",
        "notes",
    ]
    r = api.TrainReport(model_version="v1", auprc=0.5, n_train=10, n_test=2)
    assert r.notes == ""


def test_top_anomalies_signature():
    """Fails if names, order, defaults or annotations differ from the spec."""
    sig = inspect.signature(api.top_anomalies)
    assert list(sig.parameters) == ["ticker", "k"]
    assert sig.parameters["ticker"].default is None
    assert sig.parameters["k"].default == 10
    hints = typing.get_type_hints(api.top_anomalies)
    assert hints["ticker"] == (str | None)
    assert hints["k"] is int
    assert hints["return"] == list[api.AnomalyHit]


def test_train_and_score_signature():
    """Fails if train_and_score takes other parameters or returns another type."""
    sig = inspect.signature(api.train_and_score)
    assert list(sig.parameters) == ["db_url"]
    hints = typing.get_type_hints(api.train_and_score)
    assert hints["db_url"] is str
    assert hints["return"] is api.TrainReport


def test_top_anomalies_not_implemented_names_slice():
    """Fails if it returns, or the message omits the slice S3."""
    with pytest.raises(NotImplementedError, match="S3"):
        api.top_anomalies()


def test_train_and_score_message_leaks_no_password():
    """Fails if the error message echoes the database URL."""
    secret = "pass" + "word123"
    url = "postgresql://user:" + secret + "@host/db"
    with pytest.raises(NotImplementedError) as exc:
        api.train_and_score(url)
    assert "S3" in str(exc.value)
    assert secret not in str(exc.value)
