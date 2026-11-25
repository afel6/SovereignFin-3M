from datetime import date
from decimal import Decimal
from uuid import UUID

import numpy as np
import pandas as pd
import pytest

from database.charts import dataframe_to_chart
from orchestrator.api_contract import ChartSpec


def test_chart_inference():
    frame = pd.DataFrame(
        {"day": pd.to_datetime(["2024-01-01", "2024-01-02"]), "amount": [1.5, 2.5]}
    )
    chart = dataframe_to_chart(frame)
    assert isinstance(chart, ChartSpec)
    assert (chart.chart_type, chart.x_field, chart.y_fields) == ("line", "day", ["amount"])
    assert chart.data[0] == {"day": "2024-01-01T00:00:00", "amount": 1.5}
    frame["day"] = pd.Series([date(2024, 1, 1), date(2024, 1, 2)], dtype=object)
    assert dataframe_to_chart(frame).data[0]["day"] == "2024-01-01"
    assert dataframe_to_chart(frame).chart_type == "line"
    bar = pd.DataFrame({"vendor": ["a", "b"], "total": [10, 20], "n": [1, 2]})
    chart = dataframe_to_chart(bar)
    assert (chart.chart_type, chart.x_field, chart.y_fields) == ("bar", "vendor", ["total", "n"])
    assert (
        dataframe_to_chart(pd.DataFrame({"a": ["x"], "b": ["y"], "n": [1]})).chart_type == "table"
    )
    chart = dataframe_to_chart(pd.DataFrame({"a": [1], "b": [2]}))
    assert (chart.chart_type, chart.x_field, chart.y_fields) == ("table", None, ["a", "b"])
    assert dataframe_to_chart(frame[["day"]]).chart_type == "table"
    assert dataframe_to_chart(bar, chart_type="table").chart_type == "table"
    assert dataframe_to_chart(bar, chart_type="pie", y=["total"]).y_fields == ["total"]
    assert dataframe_to_chart(bar[["vendor", "total"]], chart_type="pie").chart_type == "pie"
    for kwargs in (
        {"chart_type": "pie"},
        {"chart_type": "scatter"},
        {"x": "nope"},
        {"y": ["nope"]},
    ):
        with pytest.raises(ValueError):
            dataframe_to_chart(bar, **kwargs)


def test_values_and_limit():
    chart = dataframe_to_chart(pd.DataFrame({"n": range(501)}), title="Payments")
    assert (len(chart.data), chart.data[0]["n"], chart.data[-1]["n"], chart.title) == (
        500,
        0,
        499,
        "Payments",
    )
    frame = pd.DataFrame(
        {
            "n": [np.int64(1)],
            "f": [np.float64(2.5)],
            "nan": [float("nan")],
            "date": [pd.NaT],
            "decimal": [Decimal("1.50")],
            "id": [UUID(int=1)],
        }
    )
    chart = dataframe_to_chart(frame)
    row = chart.data[0]
    assert type(row["n"]) is int and type(row["f"]) is float
    assert row["nan"] is None and row["date"] is None and row["decimal"] == 1.5
    assert row["id"] == "00000000-0000-0000-0000-000000000001"
    assert chart.title == "" and chart.model_dump_json()


def test_line_requires_numeric_selected_y():
    frame = pd.DataFrame({"day": pd.to_datetime(["2024-01-01"]), "label": ["a"]})
    assert dataframe_to_chart(frame, y=["label"]).chart_type == "table"


def test_unknown_values_are_stringified():
    frame = pd.DataFrame({"value": [np.array([1, 2])]})
    assert dataframe_to_chart(frame).data[0]["value"] == "[1 2]"


def test_nonfinite_values_are_json_safe():
    frame = pd.DataFrame({"value": [float("inf"), float("-inf")]})
    chart = dataframe_to_chart(frame)
    assert chart.data == [{"value": None}, {"value": None}]
    chart.model_dump_json()
