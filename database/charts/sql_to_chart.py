"""Convert query results into the shared chart response model."""

from datetime import date, datetime
from decimal import Decimal
from math import isfinite
from numbers import Integral, Real

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from orchestrator.api_contract import MAX_CHART_ROWS, ChartSpec


def _value(value):
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, (Real, Decimal)):
        number = float(value)
        return number if isfinite(number) else None
    return str(value)


def dataframe_to_chart(
    df: pd.DataFrame,
    chart_type: str | None = None,
    x: str | None = None,
    y: list[str] | None = None,
    title: str = "",
) -> ChartSpec:
    if not df.columns.is_unique:
        raise ValueError("Duplicate column names are not supported")
    if chart_type is not None and chart_type not in {"bar", "line", "pie", "table"}:
        raise ValueError("Unknown chart type")
    if x is not None and x not in df.columns:
        raise ValueError("Unknown x column")
    if y is not None and any(column not in df.columns for column in y):
        raise ValueError("Unknown y column")
    numeric = [column for column in df.columns if is_numeric_dtype(df[column])]
    chosen_y = numeric if y is None else y
    nonnumeric = [column for column in df.columns if column not in numeric]
    candidate = x if x is not None else (nonnumeric[0] if nonnumeric else None)
    if chart_type is None:
        dates = candidate is not None and (
            is_datetime64_any_dtype(df[candidate])
            or (
                not df[candidate].dropna().empty
                and df[candidate]
                .dropna()
                .map(lambda value: isinstance(value, (date, datetime)))
                .all()
            )
        )
        texts = [
            column
            for column in nonnumeric
            if not df[column].dropna().empty
            and df[column].dropna().map(lambda value: isinstance(value, str)).all()
        ]
        chart_type = (
            "line"
            if dates and any(column in numeric for column in chosen_y)
            else "bar"
            if len(texts) == 1 and numeric
            else "table"
        )
    if chart_type == "pie" and len(chosen_y) != 1:
        raise ValueError("Pie charts require exactly one y column")
    chosen_x = x if x is not None else (None if chart_type == "table" else candidate)
    data = [
        {column: _value(value) for column, value in row.items()}
        for row in df.head(MAX_CHART_ROWS).to_dict(orient="records")
    ]
    return ChartSpec(
        chart_type=chart_type, title=title, x_field=chosen_x, y_fields=chosen_y, data=data
    )
