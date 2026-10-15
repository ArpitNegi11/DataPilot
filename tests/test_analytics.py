import pandas as pd
import pytest
from src import analytics


def test_infer_business_columns():
    df = pd.DataFrame({
        "Order_Date": ["2023-01-01", "2023-01-02"],
        "Sales": [100.0, 200.0],
        "Profit": [20.0, 50.0],
        "Quantity": [2, 5],
        "Product_Name": ["Widget A", "Widget B"],
        "Category": ["Office", "Tech"],
        "Region": ["East", "West"]
    })

    roles = analytics.infer_business_columns(df)
    assert roles["date"] == "Order_Date"
    assert roles["revenue"] == "Sales"
    assert roles["profit"] == "Profit"
    assert roles["quantity"] == "Quantity"
    assert roles["product"] == "Product_Name"
    assert roles["category"] == "Category"
    assert roles["region"] == "Region"


def test_compute_kpis():
    df = pd.DataFrame({
        "Sales": [100.0, 300.0],
        "Profit": [20.0, 60.0],
        "Quantity": [2, 3],
        "Product": ["P1", "P2"]
    })

    kpis = analytics.compute_kpis(df)
    assert kpis["total_records"] == 2
    assert kpis["total_revenue"] == 400.0
    assert kpis["total_profit"] == 80.0
    assert kpis["profit_margin_pct"] == 20.0  # 80 / 400 * 100
    assert kpis["total_quantity"] == 5.0
    assert kpis["average_order_value"] == 200.0
    assert kpis["unique_products"] == 2


def test_compute_breakdowns():
    df = pd.DataFrame({
        "Category": ["Tech", "Tech", "Office"],
        "Sales": [500.0, 300.0, 200.0]
    })

    breakdowns = analytics.compute_breakdowns(df)
    assert "category" in breakdowns
    cat_df = breakdowns["category"]
    assert cat_df.iloc[0]["Category"] == "Tech"
    assert cat_df.iloc[0]["Total_Revenue"] == 800.0


def test_compute_data_quality_score():
    df_raw = pd.DataFrame({
        "a": [1, 2, 2, None],
        "b": ["x", "y", "y", "z"]
    })
    df_clean = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "z"]
    })

    score_info = analytics.compute_data_quality_score(
        df_raw=df_raw,
        df_clean=df_clean,
        validation_issues={"a": ["issue 1"]},
        anomalies={"a": [100]}
    )

    assert "score" in score_info
    assert "grade" in score_info
    assert 0 <= score_info["score"] <= 100
