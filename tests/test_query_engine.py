import pandas as pd
import pytest
from src.query_engine import DataPilotQueryEngine


@pytest.fixture
def sample_sales_df():
    return pd.DataFrame({
        "Order_Date": ["2023-01-15", "2023-06-20", "2024-02-10", "2024-08-05"],
        "Product": ["Widget A", "Widget B", "Widget A", "Widget C"],
        "Category": ["Hardware", "Hardware", "Hardware", "Software"],
        "Sales": [100.0, 300.0, 150.0, 250.0],
        "Profit": [20.0, 90.0, 30.0, 60.0],
        "Region": ["North", "South", "North", "West"]
    })


def test_query_engine_expanded_features(sample_sales_df):
    roles = {
        "product": "Product",
        "category": "Category",
        "revenue": "Sales",
        "profit": "Profit",
        "region": "Region",
        "date": "Order_Date",
        "quantity": None
    }

    engine = DataPilotQueryEngine(sample_sales_df)

    # 1. Total revenue
    res_rev = engine.ask("What is the total revenue?", roles)
    assert "$800.00" in res_rev["answer"]
    assert res_rev["intent"] == "total_metric"

    # 2. Date filtering (e.g. 2023)
    res_2023 = engine.ask("Total sales in 2023", roles)
    assert "$400.00" in res_2023["answer"]
    assert "2023" in res_2023["executed_sql"]

    # 3. Average metric
    res_avg = engine.ask("What is the average sale?", roles)
    assert "$200.00" in res_avg["answer"]
    assert res_avg["intent"] == "average_metric"

    # 4. Top and bottom performers
    res_top = engine.ask("Which product generated the highest profit?", roles)
    assert "Widget B" in res_top["answer"]

    res_bottom = engine.ask("What is the lowest sales product?", roles)
    assert res_bottom["intent"] == "bottom_performer"

    # 5. Direct SQL pass-through
    res_sql = engine.ask("SELECT COUNT(*) AS total_count FROM dataset", roles)
    assert res_sql["intent"] == "direct_sql"
    assert res_sql["data"].iloc[0]["total_count"] == 4

    # 6. Count query
    res_count = engine.ask("How many orders are there?", roles)
    assert "4 records" in res_count["answer"]
