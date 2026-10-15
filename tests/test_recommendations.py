import pandas as pd
import pytest
from src.recommendations import RecommendationEngine


def test_recommendation_negative_profit():
    df = pd.DataFrame({
        "Product": ["Profitable Widget", "Loss Leader A", "Loss Leader B"],
        "Sales": [10000.0, 3000.0, 2000.0],
        "Profit": [3000.0, -500.0, -300.0],
        "Region": ["East", "West", "Central"]
    })

    rec_engine = RecommendationEngine(df)
    recs = rec_engine.generate_recommendations()

    assert len(recs) > 0
    # Check for loss-making recommendation
    loss_rec = next((r for r in recs if "Loss-Making" in r["title"]), None)
    assert loss_rec is not None
    assert loss_rec["priority"] == "High"
    assert "$800.00" in loss_rec["observation"]  # 500 + 300


def test_recommendation_revenue_concentration():
    df = pd.DataFrame({
        "Product": ["Mega Product", "Small Item A", "Small Item B"],
        "Sales": [90000.0, 5000.0, 5000.0],  # Mega Product has 90% share
        "Profit": [20000.0, 1000.0, 1000.0]
    })

    rec_engine = RecommendationEngine(df)
    recs = rec_engine.generate_recommendations()

    conc_rec = next((r for r in recs if "Revenue Concentration" in r["title"]), None)
    assert conc_rec is not None
    assert "Mega Product" in conc_rec["title"]
