import pandas as pd
import numpy as np
import pytest

from src.goal_seek import GoalSeekOptimizer
from src.root_cause import RootCauseDecomposer
from src.audio_briefing import AudioBriefingEngine
from src.affinity_miner import MarketBasketAffinityMiner
from src.visualize import plot_waterfall_chart, plot_affinity_matrix, plot_goal_seek_comparison
from src.hybrid_engine import HybridQueryEngine


@pytest.fixture
def sample_business_df():
    return pd.DataFrame({
        "Order_Date": ["2023-01-01", "2023-01-15", "2023-02-01", "2023-02-15", "2023-03-01", "2023-03-15"],
        "Order_ID": ["ORD1", "ORD1", "ORD2", "ORD2", "ORD3", "ORD3"],
        "Customer_ID": ["C1", "C1", "C2", "C2", "C3", "C3"],
        "Category": ["Technology", "Office", "Technology", "Office", "Technology", "Office"],
        "Product_Name": ["Laptop Pro", "Desk Chair", "Laptop Pro", "Paper Pack", "Laptop Pro", "Desk Chair"],
        "Sales": [1200.0, 300.0, 1150.0, 50.0, 1300.0, 320.0],
        "Profit": [240.0, 60.0, 180.0, 10.0, 260.0, 65.0],
        "Quantity": [1, 2, 1, 5, 1, 2],
        "Discount": [0.0, 0.05, 0.10, 0.0, 0.0, 0.05]
    })


def test_goal_seek_optimizer(sample_business_df):
    optimizer = GoalSeekOptimizer(sample_business_df)
    res = optimizer.optimize(target_metric="profit", target_change_pct=15.0)

    assert res["success"] is True
    assert "baseline" in res
    assert "projected" in res
    assert "category_levers" in res
    assert "directives" in res
    assert len(res["directives"]) > 0
    assert len(res["category_levers"]) > 0
    assert res["projected"]["profit"] >= res["baseline"]["profit"]


def test_root_cause_decomposer(sample_business_df):
    decomposer = RootCauseDecomposer(sample_business_df)
    periods = decomposer.get_available_periods()
    assert len(periods) >= 2

    res = decomposer.decompose(metric="profit")
    assert "decomposition" in res
    assert "volume_effect" in res["decomposition"]
    assert "price_effect" in res["decomposition"]
    assert "cost_effect" in res["decomposition"]
    assert "mix_effect" in res["decomposition"]
    assert "executive_narrative" in res
    assert len(res["executive_narrative"]) > 20
    assert "waterfall_data" in res


def test_audio_briefing_engine(sample_business_df):
    engine = AudioBriefingEngine()
    summary = {"Cleaned Rows": len(sample_business_df)}
    kpis = {"total_revenue": 4320.0, "total_profit": 815.0, "profit_margin_pct": 18.8}
    quality_score = {"score": 98, "grade": "A+"}
    anomalies = []
    recommendations = [{"title": "Optimize Technology Pricing", "action": "Raise price by 4%"}]

    briefing = engine.get_briefing(summary, kpis, quality_score, anomalies, recommendations)
    assert "script" in briefing
    assert len(briefing["script"]) > 50
    assert "audio_bytes" in briefing
    assert len(briefing["audio_bytes"]) > 100
    assert briefing["mime_type"] in ["audio/mp3", "audio/wav"]
    assert briefing["word_count"] > 20


def test_market_basket_affinity_miner(sample_business_df):
    miner = MarketBasketAffinityMiner(sample_business_df)
    res = miner.mine_affinity_and_cannibalization(min_support=0.01, min_lift=0.5)

    assert "total_baskets" in res
    assert res["total_baskets"] >= 3
    assert "association_rules" in res
    assert "affinity_matrix" in res
    assert "cannibalization_alerts" in res


def test_new_visualizers():
    waterfall_data = {
        "labels": ["Base", "Volume", "Price", "Current"],
        "values": [1000, 200, -50, 1150],
        "measures": ["absolute", "relative", "relative", "total"]
    }
    fig_wf = plot_waterfall_chart(waterfall_data, "Test Waterfall")
    assert fig_wf is not None

    affinity_data = {
        "ItemA": {"ItemA": 1.0, "ItemB": 2.5},
        "ItemB": {"ItemA": 2.5, "ItemB": 1.0}
    }
    fig_aff = plot_affinity_matrix(affinity_data, "Test Affinity")
    assert fig_aff is not None

    goal_seek_res = {
        "baseline": {"revenue": 1000, "profit": 200, "cost": 800},
        "projected": {"revenue": 1100, "profit": 250, "cost": 850}
    }
    fig_gs = plot_goal_seek_comparison(goal_seek_res, "Test Goal Seek")
    assert fig_gs is not None


def test_hybrid_engine_routes_advanced_intents(sample_business_df):
    hybrid = HybridQueryEngine(sample_business_df)

    # 1. Root Cause
    intent_rc = hybrid.classify_intent("Why did profit decline last month?")
    assert intent_rc == "root_cause"
    res_rc = hybrid.ask("Why did profit decline last month?")
    assert "Diagnostic Root-Cause Analysis" in res_rc["answer"]

    # 2. Goal Seek
    intent_gs = hybrid.classify_intent("Goal seek: how to increase profit by 15%?")
    assert intent_gs == "goal_seek"
    res_gs = hybrid.ask("Goal seek: how to increase profit by 15%?")
    assert "Autonomous Goal-Seek Plan" in res_gs["answer"]

    # 3. Affinity
    intent_aff = hybrid.classify_intent("Which items are frequently bought together?")
    assert intent_aff == "affinity"
    res_aff = hybrid.ask("Which items are frequently bought together?")
    assert "Market Basket Affinity" in res_aff["answer"]
