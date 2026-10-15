import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.llm_client import LLMClient
from src.query_engine import DataPilotQueryEngine
from src.hybrid_engine import HybridQueryEngine
from src.orchestrator import DataPilotOrchestrator


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Product": ["Alpha", "Beta", "Gamma"],
        "Sales": [100.0, 250.0, 400.0],
        "Profit": [20.0, 75.0, 120.0],
        "Region": ["East", "West", "East"]
    })


def test_llm_client_configuration():
    # 1. Unconfigured deterministic client
    client_default = LLMClient()
    assert not client_default.is_configured()

    # 2. Configured Gemini client
    client_gemini = LLMClient(provider="Google Gemini", api_key="test_gemini_key")
    assert client_gemini.is_configured()

    # 3. Configured OpenAI client
    client_openai = LLMClient(provider="OpenAI", api_key="test_openai_key")
    assert client_openai.is_configured()

    # 4. Ollama (Local) does not require remote api key
    client_ollama = LLMClient(provider="Ollama (Local)")
    assert client_ollama.is_configured()


def test_llm_sql_execution_and_synthesis(sample_df):
    mock_client = LLMClient(provider="Google Gemini", api_key="dummy_key")
    mock_client._call_llm = MagicMock(side_effect=[
        'SELECT "Product", "Sales" FROM dataset WHERE "Region" = \'East\' ORDER BY "Sales" DESC',  # generate_sql
        "The East region generated top sales from Gamma ($400.00) followed by Alpha ($100.00)."      # synthesize_answer
    ])

    engine = DataPilotQueryEngine(sample_df, llm_client=mock_client)
    res = engine.ask("Show sales for East region")

    assert res["intent"] == "llm_sql"
    assert "Gamma" in res["answer"]
    assert len(res["data"]) == 2
    assert "Sales" in res["executed_sql"]


def test_llm_fallback_on_sql_error(sample_df):
    mock_client = LLMClient(provider="OpenAI", api_key="dummy_key")
    # Simulate LLM returning invalid SQL
    mock_client._call_llm = MagicMock(return_value="SELECT * FROM non_existent_table")

    roles = {"revenue": "Sales", "profit": "Profit", "product": "Product"}
    engine = DataPilotQueryEngine(sample_df, llm_client=mock_client)

    # Should catch SQL failure and smoothly fall back to deterministic pattern matching
    res = engine.ask("What is the total revenue?", roles)
    assert res["intent"] == "total_metric"
    assert "$750.00" in res["answer"]


def test_orchestrator_with_llm(sample_df):
    mock_client = LLMClient(provider="Google Gemini", api_key="dummy_key")
    mock_client._call_llm = MagicMock(side_effect=[
        'SELECT SUM("Sales") AS "Total_Sales" FROM dataset',
        "Overall total sales across all regions amounts to **$750.00**."
    ])

    roles = {"revenue": "Sales", "profit": "Profit", "product": "Product"}
    orchestrator = DataPilotOrchestrator(df=sample_df, llm_client=mock_client)
    result = orchestrator.run("What is the total revenue across all regions?", roles)

    assert result["validation_passed"] is True
    assert "$750.00" in result["answer"]
    assert not result["data"].empty
