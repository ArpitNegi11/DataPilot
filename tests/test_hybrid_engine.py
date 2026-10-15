import pandas as pd
import pytest
from src.hybrid_engine import HybridQueryEngine
from src.document_rag import BusinessDocumentRAG


@pytest.fixture
def hybrid_setup(tmp_path):
    df = pd.DataFrame({
        "Product": ["Cloud Pro", "Basic Tier"],
        "Sales": [50000.0, 12000.0],
        "Profit": [15000.0, 1000.0],
        "Region": ["North", "South"]
    })

    doc_file = str(tmp_path / "annual_review.txt")
    with open(doc_file, "w", encoding="utf-8") as f:
        f.write("Annual Strategy: Cloud Pro is our flagship product targeting enterprise clients with expected 30% margins.")

    rag = BusinessDocumentRAG()
    rag.add_document(doc_file, "annual_review.txt")

    engine = HybridQueryEngine(df=df, rag_engine=rag)
    return engine


def test_hybrid_classification(hybrid_setup):
    engine = hybrid_setup

    assert engine.classify_intent("What is the total revenue?") == "data_only"
    assert engine.classify_intent("What is the policy for annual strategy?") == "document_only"
    assert engine.classify_intent("What is our total revenue and how does it align with annual strategy?") == "hybrid"


def test_hybrid_query_execution(hybrid_setup):
    engine = hybrid_setup
    roles = {
        "product": "Product",
        "revenue": "Sales",
        "profit": "Profit",
        "region": "Region",
        "category": None,
        "date": None,
        "quantity": None
    }

    # Hybrid question execution
    result = engine.ask("What is total revenue and what is our annual strategy?", roles)
    assert result["intent"] == "hybrid"
    assert "$62,000.00" in result["answer"]
    assert "annual_review.txt" in result["answer"]
    assert len(result["citations"]) > 0
