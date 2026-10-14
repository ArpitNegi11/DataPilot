import pandas as pd
import pytest
from src.orchestrator import DataPilotOrchestrator
from src.document_rag import BusinessDocumentRAG


def test_orchestrator_graph_execution(tmp_path):
    df = pd.DataFrame({
        "Product": ["Item A", "Item B"],
        "Sales": [100.0, 400.0],
        "Profit": [20.0, 80.0]
    })

    doc_path = str(tmp_path / "memo.txt")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("Operational Memo: Target profit margin for wholesale items is 20%.")

    rag = BusinessDocumentRAG()
    rag.add_document(doc_path, "memo.txt")

    orchestrator = DataPilotOrchestrator(df=df, rag_engine=rag)

    # Test 1: Data intent execution through graph
    res_data = orchestrator.run("What is the total revenue?")
    assert res_data["validation_passed"] is True
    assert "$500.00" in res_data["answer"]
    assert res_data["intent"] == "data_only"

    # Test 2: Hybrid intent execution through graph
    res_hybrid = orchestrator.run("What is total revenue and what is our target profit margin in the operational memo?")
    assert res_hybrid["intent"] == "hybrid"
    assert "$500.00" in res_hybrid["answer"]
    assert "memo.txt" in res_hybrid["answer"]
    assert len(res_hybrid["citations"]) > 0
