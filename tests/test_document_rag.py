import os
import pytest
from src.document_rag import BusinessDocumentRAG


def test_rag_text_chunking_and_retrieval(tmp_path):
    rag = BusinessDocumentRAG(chunk_size=150, chunk_overlap=30)

    # Create sample business document
    doc_path = str(tmp_path / "q4_strategy.txt")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(
            "Quarterly Business Review Q4 2023. Our primary enterprise strategy focuses on high-margin software "
            "licenses in the North American region. Operating expenses decreased by 12% following automation initiatives. "
            "Customer churn in the basic tier was driven by pricing adjustments, while enterprise retention remained at 96%."
        )

    chunks_added = rag.add_document(doc_path, "q4_strategy.txt")
    assert chunks_added > 0
    assert len(rag.chunks) > 0

    # Query 1: Strategy & enterprise retention
    results = rag.retrieve("What was the enterprise retention rate?", top_k=2)
    assert len(results) > 0
    assert "q4_strategy.txt" == results[0]["source"]
    assert "96%" in results[0]["content"]

    # Query 2: Citations formatting
    citation_text = rag.format_citations(results)
    assert "[1] q4_strategy.txt" in citation_text
    assert "96%" in citation_text


def test_rag_empty_or_low_relevance():
    rag = BusinessDocumentRAG()
    results = rag.retrieve("Nonexistent query")
    assert results == []
    assert "No relevant" in rag.format_citations([])
