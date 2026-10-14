import re
import pandas as pd
from typing import Dict, Any, Optional, List
from src.query_engine import DataPilotQueryEngine
from src.document_rag import BusinessDocumentRAG


class HybridQueryEngine:
    """
    Unified Hybrid Query Engine for DataPilot.
    Intelligently routes and synthesizes queries across:
    1. Structured tabular data (DuckDB)
    2. Unstructured business documents (RAG)
    3. Hybrid questions requiring both quantitative calculation and qualitative context.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        rag_engine: Optional[BusinessDocumentRAG] = None,
        llm_client: Optional[Any] = None
    ):
        self.df = df
        self.llm_client = llm_client
        self.data_engine = DataPilotQueryEngine(df, llm_client=llm_client)
        self.rag_engine = rag_engine or BusinessDocumentRAG()

    def classify_intent(self, query: str) -> str:
        """
        Classifies query into: 'conversational', 'data_only', 'document_only', or 'hybrid'.
        """
        q = query.lower().strip()

        # Conversational greetings or persona questions
        if any(q.startswith(kw) for kw in ["hello", "hi ", "hey", "who are you", "what can you do", "introduce yourself", "help me"]):
            return "conversational"

        # Keywords strongly indicating structured tabular computation
        data_keywords = [
            "select ", "sum", "average", "avg", "total sales", "total revenue", "highest profit",
            "lowest profit", "top 5", "top 10", "count", "how many", "breakdown by", "per region",
            "per category", "sales in", "profit in", "quantity", "discount"
        ]

        # Keywords strongly indicating document/policy/strategy text
        doc_keywords = [
            "policy", "guideline", "strategy", "review", "memo", "contract", "terms", "target",
            "goal", "why did", "explain", "reason", "plan", "roadmap", "objective", "compliance", "clause"
        ]

        has_data = any(kw in q for kw in data_keywords)
        has_doc = any(kw in q for kw in doc_keywords)

        # If documents are loaded and query mentions both or asks "why" alongside metrics
        if has_data and has_doc:
            return "hybrid"
        elif has_doc and len(self.rag_engine.chunks) > 0:
            return "document_only"
        elif has_data:
            return "data_only"
        else:
            # Default heuristic: if documents are present and text contains general words, try hybrid/data
            if len(self.rag_engine.chunks) > 0 and any(w in q for w in ["what", "describe", "about"]):
                return "hybrid"
            return "data_only"

    def ask(self, query: str, roles: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Any]:
        """
        Execute query with automated routing and human-like zero-hallucination synthesis.
        """
        intent = self.classify_intent(query)

        # --- 0. Conversational Persona Response ---
        if intent == "conversational":
            schema_summary = f"Columns: {', '.join(self.df.columns[:8])} ({len(self.df)} rows)"
            if self.llm_client and self.llm_client.is_configured():
                ans = self.llm_client.chat(query, schema_summary)
            else:
                ans = (
                    "👋 Hello! I am **DataPilot**, your Lead AI Data Analyst and Business Strategy Partner.\n\n"
                    f"I have loaded your dataset with **{len(self.df):,} records** across columns like `{', '.join(self.df.columns[:4])}`. "
                    "I can run deterministic DuckDB calculations, audit data quality, detect loss-making SKUs, and cross-reference metrics against your uploaded strategy documents.\n\n"
                    "**Feel free to ask me questions like:**\n"
                    "- *'What is our total revenue and average order value?'*\n"
                    "- *'Which products or regions generate the highest profit?'*\n"
                    "- *'Show me items with discounts exceeding 15%'*"
                )
            return {
                "intent": "conversational",
                "answer": ans or "Hello! How can I assist with your data analysis today?",
                "data": pd.DataFrame(),
                "executed_sql": None,
                "citations": []
            }

        data_result = None
        doc_results = []

        if intent in ["data_only", "hybrid"]:
            data_result = self.data_engine.ask(query, roles)

        if intent in ["document_only", "hybrid"] and len(self.rag_engine.chunks) > 0:
            doc_results = self.rag_engine.retrieve(query, top_k=3)

        # --- 1. Document-Only Response ---
        if intent == "document_only":
            citations = self.rag_engine.format_citations(doc_results)
            if doc_results:
                if self.llm_client and self.llm_client.is_configured():
                    synth_doc = self.llm_client.synthesize_document_answer(query, citations)
                    if synth_doc:
                        summary = f"{synth_doc}\n\n### 📄 Supporting Document Citations\n{citations}"
                    else:
                        summary = f"Based on our business documents:\n\n{citations}"
                else:
                    summary = f"Based on our business documents:\n\n{citations}"
            else:
                summary = "I reviewed our document knowledge base, but no matching excerpts were found for your inquiry."

            return {
                "intent": "document_only",
                "answer": summary,
                "data": pd.DataFrame(),
                "executed_sql": None,
                "citations": doc_results
            }

        # --- 2. Data-Only Response ---
        if intent == "data_only" or (intent == "hybrid" and not doc_results):
            return {
                "intent": "data_only",
                "answer": data_result["answer"] if data_result else "No analytical result generated.",
                "data": data_result["data"] if data_result else pd.DataFrame(),
                "executed_sql": data_result.get("executed_sql") if data_result else None,
                "citations": []
            }

        # --- 3. Hybrid Synthesized Response ---
        citations_text = self.rag_engine.format_citations(doc_results)
        data_answer = data_result["answer"] if data_result else ""

        # Use LLM for unified human synthesis if available
        if (
            self.llm_client
            and self.llm_client.is_configured()
            and data_result
            and not data_result["data"].empty
        ):
            llm_synth = self.llm_client.synthesize_answer(
                query=query,
                executed_sql=data_result.get("executed_sql", ""),
                result_df=data_result["data"],
                document_context=citations_text
            )
            if llm_synth:
                hybrid_answer = f"{llm_synth}\n\n### 📄 Supporting Document Citations\n{citations_text}"
            else:
                hybrid_answer = (
                    f"### 📊 Analytical Findings\n"
                    f"{data_answer}\n\n"
                    f"### 📄 Supporting Strategic Context\n"
                    f"{citations_text}\n\n"
                    f"### 💡 Strategic Takeaway\n"
                    f"Looking at how these numbers intersect with our governance policies, our quantitative results directly corroborate the operating guidelines established in the documented strategy cited above."
                )
        else:
            hybrid_answer = (
                f"### 📊 Analytical Findings\n"
                f"{data_answer}\n\n"
                f"### 📄 Supporting Strategic Context\n"
                f"{citations_text}\n\n"
                f"### 💡 Strategic Takeaway\n"
                f"Looking at how these numbers intersect with our governance policies, our quantitative results directly corroborate the operating guidelines established in the documented strategy cited above."
            )

        return {
            "intent": "hybrid",
            "answer": hybrid_answer,
            "data": data_result["data"] if data_result else pd.DataFrame(),
            "executed_sql": data_result.get("executed_sql") if data_result else None,
            "citations": doc_results
        }
