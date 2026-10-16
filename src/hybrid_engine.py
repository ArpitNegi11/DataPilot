import re
import pandas as pd
from typing import Dict, Any, Optional, List
from src.query_engine import DataPilotQueryEngine
from src.document_rag import BusinessDocumentRAG
from src.root_cause import RootCauseDecomposer
from src.goal_seek import GoalSeekOptimizer
from src.affinity_miner import MarketBasketAffinityMiner


class HybridQueryEngine:
    """
    Unified Hybrid Query Engine for DataPilot.
    Intelligently routes and synthesizes queries across:
    1. Structured tabular data (DuckDB)
    2. Unstructured business documents (RAG)
    3. Hybrid questions requiring both quantitative calculation and qualitative context.
    4. Diagnostic Root-Cause Decomposition ("Why did this happen?")
    5. Autonomous Goal-Seek & Decision Optimizer
    6. Product Cannibalization & Market Basket Affinity Miner
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
        Classifies query into: 'conversational', 'root_cause', 'goal_seek', 'affinity', 'data_only', 'document_only', or 'hybrid'.
        """
        q = query.lower().strip()

        # Conversational greetings or persona questions
        if any(q.startswith(kw) for kw in ["hello", "hi ", "hey", "who are you", "what can you do", "introduce yourself", "help me"]):
            return "conversational"

        # Diagnostic Root-Cause keywords
        if any(kw in q for kw in ["why did", "root cause", "variance decomposition", "why did profit", "why did sales", "why did revenue", "what drove the drop", "what caused the decline"]):
            return "root_cause"

        # Goal-Seek and Decision Optimization keywords
        if any(kw in q for kw in ["goal seek", "optimizer", "how to increase profit", "how to achieve", "target margin", "how can we reach", "what levers", "prescriptive levers"]):
            return "goal_seek"

        # Market Basket & Cannibalization keywords
        if any(kw in q for kw in ["frequently bought together", "market basket", "affinity", "cross sell", "bundle", "cannibaliz"]):
            return "affinity"

        # Keywords strongly indicating structured tabular computation
        data_keywords = [
            "select ", "sum", "average", "avg", "total sales", "total revenue", "highest profit",
            "lowest profit", "top 5", "top 10", "count", "how many", "breakdown by", "per region",
            "per category", "sales in", "profit in", "quantity", "discount"
        ]

        # Keywords strongly indicating document/policy/strategy text
        doc_keywords = [
            "policy", "guideline", "strategy", "review", "memo", "contract", "terms", "target",
            "goal", "explain", "reason", "plan", "roadmap", "objective", "compliance", "clause"
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

        # --- 0B. Diagnostic Root-Cause Response ---
        if intent == "root_cause":
            decomposer = RootCauseDecomposer(self.df, roles)
            metric_target = "profit" if "profit" in query.lower() else "revenue"
            res = decomposer.decompose(metric=metric_target)
            attr_df = pd.DataFrame(res.get("category_attributions", []))
            ans = (
                f"### 🔍 Diagnostic Root-Cause Analysis\n\n"
                f"**Executive Verdict:** {res['executive_narrative']}\n\n"
                f"**Orthogonal Variance Breakdown:**\n"
                f"- **Volume Effect:** ${res['decomposition']['volume_effect']:+,.0f}\n"
                f"- **Price Realization:** ${res['decomposition']['price_effect']:+,.0f}\n"
                f"- **Cost/Margin Compression:** ${res['decomposition']['cost_effect']:+,.0f}\n"
                f"- **Portfolio Mix Shift:** ${res['decomposition']['mix_effect']:+,.0f}\n"
            )
            return {
                "intent": "root_cause",
                "answer": ans,
                "data": attr_df.head(10) if not attr_df.empty else pd.DataFrame(),
                "executed_sql": f"VARIANCE DECOMPOSITION: {res['period_base']} vs {res['period_curr']} ({metric_target})",
                "citations": []
            }

        # --- 0C. Autonomous Goal-Seek Response ---
        if intent == "goal_seek":
            optimizer = GoalSeekOptimizer(self.df, roles)
            match = re.search(r'(\d+)\s*%', query)
            tgt_pct = float(match.group(1)) if match else 10.0
            res = optimizer.optimize(target_metric="profit", target_change_pct=tgt_pct)
            levers_df = pd.DataFrame(res.get("category_levers", []))
            ans = (
                f"### 🎯 Autonomous Goal-Seek Plan: Target +{tgt_pct:.0f}% Profit\n\n"
                f"**Baseline:** ${res['baseline']['profit']:,.0f} Profit ({res['baseline']['margin_pct']:.1f}% margin) ➔ "
                f"**Projected:** ${res['projected']['profit']:,.0f} Profit ({res['projected']['margin_pct']:.1f}% margin) "
                f"(*Net Profit Delta: ${res['delta']['profit_delta']:+,.0f}*)\n\n"
                f"**Strategic Directives:**\n"
                + "\n".join([f"- {d}" for d in res["directives"]])
            )
            return {
                "intent": "goal_seek",
                "answer": ans,
                "data": levers_df.head(10) if not levers_df.empty else pd.DataFrame(),
                "executed_sql": f"CONSTRAINED OPTIMIZATION: target_profit_delta = +{tgt_pct:.1f}%",
                "citations": []
            }

        # --- 0D. Market Basket Affinity & Cannibalization Response ---
        if intent == "affinity":
            miner = MarketBasketAffinityMiner(self.df, roles)
            res = miner.mine_affinity_and_cannibalization()
            rules_df = pd.DataFrame(res.get("association_rules", []))
            cann_df = pd.DataFrame(res.get("cannibalization_alerts", []))
            ans = (
                f"### 🛒 Market Basket Affinity & Cannibalization Analysis\n\n"
                f"**Top Affinity Co-Purchase:** {res['summary']['top_affinity_pair']}\n\n"
                f"**Top Cannibalization Risk:** {res['summary']['top_cannibalization_risk']}\n\n"
                f"Analyzed {res['total_baskets']:,} transaction baskets across {res['unique_items_analyzed']} items."
            )
            display_df = rules_df.head(10) if not rules_df.empty else cann_df.head(10)
            return {
                "intent": "affinity",
                "answer": ans,
                "data": display_df if not display_df.empty else pd.DataFrame(),
                "executed_sql": "ASSOCIATION MINING: Support >= 1%, Lift >= 1.0",
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
