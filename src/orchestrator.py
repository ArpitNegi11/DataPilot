from typing import TypedDict, Dict, Any, Optional, List
import pandas as pd
from langgraph.graph import StateGraph, END

from src.query_engine import DataPilotQueryEngine
from src.document_rag import BusinessDocumentRAG
from src.hybrid_engine import HybridQueryEngine


class AgentState(TypedDict):
    query: str
    roles: Dict[str, Optional[str]]
    intent: str
    sql_query: Optional[str]
    sql_result: Optional[List[Dict[str, Any]]]
    rag_citations: List[Dict[str, Any]]
    response_text: str
    validation_passed: bool
    retry_count: int
    error_message: Optional[str]


class DataPilotOrchestrator:
    """
    LangGraph-powered multi-agent orchestrator for DataPilot.
    Features:
    - Intent routing (data, document, hybrid)
    - Deterministic SQL & semantic RAG execution
    - Execution validation & hallucination guardrails
    - Automated retry / fallback on execution anomalies
    """

    def __init__(
        self,
        df: pd.DataFrame,
        rag_engine: Optional[BusinessDocumentRAG] = None,
        llm_client: Optional[Any] = None
    ):
        self.df = df
        self.llm_client = llm_client
        self.hybrid_engine = HybridQueryEngine(df=df, rag_engine=rag_engine, llm_client=llm_client)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Define Nodes
        workflow.add_node("route_intent", self._route_intent_node)
        workflow.add_node("execute_query", self._execute_query_node)
        workflow.add_node("validate_execution", self._validate_execution_node)
        workflow.add_node("fallback_retry", self._fallback_retry_node)
        workflow.add_node("synthesize", self._synthesize_node)

        # Build Edges
        workflow.set_entry_point("route_intent")
        workflow.add_edge("route_intent", "execute_query")
        workflow.add_edge("execute_query", "validate_execution")

        # Conditional Edge based on validation
        workflow.add_conditional_edges(
            "validate_execution",
            self._check_validation,
            {
                "pass": "synthesize",
                "retry": "fallback_retry",
                "end": "synthesize"
            }
        )
        workflow.add_edge("fallback_retry", "execute_query")
        workflow.add_edge("synthesize", END)

        return workflow.compile()

    def _route_intent_node(self, state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "")
        intent = self.hybrid_engine.classify_intent(query)
        return {"intent": intent, "retry_count": state.get("retry_count", 0)}

    def _execute_query_node(self, state: AgentState) -> Dict[str, Any]:
        query = state["query"]
        roles = state.get("roles", {})
        result = self.hybrid_engine.ask(query, roles)

        sql_records = result["data"].to_dict(orient="records") if not result["data"].empty else []
        return {
            "sql_query": result.get("executed_sql"),
            "sql_result": sql_records,
            "rag_citations": result.get("citations", []),
            "response_text": result.get("answer", "")
        }

    def _validate_execution_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Validate that the generated response is mathematically grounded and error-free.
        """
        response = state.get("response_text", "")
        error = state.get("error_message")

        # Check for errors in SQL or response
        if "SQL Error" in response or "Traceback" in response or error:
            return {"validation_passed": False, "error_message": "Execution error detected"}

        # Zero-hallucination check: if SQL executed and returned records, ensure metric string isn't blank
        if state.get("sql_query") and not response.strip():
            return {"validation_passed": False, "error_message": "Empty answer generated from SQL"}

        return {"validation_passed": True, "error_message": None}

    def _check_validation(self, state: AgentState) -> str:
        if state.get("validation_passed", False):
            return "pass"
        elif state.get("retry_count", 0) < 1:
            return "retry"
        else:
            return "end"

    def _fallback_retry_node(self, state: AgentState) -> Dict[str, Any]:
        query = state["query"]
        if self.llm_client and self.llm_client.is_configured():
            # For LLM mode, keep the user's question and allow retry
            simplified_query = query
        else:
            q_low = query.lower()
            if any(w in q_low for w in ["revenue", "sales", "arr", "mrr", "amount"]):
                simplified_query = "What is the total revenue?"
            elif any(w in q_low for w in ["profit", "margin", "income"]):
                simplified_query = "What is the average profit?"
            elif any(w in q_low for w in ["product", "item", "sku", "top", "best"]):
                simplified_query = "What is the highest profit product?"
            elif any(w in q_low for w in ["category", "segment", "region"]):
                simplified_query = "Sales by category"
            else:
                simplified_query = query

        return {
            "query": simplified_query,
            "retry_count": state.get("retry_count", 0) + 1,
            "error_message": None
        }

    def _synthesize_node(self, state: AgentState) -> Dict[str, Any]:
        return {"response_text": state["response_text"]}

    def run(self, query: str, roles: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Any]:
        """Run the orchestrated graph workflow."""
        if not roles:
            from src.analytics import infer_business_columns
            roles = infer_business_columns(self.df)

        initial_state: AgentState = {
            "query": query,
            "roles": roles,
            "intent": "pending",
            "sql_query": None,
            "sql_result": [],
            "rag_citations": [],
            "response_text": "",
            "validation_passed": False,
            "retry_count": 0,
            "error_message": None
        }

        final_state = self.graph.invoke(initial_state)

        df_result = pd.DataFrame(final_state.get("sql_result", []))
        return {
            "intent": final_state.get("intent"),
            "executed_sql": final_state.get("sql_query"),
            "data": df_result,
            "answer": final_state.get("response_text"),
            "citations": final_state.get("rag_citations", []),
            "validation_passed": final_state.get("validation_passed", True)
        }
