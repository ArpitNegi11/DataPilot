import re
import pandas as pd
import duckdb
from typing import Dict, Any, Optional, List


class DataPilotQueryEngine:
    """
    Deterministic Natural Language Query Engine for DataPilot.
    Executes analytical computations directly on the dataset using DuckDB.
    Guarantees zero hallucination of numbers by grounding every metric in executed SQL.
    """

    def __init__(self, df: pd.DataFrame, llm_client: Optional[Any] = None):
        self.df = df
        self.conn = duckdb.connect(database=":memory:")
        self.conn.register("dataset", df)
        self.llm_client = llm_client

    def execute_sql(self, query: str) -> pd.DataFrame:
        """Execute a raw SQL query safely in DuckDB."""
        try:
            return self.conn.execute(query).df()
        except Exception as e:
            raise RuntimeError(f"SQL Execution Error: {str(e)}")

    def ask(self, query: str, roles: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Any]:
        """
        Process a natural language or SQL query against the dataset.
        Returns executed_sql, natural language answer, data DataFrame, and recognized intent.
        """
        q = query.strip()
        q_lower = q.lower()
        cols = list(self.df.columns)

        if not roles:
            from src.analytics import infer_business_columns
            roles = infer_business_columns(self.df)

        rev_col = roles.get("revenue")
        prof_col = roles.get("profit")
        prod_col = roles.get("product")
        cat_col = roles.get("category")
        reg_col = roles.get("region")
        date_col = roles.get("date")
        qty_col = roles.get("quantity")

        # 0. Direct SQL Pass-through
        if q_lower.startswith("select ") or q_lower.startswith("with "):
            try:
                res_df = self.execute_sql(q)
                return {
                    "intent": "direct_sql",
                    "executed_sql": q,
                    "answer": f"Executed custom SQL query returning {len(res_df)} rows.",
                    "data": res_df
                }
            except Exception as e:
                return {
                    "intent": "direct_sql_error",
                    "executed_sql": q,
                    "answer": f"SQL Error: {e}",
                    "data": pd.DataFrame()
                }

        # 0b. LLM-Powered Natural Language Query Translation (if configured)
        if self.llm_client and self.llm_client.is_configured():
            try:
                generated_sql = self.llm_client.generate_sql(q, self.df, roles)
                if generated_sql:
                    res_df = self.execute_sql(generated_sql)
                    synthesized = self.llm_client.synthesize_answer(q, generated_sql, res_df)
                    if not synthesized:
                        synthesized = f"DuckDB executed generated query returning **{len(res_df)} rows**."
                    return {
                        "intent": "llm_sql",
                        "executed_sql": generated_sql,
                        "answer": synthesized,
                        "data": res_df
                    }
            except Exception as e:
                print(f"[QUERY_ENGINE_WARN] LLM query processing failed: {e}. Falling back to deterministic rules.")

        # Check for year filter in query (e.g. 2021, 2022, 2023, 2024, 2025)
        year_match = re.search(r"\b(201\d|202\d)\b", q)
        year_val = year_match.group(1) if year_match else None
        date_where = ""
        if year_val and date_col:
            date_where = f' WHERE CAST("{date_col}" AS VARCHAR) LIKE \'%{year_val}%\''

        # 1. Total revenue / sales
        if any(term in q_lower for term in ["total sales", "total revenue", "total amount", "how much revenue", "how much sales", "overall revenue"]):
            metric = rev_col or prof_col or (cols[0] if cols else "")
            if metric:
                sql = f'SELECT SUM("{metric}") AS "Total_{metric}", AVG("{metric}") AS "Avg_{metric}", COUNT(*) AS "Record_Count" FROM dataset{date_where}'
                res_df = self.execute_sql(sql)
                tot = res_df.iloc[0][f"Total_{metric}"]
                cnt = res_df.iloc[0]["Record_Count"]
                yr_str = f" in {year_val}" if year_val else ""
                val_str = f"${tot:,.2f}" if tot is not None else "$0.00"
                avg_str = f"${(tot / cnt):,.2f}" if cnt and tot is not None else "$0.00"
                return {
                    "intent": "total_metric",
                    "executed_sql": sql,
                    "answer": f"Looking across our dataset, total {metric}{yr_str} reached **{val_str}** across **{cnt:,} records** (averaging approximately **{avg_str}** per transaction).",
                    "data": res_df
                }

        # 2. Average metric / Average Order Value (AOV)
        if any(term in q_lower for term in ["average sale", "average revenue", "average order", "aov", "mean sale", "avg revenue", "average profit"]):
            metric = prof_col if "profit" in q_lower else (rev_col or prof_col)
            if metric:
                sql = f'SELECT AVG("{metric}") AS "Average_{metric}", MIN("{metric}") AS "Min_{metric}", MAX("{metric}") AS "Max_{metric}" FROM dataset{date_where}'
                res_df = self.execute_sql(sql)
                avg_val = res_df.iloc[0][f"Average_{metric}"]
                min_val = res_df.iloc[0][f"Min_{metric}"]
                max_val = res_df.iloc[0][f"Max_{metric}"]
                return {
                    "intent": "average_metric",
                    "executed_sql": sql,
                    "answer": f"Evaluating transaction metrics, our average {metric} is **${avg_val:,.2f}**, with individual orders ranging from **${min_val:,.2f}** up to **${max_val:,.2f}**.",
                    "data": res_df
                }

        # 3. Highest / Top profit or revenue
        if any(term in q_lower for term in ["highest profit", "most profit", "top profit", "best profit", "highest revenue", "most revenue", "top revenue"]):
            target_metric = prof_col if "profit" in q_lower else (rev_col or prof_col)
            target_dim = prod_col or cat_col or cols[0]
            if target_metric and target_dim:
                sql = f'SELECT "{target_dim}", SUM("{target_metric}") AS "Total_{target_metric}" FROM dataset{date_where} GROUP BY "{target_dim}" ORDER BY "Total_{target_metric}" DESC LIMIT 5'
                res_df = self.execute_sql(sql)
                if not res_df.empty:
                    top_name = res_df.iloc[0][target_dim]
                    top_val = res_df.iloc[0][f"Total_{target_metric}"]
                    return {
                        "intent": "top_performer",
                        "executed_sql": sql,
                        "answer": f"Analyzing our performance leaders, the standout top {target_dim} by {target_metric} is **{top_name}** with **${top_val:,.2f}**. Here is the ranking of our top 5 contributors:",
                        "data": res_df
                    }

        # 4. Lowest / Least / Bottom performers
        if any(term in q_lower for term in ["lowest profit", "least profit", "lowest revenue", "least revenue", "lowest sales", "worst selling", "least selling", "lowest selling"]):
            target_metric = prof_col if "profit" in q_lower else (rev_col or prof_col)
            target_dim = prod_col or cat_col or cols[0]
            if target_metric and target_dim:
                sql = f'SELECT "{target_dim}", SUM("{target_metric}") AS "Total_{target_metric}" FROM dataset{date_where} GROUP BY "{target_dim}" ORDER BY "Total_{target_metric}" ASC LIMIT 5'
                res_df = self.execute_sql(sql)
                if not res_df.empty:
                    bottom_name = res_df.iloc[0][target_dim]
                    bottom_val = res_df.iloc[0][f"Total_{target_metric}"]
                    return {
                        "intent": "bottom_performer",
                        "executed_sql": sql,
                        "answer": f"Examining underperforming assets, the lowest {target_dim} by {target_metric} is **{bottom_name}** with **${bottom_val:,.2f}**. Here are the bottom 5 performers for review:",
                        "data": res_df
                    }

        # 5. Multi-metric Category Breakdown (Sales & Profit)
        if any(term in q_lower for term in ["by category", "category sales", "category profit", "sales by category", "profit by category"]):
            dim = cat_col or "Category"
            if dim in self.df.columns and rev_col:
                prof_select = f', SUM("{prof_col}") AS "Total_{prof_col}"' if prof_col and prof_col in self.df.columns else ""
                sql = f'SELECT "{dim}", SUM("{rev_col}") AS "Total_{rev_col}"{prof_select}, COUNT(*) AS "Orders" FROM dataset{date_where} GROUP BY "{dim}" ORDER BY "Total_{rev_col}" DESC'
                res_df = self.execute_sql(sql)
                if not res_df.empty:
                    top_cat = res_df.iloc[0][dim]
                    top_rev = res_df.iloc[0][f"Total_{rev_col}"]
                    return {
                        "intent": "category_breakdown",
                        "executed_sql": sql,
                        "answer": f"Taking a look at our categorical mix, **{top_cat}** leads the portfolio with **${top_rev:,.2f}** in total {rev_col}. Here is the complete breakdown:",
                        "data": res_df
                    }

        # 6. Multi-metric Regional Breakdown
        if any(term in q_lower for term in ["by region", "region sales", "sales by region", "regional breakdown", "profit by region"]):
            dim = reg_col or "Region"
            if dim in self.df.columns and rev_col:
                prof_select = f', SUM("{prof_col}") AS "Total_{prof_col}"' if prof_col and prof_col in self.df.columns else ""
                sql = f'SELECT "{dim}", SUM("{rev_col}") AS "Total_{rev_col}"{prof_select}, COUNT(*) AS "Orders" FROM dataset{date_where} GROUP BY "{dim}" ORDER BY "Total_{rev_col}" DESC'
                res_df = self.execute_sql(sql)
                if not res_df.empty:
                    top_reg = res_df.iloc[0][dim]
                    top_val = res_df.iloc[0][f"Total_{rev_col}"]
                    return {
                        "intent": "regional_breakdown",
                        "executed_sql": sql,
                        "answer": f"Reviewing performance across territories, our strongest regional market is **{top_reg}** generating **${top_val:,.2f}** in total {rev_col}. Here is the regional distribution:",
                        "data": res_df
                    }

        # 7. Top N Products / Items
        if any(term in q_lower for term in ["top 5", "top 10", "top products", "best selling", "top items"]):
            target_metric = rev_col or prof_col or qty_col
            target_dim = prod_col or cols[0]
            limit_n = 10 if "10" in q_lower else 5
            if target_metric and target_dim:
                sql = f'SELECT "{target_dim}", SUM("{target_metric}") AS "Total_{target_metric}" FROM dataset{date_where} GROUP BY "{target_dim}" ORDER BY "Total_{target_metric}" DESC LIMIT {limit_n}'
                res_df = self.execute_sql(sql)
                return {
                    "intent": "top_n_products",
                    "executed_sql": sql,
                    "answer": f"Here are our top {limit_n} {target_dim} ranked by total {target_metric}, highlighting our highest-performing assets:",
                    "data": res_df
                }

        # 8. Record count / Volume query
        if any(term in q_lower for term in ["how many orders", "how many rows", "total records", "number of transactions", "count of orders", "how many customers"]):
            sql = f"SELECT COUNT(*) AS \"Total_Records\" FROM dataset{date_where}"
            res_df = self.execute_sql(sql)
            count_val = res_df.iloc[0]["Total_Records"]
            return {
                "intent": "count_query",
                "executed_sql": sql,
                "answer": f"The database currently tracks **{count_val:,} records**{(' matching ' + year_val) if year_val else ''} across the active dataset.",
                "data": res_df
            }

        # 9. Generic Column Search / Inferred Filtering
        for col in cols:
            col_clean = str(col).lower()
            if col_clean in q_lower and rev_col and col != rev_col:
                sql = f'SELECT "{col}", SUM("{rev_col}") AS "Total_{rev_col}", COUNT(*) AS "Count" FROM dataset{date_where} GROUP BY "{col}" ORDER BY "Total_{rev_col}" DESC LIMIT 10'
                try:
                    res_df = self.execute_sql(sql)
                    return {
                        "intent": "column_aggregation",
                        "executed_sql": sql,
                        "answer": f"Aggregation of {rev_col} grouped by **{col}**:",
                        "data": res_df
                    }
                except Exception:
                    pass

        # 10. Fallback Preview & Query Guidance
        sql = 'SELECT * FROM dataset LIMIT 5'
        res_df = self.execute_sql(sql)
        suggestion_tip = ""
        if not (self.llm_client and self.llm_client.is_configured()):
            suggestion_tip = (
                "\n\n💡 **Tip**: Configure Google Gemini, OpenAI, or Ollama in the sidebar for unlimited open-ended natural language queries, "
                "or try pre-configured patterns like *'total revenue'*, *'sales by category'*, *'highest profit product'*, *'bottom 5 items'*, "
                "or direct SQL like `SELECT * FROM dataset LIMIT 10`."
            )
        return {
            "intent": "fallback",
            "executed_sql": sql,
            "answer": f"Here is a sample preview of the dataset for query *'{query}'*.{suggestion_tip}",
            "data": res_df
        }
