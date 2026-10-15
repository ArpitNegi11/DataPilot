import os
import re
import pandas as pd
from typing import Optional, Dict, Any, List

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class LLMClient:
    """
    Unified LLM Client for DataPilot supporting:
    - Google Gemini (google-genai)
    - OpenAI (OpenAI API)
    - Ollama / Local (OpenAI-compatible local endpoint)
    - Fallback: Gracefully defers to deterministic SQL engine if LLM is unavailable.
    """

    def __init__(
        self,
        provider: str = "Deterministic Engine (No API Key)",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        self.model = model or ("gemini-3.5-flash" if provider == "Google Gemini" else ("gpt-4o-mini" if provider == "OpenAI" else None))
        self.base_url = base_url

    def is_configured(self) -> bool:
        """Check if an active LLM provider with API key or local endpoint is configured."""
        if self.provider == "Ollama (Local)":
            return True
        if self.provider in ["Google Gemini", "OpenAI"] and bool(self.api_key.strip()):
            return True
        return False

    def generate_sql(self, query: str, df: pd.DataFrame, roles: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Use the configured LLM to translate a natural language question into valid DuckDB SQL.
        Returns the SQL string or None if LLM is unavailable or fails.
        """
        if not self.is_configured():
            return None

        # Build schema representation
        schema_lines = []
        for col, dtype in df.dtypes.items():
            sample_vals = df[col].dropna().head(3).tolist()
            sample_str = ", ".join([str(v) for v in sample_vals])
            schema_lines.append(f'  - "{col}" ({dtype}): sample values [{sample_str}]')
        schema_text = "\n".join(schema_lines)

        roles_text = ""
        if roles:
            active_roles = [f'  - {k}: "{v}"' for k, v in roles.items() if v]
            if active_roles:
                roles_text = "\nKEY BUSINESS COLUMN ROLES:\n" + "\n".join(active_roles) + "\n"

        prompt = f"""You are a senior data analyst and DuckDB SQL expert.
You have an in-memory DuckDB table named `dataset` with the following schema:
{schema_text}
{roles_text}
USER QUESTION: "{query}"

RULES:
1. Generate a single valid DuckDB SQL query that directly answers the user's question.
2. Always quote column names with double quotes like "Column Name".
3. The table name MUST be `dataset`.
4. Output ONLY the raw SQL query. Do NOT add markdown, explanations, or quotes.
"""

        try:
            raw_response = self._call_llm(prompt)
            if not raw_response:
                return None

            # Clean markdown code blocks if present
            sql = raw_response.strip()
            sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE).strip()
            sql = re.sub(r"\s*```$", "", sql).strip()
            # Remove trailing semicolon for DuckDB compatibility
            sql = sql.rstrip(";")
            return sql
        except Exception as e:
            print(f"[LLM_WARN] Error generating SQL via {self.provider}: {e}")
            return None

    def synthesize_answer(
        self,
        query: str,
        executed_sql: str,
        result_df: pd.DataFrame,
        document_context: Optional[str] = None
    ) -> Optional[str]:
        """
        Use the configured LLM to synthesize a natural, conversational response grounded in the real DuckDB result.
        Speaks like an insightful, perceptive senior data analyst presenting findings to an executive peer.
        """
        if not self.is_configured() or result_df.empty:
            return None

        data_preview = result_df.head(15).to_string(index=False)
        doc_section = f"\nRELEVANT BUSINESS DOCUMENTS & POLICIES:\n{document_context}\n" if document_context else ""

        prompt = f"""You are DataPilot, a senior Lead Data Analyst and strategic business advisor speaking directly with an executive colleague.
You speak naturally, warmly, and incisively—like an experienced human colleague walking someone through their data in a briefing, NOT like an automated robotic database dump.

USER QUESTION: "{query}"
EXECUTED DUCKDB SQL:
{executed_sql}

VERIFIED QUERY RESULTS (computed deterministically via DuckDB):
{data_preview}
{doc_section}
INSTRUCTIONS FOR YOUR RESPONSE:
1. Speak Personably as a Senior Data Analyst: Talk naturally and conversationally in the first person ("Looking at our numbers...", "Here is what stands out...", "Our data reveals that..."). Avoid dry robotic formulaic phrasing.
2. Ground Every Number: You MUST cite the exact numbers from the verified query results above. Never invent, extrapolate, or alter any numbers.
3. Provide Insight & Interpretation: Don't just list numbers—explain what they signify. Highlight key drivers, top contributors, contrasts, or margin pressure.
4. Strategic Observation: Provide a short, practical business observation, recommendation, or next question an executive should consider.
5. If relevant business documents or policies are attached above, naturally connect the quantitative findings with the documented strategy or policy guidelines.
6. Format with clean, readable Markdown (bold key metrics, bullet points for comparisons, clear takeaways).
"""

        try:
            ans = self._call_llm(prompt)
            if ans:
                return ans
        except Exception as e:
            print(f"[LLM_WARN] Error synthesizing answer via {self.provider}: {e}")

        return self._generate_data_narrative(query, executed_sql, result_df)

    def _generate_data_narrative(self, query: str, executed_sql: str, result_df: pd.DataFrame) -> str:
        """Construct a smart, conversational human narrative directly from executed DuckDB result."""
        num_rows = len(result_df)
        cols = list(result_df.columns)
        if result_df.empty:
            return f"I executed the query for *'{query}'*, but no matching records were found in the dataset."

        if num_rows == 1:
            row = result_df.iloc[0]
            metric_parts = [f"**{col}**: {row[col]}" for col in cols]
            return f"Looking into your inquiry for *'{query}'*, here are the primary verified metrics from our database:\n\n- " + "\n- ".join(metric_parts)

        # Multi-row result
        top_row = result_df.iloc[0]
        dim_col = cols[0]
        val_col = cols[1] if len(cols) > 1 else cols[0]
        narrative = [
            f"Here are the findings for *'{query}'* across **{num_rows} records**.",
            f"Leading the list is **{top_row[dim_col]}** with a recorded **{val_col}** of **{top_row[val_col]}**."
        ]
        return "\n\n".join(narrative)

    def synthesize_document_answer(
        self,
        query: str,
        document_context: str
    ) -> Optional[str]:
        """
        Synthesize an answer to a qualitative business query using retrieved document citations,
        speaking like a knowledgeable strategic advisor.
        """
        if not self.is_configured() or not document_context.strip():
            return None

        prompt = f"""You are DataPilot, a senior Lead Business Analyst and strategic advisor.
An executive colleague asked you a business strategy or policy question.
Answer based strictly on the retrieved document excerpts below.

USER QUESTION: "{query}"

RETRIEVED DOCUMENT EXCERPTS & POLICIES:
{document_context}

INSTRUCTIONS FOR YOUR RESPONSE:
1. Respond Conversationally as a Person: Speak directly, warmly, and authoritatively like an experienced strategic advisor in a meeting ("Based on our documentation...", "Here is how our current policy addresses this...").
2. Strictly Grounded: Answer using only the verified facts and clauses in the excerpts above. Do not fabricate policies.
3. Cite Sources: Naturally mention which document or policy guideline each point comes from.
4. Clear Synthesis: If the document leaves certain aspects unaddressed, transparently note what is covered and what requires further executive review.
5. Format with polished Markdown.
"""

        try:
            return self._call_llm(prompt)
        except Exception as e:
            print(f"[LLM_WARN] Error answering document query via {self.provider}: {e}")
            return None

    def chat(self, message: str, context: Optional[str] = None) -> Optional[str]:
        """
        Direct conversational response from the Lead Data Analyst persona for general
        questions, introductions, or guidance on what to explore in the dataset.
        """
        if not self.is_configured():
            return None

        context_str = f"\nDATASET CONTEXT PREVIEW:\n{context}\n" if context else ""
        prompt = f"""You are DataPilot, a personable, highly skilled Lead AI Data Analyst & Business Consultant.
You work closely with executives, founders, and business leaders to turn raw business data into actionable growth strategies.
{context_str}
USER INQUIRY: "{message}"

INSTRUCTIONS:
1. Respond naturally, warmly, and conversationally in the first person as a human colleague.
2. If the user is asking about who you are or how to analyze their dataset, explain clearly how you can help (e.g. running DuckDB queries, identifying loss-making SKUs, auditing data quality, and synthesizing policies with numbers).
3. Offer 2-3 specific, compelling sample questions they could ask right now based on their data.
4. Keep the response concise, engaging, and professional.
"""
        try:
            return self._call_llm(prompt)
        except Exception as e:
            print(f"[LLM_WARN] Error in conversational chat via {self.provider}: {e}")
            return None

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Internal dispatcher to the appropriate LLM SDK with candidate model cascade."""
        if self.provider == "Google Gemini":
            from google import genai
            client = genai.Client(api_key=self.api_key)
            candidates = [self.model or "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-2.5-pro"]
            unique_candidates = []
            for c in candidates:
                if c and c not in unique_candidates:
                    unique_candidates.append(c)

            last_err = None
            for model_cand in unique_candidates:
                try:
                    response = client.models.generate_content(
                        model=model_cand,
                        contents=prompt
                    )
                    if response and response.text:
                        return response.text
                except Exception as e:
                    last_err = e
                    print(f"[LLM_WARN] Gemini model '{model_cand}' failed: {e}. Trying fallback model...")
                    continue

            if last_err:
                raise last_err

        elif self.provider in ["OpenAI", "Ollama (Local)"]:
            from openai import OpenAI
            if self.provider == "Ollama (Local)":
                client = OpenAI(
                    base_url=self.base_url or "http://localhost:11434/v1",
                    api_key="ollama"
                )
                model_name = self.model or "llama3.2"
            else:
                client = OpenAI(api_key=self.api_key)
                model_name = self.model or "gpt-4o-mini"

            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            return response.choices[0].message.content

        return None
