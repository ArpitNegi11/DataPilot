import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any

def infer_business_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Infer column roles: date, revenue, profit, quantity, product, category, region.
    """
    roles = {
        "date": None,
        "revenue": None,
        "profit": None,
        "quantity": None,
        "product": None,
        "category": None,
        "region": None,
    }

    cols = list(df.columns)
    cols_lower = {str(c).lower().replace(" ", "_"): c for c in cols}

    # First pass: exact matches and high priority
    for key, orig in cols_lower.items():
        if not roles["date"] and ("date" in key or "time" in key or pd.api.types.is_datetime64_any_dtype(df[orig])):
            roles["date"] = orig
        elif not roles["revenue"] and key in ["sales", "revenue", "total", "total_(usd)", "subscription_cost", "cost_price_total_(usd)"]:
            roles["revenue"] = orig
        elif not roles["profit"] and "profit" in key:
            roles["profit"] = orig
        elif not roles["quantity"] and key in ["quantity", "qty", "units", "units_sold", "number_of_units"]:
            roles["quantity"] = orig
        elif not roles["product"] and key in ["product_name", "product", "item", "item_name", "product_id"]:
            roles["product"] = orig
        elif not roles["category"] and key in ["category", "segment", "subscription_interval", "department", "type"]:
            roles["category"] = orig
        elif not roles["region"] and key in ["region", "country", "city", "location", "rep"]:
            roles["region"] = orig

    # Second pass: Fallbacks
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not roles["revenue"] and num_cols:
        roles["revenue"] = num_cols[-1]

    text_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not roles["product"] and text_cols:
        roles["product"] = text_cols[0]
    if not roles["category"] and len(text_cols) > 1:
        roles["category"] = text_cols[1]

    return roles


def generate_business_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive, data-driven business insights grounded in actual dataset numbers.
    Computes key performance indicators, Pareto concentrations, margin health,
    trend velocity, and strategic recommendations.
    """
    roles = infer_business_columns(df)
    insights = {
        "roles": roles,
        "kpis": {},
        "highlights": [],
        "drivers": [],
        "risks": [],
        "recommendations": []
    }

    rev_col = roles.get("revenue")
    prof_col = roles.get("profit")
    qty_col = roles.get("quantity")
    prod_col = roles.get("product")
    cat_col = roles.get("category")
    date_col = roles.get("date")
    reg_col = roles.get("region")

    total_records = len(df)
    insights["kpis"]["total_records"] = total_records

    # 1. Revenue & Profit Margin Analysis
    if rev_col and rev_col in df.columns:
        tot_rev = float(df[rev_col].sum())
        avg_rev = float(df[rev_col].mean())
        insights["kpis"]["total_revenue"] = tot_rev
        insights["kpis"]["avg_revenue"] = avg_rev

        insights["highlights"].append({
            "title": f"Total {rev_col}",
            "value": f"${tot_rev:,.2f}" if tot_rev >= 100 else f"{tot_rev:,.2f}",
            "subtitle": f"Averaging ${avg_rev:,.2f} across {total_records:,} transactions",
            "type": "positive"
        })

        if prof_col and prof_col in df.columns:
            tot_prof = float(df[prof_col].sum())
            margin_pct = (tot_prof / tot_rev * 100) if tot_rev != 0 else 0.0
            insights["kpis"]["total_profit"] = tot_prof
            insights["kpis"]["profit_margin"] = margin_pct

            insights["highlights"].append({
                "title": "Overall Profit Margin",
                "value": f"{margin_pct:.1f}%",
                "subtitle": f"${tot_prof:,.2f} total net profit",
                "type": "positive" if margin_pct > 15 else ("neutral" if margin_pct > 5 else "warning")
            })

            loss_rows = df[df[prof_col] < 0]
            if len(loss_rows) > 0:
                loss_rev = float(loss_rows[rev_col].sum())
                loss_pct = (len(loss_rows) / total_records) * 100
                insights["risks"].append(
                    f"⚠️ **Margin Erosion**: {len(loss_rows):,} transactions ({loss_pct:.1f}%) generated negative profit, representing ${loss_rev:,.2f} in revenue sold at a loss."
                )
                insights["recommendations"].append(
                    f"Audit the {len(loss_rows)} loss-making transactions to eliminate excessive discounts or renegotiate unit costs."
                )

    # 2. Pareto 80/20 Driver Analysis
    target_dim = prod_col or cat_col
    if target_dim and rev_col and target_dim in df.columns and rev_col in df.columns:
        grouped = df.groupby(target_dim)[rev_col].sum().sort_values(ascending=False)
        if not grouped.empty and grouped.sum() > 0:
            top_entity = grouped.index[0]
            top_val = float(grouped.iloc[0])
            top_share = (top_val / grouped.sum()) * 100

            insights["drivers"].append(
                f"🏆 **Top Performer**: **{top_entity}** is the primary driver, delivering **${top_val:,.2f}** ({top_share:.1f}% of total {rev_col})."
            )

            # 80/20 rule calculation
            n_20 = max(1, int(round(len(grouped) * 0.2)))
            top_20_share = (grouped.head(n_20).sum() / grouped.sum()) * 100
            insights["drivers"].append(
                f"⚖️ **Pareto Concentration (80/20)**: The top {n_20} {target_dim}(s) (top 20% of catalog) account for **{top_20_share:.1f}%** of total volume."
            )

            if top_share > 30:
                insights["risks"].append(
                    f"⚠️ **Concentration Risk**: High reliance on **{top_entity}** ({top_share:.1f}% of volume). Diversifying catalog sales is strongly advised."
                )
                insights["recommendations"].append(
                    f"Cross-sell secondary product lines alongside **{top_entity}** to mitigate reliance on a single champion product."
                )

    # 3. Regional / Category Share
    if reg_col and rev_col and reg_col in df.columns and rev_col in df.columns:
        reg_grouped = df.groupby(reg_col)[rev_col].sum().sort_values(ascending=False)
        if not reg_grouped.empty:
            top_reg = reg_grouped.index[0]
            top_reg_val = float(reg_grouped.iloc[0])
            top_reg_pct = (top_reg_val / reg_grouped.sum()) * 100
            insights["drivers"].append(
                f"🌍 **Geographic Leader**: **{top_reg}** leads regional performance with **${top_reg_val:,.2f}** ({top_reg_pct:.1f}% regional share)."
            )

    # 4. Date & Trend Velocity
    if date_col and rev_col and date_col in df.columns and rev_col in df.columns:
        try:
            d_df = df.copy()
            d_df[date_col] = pd.to_datetime(d_df[date_col], errors="coerce")
            d_df = d_df.dropna(subset=[date_col]).sort_values(date_col)
            if len(d_df) >= 4:
                monthly = d_df.set_index(date_col).resample("ME")[rev_col].sum()
                if len(monthly) >= 2:
                    first_val = float(monthly.iloc[0])
                    last_val = float(monthly.iloc[-1])
                    peak_month = monthly.idxmax().strftime("%B %Y")
                    peak_val = float(monthly.max())
                    growth_rate = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0.0

                    insights["drivers"].append(
                        f"📅 **Trend Velocity**: Highest monthly performance occurred in **{peak_month}** (${peak_val:,.2f}). Trajectory shows **{growth_rate:+.1f}%** period-over-period change."
                    )
        except Exception:
            pass

    # 5. Default Recommendations if empty
    if not insights["recommendations"]:
        insights["recommendations"].append("Maintain real-time tracking of top-performing items to capitalize on high-demand inventory cycles.")
        insights["recommendations"].append("Leverage customer and product segmentation to tailor targeted marketing campaigns.")
        insights["recommendations"].append("Implement automated data hygiene checks to prevent missing values and duplicate records.")

    return insights

