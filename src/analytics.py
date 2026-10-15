import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List


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

    text_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if not roles["product"] and text_cols:
        roles["product"] = text_cols[0]
    if not roles["category"] and len(text_cols) > 1:
        roles["category"] = text_cols[1]

    return roles


def compute_kpis(df: pd.DataFrame, roles: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Any]:
    """
    Compute deterministic core business KPIs from dataset.
    Guarantees zero hallucination of aggregates.
    """
    if roles is None:
        roles = infer_business_columns(df)

    kpis = {
        "total_records": int(len(df)),
        "total_revenue": None,
        "total_profit": None,
        "profit_margin_pct": None,
        "total_quantity": None,
        "average_order_value": None,
        "unique_products": None,
        "unique_categories": None,
        "unique_regions": None,
    }

    rev_col = roles.get("revenue")
    prof_col = roles.get("profit")
    qty_col = roles.get("quantity")
    prod_col = roles.get("product")
    cat_col = roles.get("category")
    reg_col = roles.get("region")

    if rev_col and rev_col in df.columns and pd.api.types.is_numeric_dtype(df[rev_col]):
        rev_sum = float(df[rev_col].sum())
        kpis["total_revenue"] = round(rev_sum, 2)
        if len(df) > 0:
            kpis["average_order_value"] = round(rev_sum / len(df), 2)

    if prof_col and prof_col in df.columns and pd.api.types.is_numeric_dtype(df[prof_col]):
        prof_sum = float(df[prof_col].sum())
        kpis["total_profit"] = round(prof_sum, 2)
        if kpis["total_revenue"] and kpis["total_revenue"] > 0:
            kpis["profit_margin_pct"] = round((prof_sum / kpis["total_revenue"]) * 100, 2)

    if qty_col and qty_col in df.columns and pd.api.types.is_numeric_dtype(df[qty_col]):
        kpis["total_quantity"] = float(df[qty_col].sum())

    if prod_col and prod_col in df.columns:
        kpis["unique_products"] = int(df[prod_col].nunique())

    if cat_col and cat_col in df.columns:
        kpis["unique_categories"] = int(df[cat_col].nunique())

    if reg_col and reg_col in df.columns:
        kpis["unique_regions"] = int(df[reg_col].nunique())

    return kpis


def compute_breakdowns(df: pd.DataFrame, roles: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Any]:
    """
    Compute categorical, regional, and product dimensional aggregations.
    """
    if roles is None:
        roles = infer_business_columns(df)

    rev_col = roles.get("revenue")
    prof_col = roles.get("profit")
    cat_col = roles.get("category")
    reg_col = roles.get("region")
    prod_col = roles.get("product")

    breakdowns = {}

    if cat_col and cat_col in df.columns and rev_col and rev_col in df.columns:
        cat_agg = df.groupby(cat_col)[rev_col].agg(["sum", "mean", "count"]).reset_index()
        cat_agg.columns = [cat_col, "Total_Revenue", "Average_Revenue", "Transactions"]
        cat_agg = cat_agg.sort_values(by="Total_Revenue", ascending=False)
        breakdowns["category"] = cat_agg

    if reg_col and reg_col in df.columns and rev_col and rev_col in df.columns:
        reg_agg = df.groupby(reg_col)[rev_col].agg(["sum", "count"]).reset_index()
        reg_agg.columns = [reg_col, "Total_Revenue", "Transactions"]
        reg_agg = reg_agg.sort_values(by="Total_Revenue", ascending=False)
        breakdowns["region"] = reg_agg

    if prod_col and prod_col in df.columns and rev_col and rev_col in df.columns:
        prod_agg = df.groupby(prod_col)[rev_col].sum().reset_index()
        prod_agg.columns = [prod_col, "Total_Revenue"]
        prod_agg = prod_agg.sort_values(by="Total_Revenue", ascending=False).head(10)
        breakdowns["top_products"] = prod_agg

    return breakdowns


def compute_time_trends(df: pd.DataFrame, roles: Optional[Dict[str, Optional[str]]] = None) -> Optional[pd.DataFrame]:
    """
    Compute monthly revenue aggregations and Month-over-Month (MoM) growth rates.
    """
    if roles is None:
        roles = infer_business_columns(df)

    date_col = roles.get("date")
    rev_col = roles.get("revenue")

    if not date_col or not rev_col or date_col not in df.columns or rev_col not in df.columns:
        return None

    try:
        temp = df.copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.dropna(subset=[date_col, rev_col])

        if temp.empty:
            return None

        monthly = temp.set_index(date_col).resample("ME")[rev_col].sum().reset_index()
        monthly["Period"] = monthly[date_col].dt.strftime("%Y-%m")
        monthly["MoM_Growth_Pct"] = monthly[rev_col].pct_change() * 100
        monthly["MoM_Growth_Pct"] = monthly["MoM_Growth_Pct"].round(2)
        return monthly
    except Exception as e:
        print(f"[ANALYTICS_WARN] Time trends computation error: {e}")
        return None


def compute_data_quality_score(
    df_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    validation_issues: Dict[str, List[str]],
    anomalies: Dict[str, List[Any]]
) -> Dict[str, Any]:
    """
    Compute a deterministic data quality health score from 0 to 100%.
    """
    score = 100.0

    total_cells = df_raw.shape[0] * df_raw.shape[1] if df_raw.size > 0 else 1
    raw_nulls = df_raw.isna().sum().sum()
    null_ratio = raw_nulls / total_cells
    # Deduct up to 30 points for missing values
    score -= min(30.0, null_ratio * 100)

    # Duplicates penalty (up to 20 points)
    raw_dups = df_raw.duplicated().sum()
    dup_ratio = raw_dups / max(1, len(df_raw))
    score -= min(20.0, dup_ratio * 100)

    # Validation issues penalty (up to 30 points)
    total_issues = sum(len(v) for v in validation_issues.values()) if validation_issues else 0
    score -= min(30.0, total_issues * 5.0)

    # Anomalies penalty (up to 20 points)
    total_anomalies = sum(len(v) for v in anomalies.values()) if anomalies else 0
    score -= min(20.0, total_anomalies * 2.0)

    final_score = max(0.0, min(100.0, round(score, 1)))

    if final_score >= 85:
        grade = "Excellent"
    elif final_score >= 70:
        grade = "Good"
    elif final_score >= 50:
        grade = "Fair"
    else:
        grade = "Poor"

    return {
        "score": final_score,
        "grade": grade,
        "raw_nulls": int(raw_nulls),
        "raw_duplicates": int(raw_dups),
        "validation_issues_count": int(total_issues),
        "anomalies_count": int(total_anomalies)
    }


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
    prod_col = roles.get("product")
    cat_col = roles.get("category")
    date_col = roles.get("date")
    reg_col = roles.get("region")

    total_records = len(df)
    insights["kpis"]["total_records"] = total_records

    # 1. Revenue & Profit Margin Analysis
    if rev_col and rev_col in df.columns and pd.api.types.is_numeric_dtype(df[rev_col]):
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

        if prof_col and prof_col in df.columns and pd.api.types.is_numeric_dtype(df[prof_col]):
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
