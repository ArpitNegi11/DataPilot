import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from src.analytics import infer_business_columns


class RootCauseDecomposer:
    """
    Diagnostic Root-Cause Decomposition Engine ('Why Did This Happen?').
    Performs deterministic variance decomposition across Volume, Price, Cost, and Mix effects,
    attributing performance swings to concrete drivers and generating executive narratives.
    """

    def __init__(self, df: pd.DataFrame, roles: Optional[Dict[str, Optional[str]]] = None):
        self.df = df.copy()
        self.roles = roles or infer_business_columns(self.df)
        self._prepare_data()

    def _prepare_data(self):
        date_col = self.roles.get("date")
        if date_col and date_col in self.df.columns:
            try:
                self.df["_parsed_date_"] = pd.to_datetime(self.df[date_col], errors="coerce")
            except Exception:
                self.df["_parsed_date_"] = None
        else:
            self.df["_parsed_date_"] = None

    def get_available_periods(self) -> List[str]:
        """Return available temporal periods if date column exists, or split by half."""
        if self.df["_parsed_date_"].notna().sum() > 4:
            valid_dates = self.df.dropna(subset=["_parsed_date_"])
            periods = valid_dates["_parsed_date_"].dt.to_period("M").astype(str).unique().tolist()
            if len(periods) >= 2:
                return sorted(periods)
        return ["Baseline Period (First 50%)", "Current Period (Second 50%)"]

    def decompose(
        self,
        period_base: Optional[str] = None,
        period_curr: Optional[str] = None,
        metric: str = "profit"
    ) -> Dict[str, Any]:
        """
        Execute deterministic variance decomposition.
        
        Args:
            period_base: Identifier for base/prior period (e.g. '2023-01' or None for auto)
            period_curr: Identifier for current/recent period (e.g. '2023-02' or None for auto)
            metric: 'profit' or 'revenue'
        """
        rev_col = self.roles.get("revenue")
        prof_col = self.roles.get("profit")
        qty_col = self.roles.get("quantity")
        cat_col = self.roles.get("category") or self.roles.get("product") or "Category"
        cost_col = self.roles.get("cost")

        if cat_col not in self.df.columns:
            self.df["_Segment_"] = "All Operations"
            cat_col = "_Segment_"

        # Partition dataframe into Base Period and Current Period
        available_periods = self.get_available_periods()
        
        if len(available_periods) >= 2 and self.df["_parsed_date_"].notna().sum() > 4:
            if not period_base or not period_curr:
                period_base = available_periods[-2]
                period_curr = available_periods[-1]

            valid_dates = self.df.dropna(subset=["_parsed_date_"])
            p_series = valid_dates["_parsed_date_"].dt.to_period("M").astype(str)
            df_base = valid_dates[p_series == period_base].copy()
            df_curr = valid_dates[p_series == period_curr].copy()
            label_base = f"Period {period_base}"
            label_curr = f"Period {period_curr}"
        else:
            # Fallback to chronological or index split
            n = len(self.df)
            mid = n // 2
            df_base = self.df.iloc[:mid].copy()
            df_curr = self.df.iloc[mid:].copy()
            label_base = "Base Period"
            label_curr = "Current Period"

        # Helper to compute category aggregates
        def agg_period(sub_df):
            records = {}
            for c_name, g in sub_df.groupby(cat_col):
                if pd.isna(c_name) or str(c_name).strip() == "":
                    continue
                r = float(g[rev_col].sum()) if rev_col and rev_col in g.columns and pd.api.types.is_numeric_dtype(g[rev_col]) else float(len(g))
                q = float(g[qty_col].sum()) if qty_col and qty_col in g.columns and pd.api.types.is_numeric_dtype(g[qty_col]) and g[qty_col].sum() > 0 else float(len(g))
                
                if prof_col and prof_col in g.columns and pd.api.types.is_numeric_dtype(g[prof_col]):
                    p = float(g[prof_col].sum())
                else:
                    p = r * 0.18
                
                if cost_col and cost_col in g.columns and pd.api.types.is_numeric_dtype(g[cost_col]):
                    c = float(g[cost_col].sum())
                else:
                    c = max(r - p, 0.0)

                unit_p = r / q if q > 0 else r
                unit_c = c / q if q > 0 else c
                unit_m = (p / q) if q > 0 else p

                records[str(c_name)] = {
                    "revenue": r,
                    "profit": p,
                    "cost": c,
                    "quantity": q,
                    "unit_price": unit_p,
                    "unit_cost": unit_c,
                    "unit_margin": unit_m
                }
            return records

        base_cats = agg_period(df_base)
        curr_cats = agg_period(df_curr)

        all_cats = sorted(set(list(base_cats.keys()) + list(curr_cats.keys())))

        total_base_rev = sum(d["revenue"] for d in base_cats.values())
        total_curr_rev = sum(d["revenue"] for d in curr_cats.values())
        total_base_prof = sum(d["profit"] for d in base_cats.values())
        total_curr_prof = sum(d["profit"] for d in curr_cats.values())

        val_base = total_base_prof if metric == "profit" else total_base_rev
        val_curr = total_curr_prof if metric == "profit" else total_curr_rev
        total_delta = val_curr - val_base

        # Calculate 4 orthogonal variance components
        volume_effect = 0.0
        price_effect = 0.0
        cost_effect = 0.0
        category_attributions = []

        for c in all_cats:
            b = base_cats.get(c, {"revenue": 0.0, "profit": 0.0, "cost": 0.0, "quantity": 0.0, "unit_price": 0.0, "unit_cost": 0.0, "unit_margin": 0.0})
            cur = curr_cats.get(c, {"revenue": 0.0, "profit": 0.0, "cost": 0.0, "quantity": 0.0, "unit_price": 0.0, "unit_cost": 0.0, "unit_margin": 0.0})

            # Base & Current values
            q0, q1 = b["quantity"], cur["quantity"]
            p0, p1 = b["unit_price"], cur["unit_price"]
            c0, c1 = b["unit_cost"], cur["unit_cost"]
            m0 = b["unit_margin"]

            # Volume Effect: (Q1 - Q0) * P0 (or Margin0)
            if metric == "profit":
                vol_c = (q1 - q0) * m0
                prc_c = (p1 - p0) * q1
                cst_c = (c0 - c1) * q1
            else:
                vol_c = (q1 - q0) * p0
                prc_c = (p1 - p0) * q1
                cst_c = 0.0

            cat_total_delta = (cur["profit"] - b["profit"]) if metric == "profit" else (cur["revenue"] - b["revenue"])

            volume_effect += vol_c
            price_effect += prc_c
            cost_effect += cst_c

            category_attributions.append({
                "category": c,
                "base_value": round(b["profit"] if metric == "profit" else b["revenue"], 2),
                "current_value": round(cur["profit"] if metric == "profit" else cur["revenue"], 2),
                "net_delta": round(cat_total_delta, 2),
                "volume_effect": round(vol_c, 2),
                "price_effect": round(prc_c, 2),
                "cost_effect": round(cst_c, 2),
                "primary_driver": "Volume" if abs(vol_c) > max(abs(prc_c), abs(cst_c)) else ("Price" if abs(prc_c) > abs(cst_c) else "Cost")
            })

        # Mix Effect reconciles any multi-product share shift and cross-elasticity
        explained_so_far = volume_effect + price_effect + cost_effect
        mix_effect = total_delta - explained_so_far

        # Identify Key Positive Drivers and Negative Drags
        sorted_by_impact = sorted(category_attributions, key=lambda x: x["net_delta"], reverse=True)
        top_positive_drivers = [x for x in sorted_by_impact if x["net_delta"] > 0][:3]
        top_negative_drags = [x for x in sorted(category_attributions, key=lambda x: x["net_delta"]) if x["net_delta"] < 0][:3]

        # Synthesize Executive Root-Cause Narrative
        pct_change = (total_delta / abs(val_base) * 100) if abs(val_base) > 0 else 0.0
        direction = "increased" if total_delta >= 0 else "decreased"
        
        effects_summary = [
            ("Volume Effect", volume_effect, f"{'expanded unit demand' if volume_effect >= 0 else 'contracted unit sales'}"),
            ("Price Realization", price_effect, f"{'favorable pricing realizations' if price_effect >= 0 else 'pricing erosion / discounting'}"),
            ("Cost/Margin Compression", cost_effect, f"{'cost optimizations' if cost_effect >= 0 else 'inflationary cost drag'}"),
            ("Portfolio Mix Shift", mix_effect, f"{'shift toward higher margin products' if mix_effect >= 0 else 'shift toward lower margin products'}")
        ]
        
        # Sort effects by absolute magnitude to find primary root cause
        primary_effect_name, primary_effect_val, primary_effect_desc = sorted(effects_summary, key=lambda x: abs(x[1]), reverse=True)[0]

        driver_sentences = []
        if top_negative_drags:
            worst = top_negative_drags[0]
            driver_sentences.append(f"Largest drag was observed in '{worst['category']}' ({worst['net_delta']:+,.0f}), primarily due to {worst['primary_driver'].lower()} changes.")
        if top_positive_drivers:
            best = top_positive_drivers[0]
            driver_sentences.append(f"Strongest offsetting growth stemmed from '{best['category']}' ({best['net_delta']:+,.0f}).")

        executive_narrative = (
            f"{metric.capitalize()} {direction} by ${abs(total_delta):,.0f} ({pct_change:+.1f}%) from {label_base} (${val_base:,.0f}) to {label_curr} (${val_curr:,.0f}). "
            f"The dominant underlying root cause was {primary_effect_name} (${primary_effect_val:+,.0f}), reflecting {primary_effect_desc}. "
            + " ".join(driver_sentences)
        )

        # Plotly Waterfall components
        waterfall_labels = [label_base, "Volume Effect", "Price Effect"]
        waterfall_values = [val_base, volume_effect, price_effect]
        waterfall_measures = ["absolute", "relative", "relative"]

        if metric == "profit":
            waterfall_labels.append("Cost Effect")
            waterfall_values.append(cost_effect)
            waterfall_measures.append("relative")

        waterfall_labels.extend(["Portfolio Mix", label_curr])
        waterfall_values.extend([mix_effect, val_curr])
        waterfall_measures.extend(["relative", "total"])

        return {
            "metric": metric,
            "period_base": label_base,
            "period_curr": label_curr,
            "baseline_value": round(val_base, 2),
            "current_value": round(val_curr, 2),
            "total_delta": round(total_delta, 2),
            "pct_change": round(pct_change, 2),
            "decomposition": {
                "volume_effect": round(volume_effect, 2),
                "price_effect": round(price_effect, 2),
                "cost_effect": round(cost_effect, 2),
                "mix_effect": round(mix_effect, 2)
            },
            "primary_root_cause": primary_effect_name,
            "executive_narrative": executive_narrative,
            "top_positive_drivers": top_positive_drivers,
            "top_negative_drags": top_negative_drags,
            "category_attributions": category_attributions,
            "waterfall_data": {
                "labels": waterfall_labels,
                "values": [round(v, 2) for v in waterfall_values],
                "measures": waterfall_measures
            }
        }
