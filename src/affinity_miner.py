import pandas as pd
import numpy as np
from itertools import combinations
from typing import Dict, Any, List, Optional
from src.analytics import infer_business_columns


class MarketBasketAffinityMiner:
    """
    Product Cannibalization & Market Basket Affinity Miner.
    Mines co-purchase cross-sell associations (Support, Confidence, Lift)
    and detects internal product cannibalization where promotional discounts
    on one product erode higher-margin sales of substitute items.
    """

    def __init__(self, df: pd.DataFrame, roles: Optional[Dict[str, Optional[str]]] = None):
        self.df = df.copy()
        self.roles = roles or infer_business_columns(self.df)

    def mine_affinity_and_cannibalization(
        self,
        min_support: float = 0.01,
        min_lift: float = 1.0,
        max_items_to_analyze: int = 40
    ) -> Dict[str, Any]:
        """
        Execute full affinity association mining and product cannibalization scan.
        """
        prod_col = self.roles.get("product")
        cat_col = self.roles.get("category")
        rev_col = self.roles.get("revenue")
        prof_col = self.roles.get("profit")
        disc_col = self.roles.get("discount")
        date_col = self.roles.get("date")
        order_col = self.roles.get("order_id")
        cust_col = self.roles.get("customer_id")

        if not prod_col or prod_col not in self.df.columns:
            # Fallback to category or first text column
            text_cols = self.df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
            prod_col = text_cols[0] if text_cols else self.df.columns[0]

        # -------------------------------------------------------------
        # 1. Determine Basket ID
        # -------------------------------------------------------------
        df_work = self.df.dropna(subset=[prod_col]).copy()
        df_work[prod_col] = df_work[prod_col].astype(str)

        if order_col and order_col in df_work.columns:
            basket_col = order_col
        elif cust_col and date_col and cust_col in df_work.columns and date_col in df_work.columns:
            df_work["_basket_id_"] = df_work[cust_col].astype(str) + "_" + df_work[date_col].astype(str)
            basket_col = "_basket_id_"
        elif date_col and date_col in df_work.columns:
            df_work["_basket_id_"] = df_work[date_col].astype(str)
            basket_col = "_basket_id_"
        else:
            # Synthetic 4-item batches
            df_work["_basket_id_"] = np.arange(len(df_work)) // 4
            basket_col = "_basket_id_"

        # Keep top most frequent products to maintain efficiency and relevance
        top_prods = df_work[prod_col].value_counts().head(max_items_to_analyze).index.tolist()
        df_filtered = df_work[df_work[prod_col].isin(top_prods)]

        # Group items by basket
        baskets = df_filtered.groupby(basket_col)[prod_col].apply(lambda s: sorted(set(s))).tolist()
        total_baskets = len(baskets)

        if total_baskets == 0:
            return self._empty_result()

        # Item support counts
        item_counts = {}
        for b in baskets:
            for item in b:
                item_counts[item] = item_counts.get(item, 0) + 1

        # Pair co-occurrence counts
        pair_counts = {}
        for b in baskets:
            if len(b) >= 2:
                for a, b_item in combinations(b, 2):
                    pair = tuple(sorted([a, b_item]))
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # Compute Support, Confidence, Lift
        association_rules = []
        matrix_items = sorted(top_prods[:15])  # Top items for matrix visualization
        affinity_matrix = pd.DataFrame(1.0, index=matrix_items, columns=matrix_items)

        for (a, b), co_cnt in pair_counts.items():
            supp_ab = co_cnt / total_baskets
            supp_a = item_counts[a] / total_baskets
            supp_b = item_counts[b] / total_baskets
            
            lift = supp_ab / (supp_a * supp_b + 1e-9)

            if a in matrix_items and b in matrix_items:
                affinity_matrix.loc[a, b] = round(lift, 2)
                affinity_matrix.loc[b, a] = round(lift, 2)

            if supp_ab >= min_support and lift >= min_lift:
                conf_a_to_b = supp_ab / (supp_a + 1e-9)
                conf_b_to_a = supp_ab / (supp_b + 1e-9)

                association_rules.append({
                    "item_a": a,
                    "item_b": b,
                    "support_pct": round(supp_ab * 100, 2),
                    "confidence_a_to_b_pct": round(conf_a_to_b * 100, 2),
                    "confidence_b_to_a_pct": round(conf_b_to_a * 100, 2),
                    "lift": round(lift, 2),
                    "co_purchases": co_cnt,
                    "recommended_action": f"Bundle '{a}' with '{b}' to lift cross-sell conversion by {lift:.1f}x"
                })

        association_rules.sort(key=lambda x: x["lift"], reverse=True)

        # -------------------------------------------------------------
        # 2. Product Cannibalization Scan
        # -------------------------------------------------------------
        # We examine whether high sales of Product A negatively impact Product B
        cannibalization_alerts = []

        # Build periodic or categorical time-series / segment matrix
        pivot_dim = date_col if (date_col and date_col in self.df.columns and self.df[date_col].nunique() >= 4) else (cat_col if (cat_col and cat_col in self.df.columns and self.df[cat_col].nunique() >= 4) else basket_col)
        
        qty_col = self.roles.get("quantity")
        metric_col = rev_col if rev_col and rev_col in self.df.columns else (qty_col if qty_col and qty_col in self.df.columns else None)

        if metric_col and pivot_dim:
            try:
                sales_pivot = self.df[self.df[prod_col].isin(top_prods[:25])].pivot_table(
                    index=pivot_dim,
                    columns=prod_col,
                    values=metric_col,
                    aggfunc="sum",
                    fill_value=0.0
                )

                corr_matrix = sales_pivot.corr()

                for i, p1 in enumerate(sales_pivot.columns):
                    for j, p2 in enumerate(sales_pivot.columns):
                        if i < j:
                            r_val = float(corr_matrix.loc[p1, p2])
                            # Negative correlation implies potential cannibalization / substitution
                            if r_val <= -0.25 and not np.isnan(r_val):
                                # Determine which product has lower margin or higher discount
                                p1_rows = self.df[self.df[prod_col] == p1]
                                p2_rows = self.df[self.df[prod_col] == p2]

                                p1_rev = float(p1_rows[rev_col].sum()) if rev_col and rev_col in p1_rows else 1.0
                                p2_rev = float(p2_rows[rev_col].sum()) if rev_col and rev_col in p2_rows else 1.0

                                p1_prof = float(p1_rows[prof_col].sum()) if prof_col and prof_col in p1_rows else (p1_rev * 0.15)
                                p2_prof = float(p2_rows[prof_col].sum()) if prof_col and prof_col in p2_rows else (p2_rev * 0.25)

                                p1_margin = (p1_prof / p1_rev * 100) if p1_rev > 0 else 15.0
                                p2_margin = (p2_prof / p2_rev * 100) if p2_rev > 0 else 25.0

                                # Cannibalizer is the lower-margin or more aggressive item stealing from higher-margin item
                                if p1_margin <= p2_margin:
                                    cannibalizer, victim = p1, p2
                                    c_m, v_m = p1_margin, p2_margin
                                    c_rev, v_rev = p1_rev, p2_rev
                                else:
                                    cannibalizer, victim = p2, p1
                                    c_m, v_m = p2_margin, p1_margin
                                    c_rev, v_rev = p2_rev, p1_rev

                                severity = min(100.0, round(abs(r_val) * (1.0 + (v_m - c_m) / 100.0) * 80.0, 1))
                                est_displacement = round(abs(r_val) * min(c_rev, v_rev) * 0.12, 2)

                                cannibalization_alerts.append({
                                    "cannibalizer_product": cannibalizer,
                                    "victim_product": victim,
                                    "correlation": round(r_val, 2),
                                    "severity_score": severity,
                                    "cannibalizer_margin_pct": round(c_m, 1),
                                    "victim_margin_pct": round(v_m, 1),
                                    "estimated_margin_displacement": est_displacement,
                                    "countermeasure": f"Re-evaluate promotion timing on '{cannibalizer}' to protect {victim_margin_pct:.1f}% margin sales on '{victim}'" if 'victim_margin_pct' in locals() else f"Re-evaluate promotions on '{cannibalizer}'"
                                })
            except Exception:
                pass

        cannibalization_alerts.sort(key=lambda x: x["severity_score"], reverse=True)

        return {
            "total_baskets": total_baskets,
            "unique_items_analyzed": len(top_prods),
            "association_rules": association_rules[:20],
            "cannibalization_alerts": cannibalization_alerts[:15],
            "affinity_matrix": affinity_matrix.to_dict(),
            "summary": {
                "top_affinity_pair": f"{association_rules[0]['item_a']} + {association_rules[0]['item_b']} (Lift: {association_rules[0]['lift']}x)" if association_rules else "None detected",
                "top_cannibalization_risk": f"{cannibalization_alerts[0]['cannibalizer_product']} cannibalizing {cannibalization_alerts[0]['victim_product']} (Severity: {cannibalization_alerts[0]['severity_score']}%)" if cannibalization_alerts else "Zero critical cannibalization detected"
            }
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "total_baskets": 0,
            "unique_items_analyzed": 0,
            "association_rules": [],
            "cannibalization_alerts": [],
            "affinity_matrix": {},
            "summary": {
                "top_affinity_pair": "None",
                "top_cannibalization_risk": "None"
            }
        }
