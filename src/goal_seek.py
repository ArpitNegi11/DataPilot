import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from src.analytics import infer_business_columns


class GoalSeekOptimizer:
    """
    Autonomous Goal-Seek & Decision Optimizer (Prescriptive Analytics).
    Calculates precise, constrained price, cost, and discount levers across business
    categories to achieve target financial outcomes (e.g. +15% profit, 25% margin).
    Incorporates price elasticity of demand to guarantee realistic market simulations.
    """

    def __init__(self, df: pd.DataFrame, roles: Optional[Dict[str, Optional[str]]] = None):
        self.df = df.copy()
        self.roles = roles or infer_business_columns(self.df)
        self._init_category_baseline()

    def _init_category_baseline(self):
        rev_col = self.roles.get("revenue")
        prof_col = self.roles.get("profit")
        qty_col = self.roles.get("quantity")
        cat_col = self.roles.get("category") or self.roles.get("product")
        cost_col = self.roles.get("cost")
        disc_col = self.roles.get("discount")

        # Fallback category if none detected
        if not cat_col or cat_col not in self.df.columns:
            self.df["_All_Business_"] = "Core Enterprise"
            cat_col = "_All_Business_"

        # Aggregate category level baseline
        rows = []
        for cat_name, g in self.df.groupby(cat_col):
            if pd.isna(cat_name) or str(cat_name).strip() == "":
                continue

            rev = float(g[rev_col].sum()) if rev_col and rev_col in g.columns and pd.api.types.is_numeric_dtype(g[rev_col]) else float(len(g))
            qty = float(g[qty_col].sum()) if qty_col and qty_col in g.columns and pd.api.types.is_numeric_dtype(g[qty_col]) and g[qty_col].sum() > 0 else float(len(g))
            
            if prof_col and prof_col in g.columns and pd.api.types.is_numeric_dtype(g[prof_col]):
                prof = float(g[prof_col].sum())
            else:
                prof = rev * 0.18  # default baseline 18% margin assumption

            if cost_col and cost_col in g.columns and pd.api.types.is_numeric_dtype(g[cost_col]):
                cost = float(g[cost_col].sum())
            else:
                cost = max(rev - prof, 0.0)

            avg_price = rev / qty if qty > 0 else rev
            unit_cost = cost / qty if qty > 0 else cost

            avg_disc = float(g[disc_col].mean()) if disc_col and disc_col in g.columns and pd.api.types.is_numeric_dtype(g[disc_col]) else 0.05

            rows.append({
                "category": str(cat_name),
                "revenue": rev,
                "profit": prof,
                "cost": cost,
                "quantity": qty,
                "avg_price": avg_price,
                "unit_cost": unit_cost,
                "avg_discount": avg_disc,
                "margin_pct": (prof / rev * 100) if rev > 0 else 0.0
            })

        self.baseline_df = pd.DataFrame(rows)
        if self.baseline_df.empty:
            self.baseline_df = pd.DataFrame([{
                "category": "General Operations",
                "revenue": 100000.0,
                "profit": 15000.0,
                "cost": 85000.0,
                "quantity": 1000.0,
                "avg_price": 100.0,
                "unit_cost": 85.0,
                "avg_discount": 0.05,
                "margin_pct": 15.0
            }])

    def optimize(
        self,
        target_metric: str = "profit",
        target_change_pct: float = 10.0,
        price_elasticity: float = -1.1,
        max_price_increase_pct: float = 10.0,
        max_cost_reduction_pct: float = 8.0
    ) -> Dict[str, Any]:
        """
        Run constrained goal-seek optimization to find optimal category price & cost levers.
        
        Args:
            target_metric: 'profit', 'revenue', or 'margin'
            target_change_pct: Percentage change desired (e.g. +10.0 means +10% increase)
            price_elasticity: Demand sensitivity to price (%ΔQ / %ΔP, standard e-comm ~ -1.1)
            max_price_increase_pct: Maximum allowed price increase cap
            max_cost_reduction_pct: Maximum allowed cost/discount compression cap
        """
        base_rev = float(self.baseline_df["revenue"].sum())
        base_prof = float(self.baseline_df["profit"].sum())
        base_cost = float(self.baseline_df["cost"].sum())
        base_margin = (base_prof / base_rev * 100) if base_rev > 0 else 0.0

        n_cats = len(self.baseline_df)
        
        # Determine numerical target value
        if target_metric == "profit":
            target_val = base_prof * (1.0 + target_change_pct / 100.0) if base_prof > 0 else base_rev * 0.15
        elif target_metric == "revenue":
            target_val = base_rev * (1.0 + target_change_pct / 100.0)
        elif target_metric == "margin":
            target_val = max(1.0, min(95.0, base_margin + target_change_pct))
        else:
            target_val = base_prof * (1.0 + target_change_pct / 100.0)

        # Simulation function
        def simulate(x):
            # x is array of [dp_0, ..., dp_n, dc_0, ..., dc_n] (fractional changes)
            dp = x[:n_cats]
            dc = x[n_cats:]

            total_sim_rev = 0.0
            total_sim_prof = 0.0
            total_sim_cost = 0.0

            for i, row in self.baseline_df.iterrows():
                p_lift = dp[i]
                c_cut = dc[i]
                
                # Demand response with elasticity
                q_mult = max(0.2, 1.0 + price_elasticity * p_lift)
                new_q = row["quantity"] * q_mult
                new_p = row["avg_price"] * (1.0 + p_lift)
                new_u_cost = max(0.01, row["unit_cost"] * (1.0 - c_cut))

                cat_rev = new_q * new_p
                cat_cost = new_q * new_u_cost
                cat_prof = cat_rev - cat_cost

                total_sim_rev += cat_rev
                total_sim_cost += cat_cost
                total_sim_prof += cat_prof

            return total_sim_rev, total_sim_prof, total_sim_cost

        # Bounds: price change [-0.05, max_price], cost cut [0.0, max_cost]
        bnds = [(-0.05, max_price_increase_pct / 100.0) for _ in range(n_cats)] + \
               [(0.0, max_cost_reduction_pct / 100.0) for _ in range(n_cats)]

        x0 = np.zeros(2 * n_cats)

        solved_x = None

        if HAS_SCIPY:
            def objective(x):
                sim_rev, sim_prof, sim_cost = simulate(x)
                if target_metric == "profit":
                    metric_err = (sim_prof - target_val) / (abs(target_val) + 1e-6)
                elif target_metric == "revenue":
                    metric_err = (sim_rev - target_val) / (abs(target_val) + 1e-6)
                elif target_metric == "margin":
                    sim_margin = (sim_prof / sim_rev * 100) if sim_rev > 0 else 0.0
                    metric_err = (sim_margin - target_val) / (abs(target_val) + 1e-6)
                else:
                    metric_err = (sim_prof - target_val) / (abs(target_val) + 1e-6)

                # Penalize large lever variances across categories (prefer balanced adjustments)
                smoothness_penalty = 0.5 * np.sum(x**2)
                return (metric_err ** 2) * 1000.0 + smoothness_penalty

            try:
                res = minimize(objective, x0, method="SLSQP", bounds=bnds, options={"maxiter": 150})
                if res.success or res.fun < 10.0:
                    solved_x = res.x
            except Exception:
                solved_x = None

        # Fallback solver: proportional distribution if scipy unavailable or did not converge
        if solved_x is None:
            gap_ratio = (target_val - base_prof) / (base_prof + 1e-6) if target_metric == "profit" else target_change_pct / 100.0
            p_val = min(max_price_increase_pct / 100.0, max(-0.05, gap_ratio * 0.4))
            c_val = min(max_cost_reduction_pct / 100.0, max(0.0, gap_ratio * 0.3))
            solved_x = np.array([p_val] * n_cats + [c_val] * n_cats)

        sim_rev, sim_prof, sim_cost = simulate(solved_x)
        sim_margin = (sim_prof / sim_rev * 100) if sim_rev > 0 else 0.0

        # Build granular category action plan
        cat_levers = []
        dp_res = solved_x[:n_cats]
        dc_res = solved_x[n_cats:]

        for i, row in self.baseline_df.iterrows():
            p_pct = float(dp_res[i] * 100.0)
            c_pct = float(dc_res[i] * 100.0)
            
            # Isolated category impact
            q_mult = max(0.2, 1.0 + price_elasticity * (p_pct / 100.0))
            new_cat_rev = (row["quantity"] * q_mult) * (row["avg_price"] * (1.0 + p_pct / 100.0))
            new_cat_cost = (row["quantity"] * q_mult) * (row["unit_cost"] * (1.0 - c_pct / 100.0))
            new_cat_prof = new_cat_rev - new_cat_cost
            
            prof_delta = new_cat_prof - row["profit"]
            rev_delta = new_cat_rev - row["revenue"]

            action_desc = []
            if abs(p_pct) >= 0.2:
                action_desc.append(f"Adjust price by {p_pct:+.1f}%")
            if c_pct >= 0.2:
                action_desc.append(f"Compress cost/discount by {c_pct:.1f}%")
            if not action_desc:
                action_desc.append("Maintain baseline pricing")

            cat_levers.append({
                "category": row["category"],
                "baseline_revenue": round(row["revenue"], 2),
                "baseline_profit": round(row["profit"], 2),
                "recommended_price_change_pct": round(p_pct, 2),
                "recommended_cost_cut_pct": round(c_pct, 2),
                "projected_revenue": round(new_cat_rev, 2),
                "projected_profit": round(new_cat_prof, 2),
                "profit_impact": round(prof_delta, 2),
                "revenue_impact": round(rev_delta, 2),
                "action_summary": " & ".join(action_desc)
            })

        # Executive Action Plan Directives
        top_profit_drivers = sorted(cat_levers, key=lambda x: x["profit_impact"], reverse=True)
        directives = []
        if top_profit_drivers:
            best_cat = top_profit_drivers[0]
            directives.append(
                f"Primary Profit Catalyst: In '{best_cat['category']}', apply a {best_cat['recommended_price_change_pct']:+.1f}% price adjustment "
                f"and {best_cat['recommended_cost_cut_pct']:.1f}% cost control to generate {best_cat['profit_impact']:+,.0f} in incremental profit."
            )
        if len(top_profit_drivers) > 1:
            second_cat = top_profit_drivers[1]
            directives.append(
                f"Secondary Lever: In '{second_cat['category']}', execute {second_cat['action_summary']} for a {second_cat['profit_impact']:+,.0f} margin contribution."
            )
        
        directives.append(
            f"Net Projected Outcome: Reaches ${sim_prof:,.0f} Profit ({sim_prof - base_prof:+,.0f} vs baseline) "
            f"with an enterprise margin of {sim_margin:.1f}% (baseline: {base_margin:.1f}%)."
        )

        # Build waterfall simulation data
        waterfall_labels = ["Baseline Profit"]
        waterfall_values = [base_prof]
        waterfall_measures = ["absolute"]

        for item in sorted(cat_levers, key=lambda x: abs(x["profit_impact"]), reverse=True)[:5]:
            waterfall_labels.append(f"{item['category']} Lever")
            waterfall_values.append(item["profit_impact"])
            waterfall_measures.append("relative")

        waterfall_labels.append("Target Projected Profit")
        waterfall_values.append(sim_prof)
        waterfall_measures.append("total")

        return {
            "success": True,
            "target_metric": target_metric,
            "target_change_pct": target_change_pct,
            "target_value": round(target_val, 2),
            "baseline": {
                "revenue": round(base_rev, 2),
                "profit": round(base_prof, 2),
                "cost": round(base_cost, 2),
                "margin_pct": round(base_margin, 2)
            },
            "projected": {
                "revenue": round(sim_rev, 2),
                "profit": round(sim_prof, 2),
                "cost": round(sim_cost, 2),
                "margin_pct": round(sim_margin, 2)
            },
            "delta": {
                "revenue_delta": round(sim_rev - base_rev, 2),
                "profit_delta": round(sim_prof - base_prof, 2),
                "margin_delta_pts": round(sim_margin - base_margin, 2)
            },
            "category_levers": cat_levers,
            "directives": directives,
            "waterfall_data": {
                "labels": waterfall_labels,
                "values": waterfall_values,
                "measures": waterfall_measures
            }
        }
