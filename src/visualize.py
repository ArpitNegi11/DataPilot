import os
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def _fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64


def _safe_filename(name: str) -> str:
    """Sanitize a string for use in file paths."""
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(name))


# ---------------------------------------------------------------------------
# Shared Plotly layout helper
# ---------------------------------------------------------------------------

_PLOTLY_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    font=dict(family="Inter, sans-serif", size=13),
    margin=dict(l=60, r=30, t=50, b=60),
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
)


def _apply_layout(fig, *, height: int = 400, title: str = ""):
    """Apply consistent layout defaults to a Plotly figure."""
    fig.update_layout(
        **_PLOTLY_LAYOUT_DEFAULTS,
        height=height,
        title=dict(text=title, x=0.02, font=dict(size=16)),
    )
    return fig


# ---------------------------------------------------------------------------
# Static charts for HTML report (matplotlib / seaborn)
# ---------------------------------------------------------------------------

def generate_visuals(df: pd.DataFrame, output_dir: str = "outputs") -> dict:
    """
    Generate static matplotlib/seaborn charts for report generation.
    """
    print("[VISUALS] Generating visualizations for report...")
    figs = {}
    
    # Ensure outputs are routed into an output folder and never root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not output_dir or output_dir.strip() in [".", "", "./", ".\\"] or os.path.abspath(output_dir) == project_root:
        output_dir = os.path.join(project_root, "outputs")

    os.makedirs(output_dir, exist_ok=True)

    # Apply dark style
    plt.style.use('dark_background')
    sns.set_style("darkgrid")

    # 1. Missing values heatmap
    if df.isnull().sum().sum() > 0:
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
            ax.set_title("Missing Values Heatmap", color='white')
            path = os.path.join(output_dir, "missing_values.png")
            fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
            figs["missing_values"] = _fig_to_base64(fig)
        except Exception as e:
            print(f"[VISUAL_WARN] Missing values plot error: {e}")

    # 2. Correlation heatmap
    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
            ax.set_title("Correlation Heatmap", color='white')
            path = os.path.join(output_dir, "correlation_matrix.png")
            fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
            figs["correlation_matrix"] = _fig_to_base64(fig)
        except Exception as e:
            print(f"[VISUAL_WARN] Correlation heatmap error: {e}")

    # 3. Distribution plots for numeric columns (up to 3)
    if not numeric_df.empty:
        for col in numeric_df.columns[:3]:
            try:
                valid_data = numeric_df[col].dropna()
                if not valid_data.empty:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.histplot(valid_data, kde=True, bins=25, color="#1f77b4", ax=ax)
                    ax.set_title(f"Distribution of {col}", color='white')
                    path = os.path.join(output_dir, f"dist_{_safe_filename(col)}.png")
                    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
                    figs[f"dist_{col}"] = _fig_to_base64(fig)
            except Exception as e:
                print(f"[VISUAL_WARN] Distribution error for {col}: {e}")

    # 4. Bar plot for first categorical column frequencies
    cat_df = df.select_dtypes(include=["object", "string", "category"])
    if not cat_df.empty:
        col = cat_df.columns[0]
        try:
            val_counts = df[col].value_counts().head(10)
            if not val_counts.empty:
                fig, ax = plt.subplots(figsize=(7, 4))
                val_counts.plot(kind="bar", color="#ff7f0e", ax=ax)
                ax.set_title(f"Top 10 Frequency: {col}", color='white')
                plt.xticks(rotation=45, ha="right")
                path = os.path.join(output_dir, f"freq_{_safe_filename(col)}.png")
                fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
                figs[f"freq_{col}"] = _fig_to_base64(fig)
        except Exception as e:
            print(f"[VISUAL_WARN] Frequency plot error for {col}: {e}")

    # 5. Regression model plot
    reg_x, reg_y = None, None
    if "Experience" in df.columns and "Salary" in df.columns:
        reg_x, reg_y = "Experience", "Salary"
    elif "Quantity" in df.columns and "Sales" in df.columns:
        reg_x, reg_y = "Quantity", "Sales"
    elif "Units" in df.columns and "Total" in df.columns:
        reg_x, reg_y = "Units", "Total"
    elif "Sales" in df.columns and "Profit" in df.columns:
        reg_x, reg_y = "Sales", "Profit"
    elif len(numeric_df.columns) >= 2:
        try:
            corr_mat = numeric_df.corr().abs()
            corr_vals = corr_mat.to_numpy(copy=True)
            np.fill_diagonal(corr_vals, 0)
            corr_mat = pd.DataFrame(corr_vals, index=corr_mat.index, columns=corr_mat.columns)
            if not corr_mat.empty and corr_mat.max().max() > 0.05:
                col_max = corr_mat.stack().idxmax()
                reg_x, reg_y = col_max[0], col_max[1]
        except Exception as e:
            print(f"[VISUAL_WARN] Correlation pair selection error: {e}")

    if reg_x and reg_y:
        try:
            reg_valid = df[[reg_x, reg_y]].dropna()
            if len(reg_valid) >= 3:
                X_v = reg_valid[[reg_x]].values
                y_v = reg_valid[reg_y].values
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score
                lr_model = LinearRegression()
                lr_model.fit(X_v, y_v)
                y_fit = lr_model.predict(X_v)
                r2_val = r2_score(y_v, y_fit)

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.scatter(X_v, y_v, color="#4CC9F0", alpha=0.7, label="Observed Data")
                sort_idx = np.argsort(X_v.flatten())
                ax.plot(
                    X_v.flatten()[sort_idx],
                    y_fit[sort_idx],
                    color="#F72585",
                    linewidth=2.5,
                    label=f"Fit (R2={r2_val:.3f})",
                )
                ax.set_title(f"Linear Regression: {reg_y} vs {reg_x}", color="white")
                ax.set_xlabel(reg_x, color="white")
                ax.set_ylabel(reg_y, color="white")
                ax.legend(facecolor="#1e1e1e", edgecolor="#333333", labelcolor="white")
                plt.tight_layout()
                path = os.path.join(output_dir, "regression_model.png")
                fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
                figs["regression_model"] = _fig_to_base64(fig)
        except Exception as e:
            print(f"[VISUAL_WARN] Static regression plot error: {e}")

    print(f"[VISUALS_OK] Generated {len(figs)} figures successfully")
    return figs


# ---------------------------------------------------------------------------
# Interactive Plotly charts for the Streamlit dashboard
# ---------------------------------------------------------------------------

def generate_interactive_visuals(df: pd.DataFrame) -> dict:
    """
    Generate interactive Plotly charts for the Streamlit dashboard.
    Returns a dict keyed by chart name -> Plotly figure.
    """
    plotly_figs = {}

    # ------------------------------------------------------------------
    # 1. Missing Values Bar Chart (if any)
    # ------------------------------------------------------------------
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    if not missing_cols.empty:
        miss_df = pd.DataFrame({
            "Column": missing_cols.index,
            "Missing_Count": missing_cols.values,
            "Missing_Pct": (missing_cols.values / len(df) * 100).round(1)
        }).sort_values(by="Missing_Count", ascending=True)

        fig_miss = px.bar(
            miss_df,
            x="Missing_Pct",
            y="Column",
            orientation="h",
            text="Missing_Pct",
            color="Missing_Pct",
            color_continuous_scale="Reds"
        )
        fig_miss.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        _apply_layout(fig_miss, height=max(280, len(miss_df) * 40), title="Missing Values by Column (%)")
        fig_miss.update_layout(coloraxis_showscale=False)
        plotly_figs["missing_values"] = fig_miss

    # ------------------------------------------------------------------
    # Column role detection (shared across charts)
    # ------------------------------------------------------------------
    cols_lower = {str(c).lower().replace(" ", "_"): c for c in df.columns}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Find metric (Revenue / Total / Sales / Profit / Units)
    metric_col = None
    for key in ["sales", "revenue", "total", "total_(usd)", "subscription_cost",
                 "profit", "units", "quantity", "cost_price_total_(usd)"]:
        if key in cols_lower and cols_lower[key] in numeric_cols:
            metric_col = cols_lower[key]
            break
    if not metric_col and numeric_cols:
        metric_col = numeric_cols[0]

    # Find product / item column
    prod_col = None
    for key in ["product_name", "product", "item", "item_name", "product_id", "task_name"]:
        if key in cols_lower:
            prod_col = cols_lower[key]
            break

    # Find category / region column
    cat_col = None
    for key in ["category", "segment", "region", "rep",
                 "subscription_interval", "department", "assigned_to"]:
        if key in cols_lower:
            cat_col = cols_lower[key]
            break
    if not cat_col and cat_cols:
        cat_col = cat_cols[0]

    # Find date column
    date_col = None
    for c in df.columns:
        if "date" in str(c).lower() or "time" in str(c).lower() or pd.api.types.is_datetime64_any_dtype(df[c]):
            date_col = c
            break

    # ------------------------------------------------------------------
    # 2. Top Products / Items Chart
    # ------------------------------------------------------------------
    if prod_col and metric_col:
        try:
            top_items = (
                df.groupby(prod_col, as_index=False)[metric_col]
                .sum()
                .sort_values(by=metric_col, ascending=False)
                .head(10)
            )
            fig_top = px.bar(
                top_items,
                x=prod_col,
                y=metric_col,
                text=metric_col,
                color=metric_col,
                color_continuous_scale="Blues"
            )
            fig_top.update_traces(texttemplate="%{text:,.2s}", textposition="outside")
            _apply_layout(fig_top, height=420, title=f"Top 10 {prod_col} by {metric_col}")
            fig_top.update_layout(
                coloraxis_showscale=False,
                xaxis_tickangle=-40,
            )
            plotly_figs["top_products"] = fig_top
        except Exception as e:
            print(f"[VISUAL_WARN] Top products chart error: {e}")

    # ------------------------------------------------------------------
    # 3. Category / Segment Breakdown (Donut)
    # ------------------------------------------------------------------
    if cat_col and metric_col:
        try:
            cat_data = (
                df.groupby(cat_col, as_index=False)[metric_col]
                .sum()
                .sort_values(by=metric_col, ascending=False)
            )
            fig_cat = px.pie(
                cat_data,
                names=cat_col,
                values=metric_col,
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_cat.update_traces(textposition="inside", textinfo="percent+label")
            _apply_layout(fig_cat, height=420, title=f"{metric_col} by {cat_col}")
            plotly_figs["category_breakdown"] = fig_cat
        except Exception as e:
            print(f"[VISUAL_WARN] Category breakdown chart error: {e}")

    # ------------------------------------------------------------------
    # 4. Time Trends (if date column exists)
    # ------------------------------------------------------------------
    if date_col and metric_col:
        try:
            temp_df = df.copy()
            temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors="coerce")
            temp_df = temp_df.dropna(subset=[date_col])
            if len(temp_df) >= 2:
                # Sort by date and aggregate monthly
                temp_df = temp_df.sort_values(date_col)
                trend_df = (
                    temp_df
                    .set_index(date_col)
                    .resample("ME")[metric_col]
                    .sum()
                    .reset_index()
                )
                if len(trend_df) >= 2:
                    trend_df["Period"] = trend_df[date_col].dt.strftime("%Y-%m")
                    fig_trend = px.line(
                        trend_df,
                        x="Period",
                        y=metric_col,
                        markers=True,
                        line_shape="spline"
                    )
                    fig_trend.update_traces(
                        line_color="#4CC9F0",
                        line_width=3,
                        marker=dict(size=8)
                    )
                    _apply_layout(fig_trend, height=400, title=f"Monthly Trend: {metric_col}")
                    plotly_figs["monthly_trend"] = fig_trend
        except Exception as e:
            print(f"[VISUAL_WARN] Monthly trend chart error: {e}")

    # ------------------------------------------------------------------
    # 5. Correlation Heatmap
    # ------------------------------------------------------------------
    if len(numeric_cols) > 1:
        try:
            corr_matrix = df[numeric_cols].corr().round(2)
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
            )
            _apply_layout(fig_corr, height=max(380, len(numeric_cols) * 50), title="Correlation Matrix")
            plotly_figs["correlation_matrix"] = fig_corr
        except Exception as e:
            print(f"[VISUAL_WARN] Correlation matrix error: {e}")

    # ------------------------------------------------------------------
    # 6. Feature Distributions (up to 3 numeric columns)
    # ------------------------------------------------------------------
    dist_cols = numeric_cols[:3]
    for i, dist_col in enumerate(dist_cols):
        try:
            valid = df[dist_col].dropna()
            if len(valid) >= 2:
                colors = ["#4895EF", "#F72585", "#4CC9F0"]
                fig_dist = px.histogram(
                    df.dropna(subset=[dist_col]),
                    x=dist_col,
                    marginal="box",
                    color_discrete_sequence=[colors[i % len(colors)]]
                )
                _apply_layout(fig_dist, height=380, title=f"Distribution: {dist_col}")
                key = f"distribution_{i}" if i > 0 else "distribution"
                plotly_figs[key] = fig_dist
        except Exception as e:
            print(f"[VISUAL_WARN] Distribution chart error for {dist_col}: {e}")

    # ------------------------------------------------------------------
    # 7. Linear Regression Model (Scatter + Fit Line)
    # ------------------------------------------------------------------
    reg_x, reg_y = None, None
    if "Experience" in df.columns and "Salary" in df.columns:
        reg_x, reg_y = "Experience", "Salary"
    elif "Quantity" in df.columns and "Sales" in df.columns:
        reg_x, reg_y = "Quantity", "Sales"
    elif "Units" in df.columns and "Total" in df.columns:
        reg_x, reg_y = "Units", "Total"
    elif "Sales" in df.columns and "Profit" in df.columns:
        reg_x, reg_y = "Sales", "Profit"
    elif len(numeric_cols) >= 2:
        try:
            corr_mat = df[numeric_cols].corr().abs()
            corr_vals = corr_mat.to_numpy(copy=True)
            np.fill_diagonal(corr_vals, 0)
            corr_mat = pd.DataFrame(corr_vals, index=corr_mat.index, columns=corr_mat.columns)
            if not corr_mat.empty and corr_mat.max().max() > 0.05:
                col_max = corr_mat.stack().idxmax()
                reg_x, reg_y = col_max[0], col_max[1]
        except Exception as e:
            print(f"[VISUAL_WARN] Correlation pair selection error: {e}")

    if reg_x and reg_y:
        try:
            reg_df = df[[reg_x, reg_y]].dropna()
            if len(reg_df) >= 3:
                X_arr = reg_df[[reg_x]].values
                y_arr = reg_df[reg_y].values
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score
                lr = LinearRegression()
                lr.fit(X_arr, y_arr)
                y_pred = lr.predict(X_arr)
                r2 = float(r2_score(y_arr, y_pred))
                coef = lr.coef_[0]
                intercept = lr.intercept_

                fig_reg = px.scatter(
                    reg_df,
                    x=reg_x,
                    y=reg_y,
                    opacity=0.7,
                    color_discrete_sequence=["#4CC9F0"]
                )
                fig_reg.update_traces(name="Observed Data", showlegend=True)
                sort_idx = np.argsort(X_arr.flatten())
                fig_reg.add_trace(
                    go.Scatter(
                        x=X_arr.flatten()[sort_idx],
                        y=y_pred[sort_idx],
                        mode="lines",
                        name=f"Fit: {reg_y} = {coef:.2f}*{reg_x} + {intercept:.2f}",
                        line=dict(color="#F72585", width=3)
                    )
                )
                _apply_layout(
                    fig_reg,
                    height=420,
                    title=f"Linear Regression: {reg_y} vs {reg_x} (R2 = {r2:.3f})"
                )
                fig_reg.update_layout(
                    legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02)
                )
                plotly_figs["regression_model"] = fig_reg
        except Exception as e:
            print(f"[VISUAL_WARN] Interactive regression chart error: {e}")

    return plotly_figs
