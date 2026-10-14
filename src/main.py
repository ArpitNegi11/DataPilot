import sys
import os
import argparse
import yaml
import pandas as pd

# Ensure stdout/stderr use utf-8 encoding for Windows terminals
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on sys.path for direct CLI execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import cleaning, validation, anomalies, visualize, reporting, predictive, io_utils, analytics, recommendations


def load_config(config_path: str):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def extract_validation_rules(config: dict) -> list:
    """Extract validation rules from various config schema formats."""
    raw_rules = config.get("validation_rules") or config.get("validation") or config.get("validations") or []
    if isinstance(raw_rules, list):
        return raw_rules
    elif isinstance(raw_rules, dict):
        rules_list = []
        for col_name, rule_body in raw_rules.items():
            if isinstance(rule_body, dict):
                rules_list.append({"column": col_name, **rule_body})
        return rules_list
    return []


def run_pipeline(input_file: str, output_dir: str, config_path: str):
    # Ensure outputs are always routed into the outputs directory, never project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not output_dir or output_dir.strip() in [".", "", "./", ".\\"] or os.path.abspath(output_dir) == project_root:
        output_dir = os.path.join(project_root, "outputs")

    os.makedirs(output_dir, exist_ok=True)

    # Load config
    config = load_config(config_path)

    # Load data via smart ingestion
    print(f"[INFO] Loading file: {input_file}")
    df = io_utils.load_file(input_file, auto_clean_header=True)
    print(f"[OK] Loaded {df.shape[0]} rows and {df.shape[1]} columns")

    # --- Cleaning ---
    df_clean = cleaning.clean_data(df, config)

    # Save cleaned file matching input format
    ext = os.path.splitext(input_file)[1].lower()
    cleaned_ext = ".csv" if ext == ".csv" else ".xlsx"
    cleaned_file = os.path.join(output_dir, f"cleaned{cleaned_ext}")
    io_utils.save_file(df_clean, cleaned_file)

    # --- Validation ---
    validation_rules = extract_validation_rules(config)
    validation_issues = validation.validate_data(df_clean, validation_rules)

    # --- Anomalies ---
    anomalies_found = anomalies.detect_anomalies(df_clean)

    # --- Visualizations ---
    figures = visualize.generate_visuals(df_clean, output_dir)

    # --- Predictive Insights ---
    insights = predictive.run_predictive_models(df_clean)

    # --- Summary for Report ---
    roles = analytics.infer_business_columns(df_clean)
    kpis = analytics.compute_kpis(df_clean, roles)

    summary = {
        "Original Rows": df.shape[0],
        "Original Columns": df.shape[1],
        "Cleaned Rows": df_clean.shape[0],
        "Columns After Cleaning": df_clean.shape[1],
        "Validation Issues Found": sum(len(v) for v in validation_issues.values()),
        "Columns With Issues": len(validation_issues),
        "Anomalies Detected": len(anomalies_found),
        "Output File": os.path.basename(cleaned_file),
    }

    if kpis.get("total_revenue") is not None:
        summary["Total Revenue"] = f"${kpis['total_revenue']:,.2f}"
    if kpis.get("total_profit") is not None:
        summary["Total Profit"] = f"${kpis['total_profit']:,.2f}"
    if kpis.get("profit_margin_pct") is not None:
        summary["Profit Margin"] = f"{kpis['profit_margin_pct']:.1f}%"
    if kpis.get("average_order_value") is not None:
        summary["Average Order Value"] = f"${kpis['average_order_value']:,.2f}"

    # --- Strategic Recommendations ---
    rec_engine = recommendations.RecommendationEngine(df_clean, roles)
    business_recs = rec_engine.generate_recommendations()

    # --- Report ---
    report_file = os.path.join(output_dir, "report.html")
    reporting.generate_report(
        validation=validation_issues,
        anomalies=anomalies_found,
        figures=figures,
        summary=summary,
        insights=insights,
        output_path=report_file,
        recommendations=business_recs
    )



def main():
    parser = argparse.ArgumentParser(description="DataPilot - AI Data Analyst")
    parser.add_argument("--input", required=True, help="Path to input Excel file")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")

    args = parser.parse_args()
    run_pipeline(args.input, args.outdir, args.config)


if __name__ == "__main__":
    main()
