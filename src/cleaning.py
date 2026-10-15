import pandas as pd
import numpy as np
from src import io_utils


def load_data(filepath: str) -> pd.DataFrame:
    """Load file into a pandas DataFrame using smart ingestion."""
    return io_utils.load_file(filepath)


def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Clean the dataset based on config rules."""
    print("[CLEANING] Starting data cleaning...")
    df = df.copy()

    cfg = {**config, **config["cleaning"]} if isinstance(config, dict) and "cleaning" in config and isinstance(config["cleaning"], dict) else (config or {})

    # Smart numeric & object inference
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

    # Drop duplicates
    if cfg.get("drop_duplicates", True):
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        print(f"[DUPLICATES] Removed {before - after} duplicate rows")

    # Handle missing values: row dropping threshold
    fill_cfg = cfg.get("fill_missing", {})
    strategy = fill_cfg.get("strategy", cfg.get("impute_strategy", "median"))

    if "dropna_threshold" in cfg:
        max_missing_ratio = float(cfg["dropna_threshold"])
        row_threshold = max(1, int(np.ceil((1.0 - max_missing_ratio) * df.shape[1])))
    else:
        threshold = float(fill_cfg.get("threshold", 0.3))
        row_threshold = max(1, int(threshold * df.shape[1]))

    before = len(df)
    df = df.dropna(thresh=row_threshold)
    after = len(df)
    print(f"[THRESHOLD] Dropped {before - after} rows with too many missing values")

    # Explicit column fillna mapping from config
    fillna_rules = cfg.get("fillna", {})
    if isinstance(fillna_rules, dict):
        for col, fill_val in fillna_rules.items():
            if col in df.columns and df[col].isna().any():
                df[col] = df[col].fillna(fill_val)
                print(f"[IMPUTE] Filled NaNs in '{col}' with config value '{fill_val}'")

    # Fill remaining missing values
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                if strategy == "mean":
                    value = df[col].mean()
                else:  # default to median
                    value = df[col].median()
                df[col] = df[col].fillna(value)
                print(f"[IMPUTE] Filled NaNs in numeric column '{col}' with {strategy} ({value:.2f})")
            else:
                df[col] = df[col].fillna("Unknown")
                print(f"[IMPUTE] Filled NaNs in text column '{col}' with 'Unknown'")

    # Drop any duplicate rows resulting from imputation
    if cfg.get("drop_duplicates", True):
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if before > after:
            print(f"[DUPLICATES] Removed {before - after} duplicate rows after imputation")

    # Standardize text columns safely
    text_std = cfg.get("text_standardization", "title")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        try:
            if text_std == "lower":
                df[col] = df[col].apply(lambda x: str(x).lower() if isinstance(x, str) else x)
            elif text_std == "upper":
                df[col] = df[col].apply(lambda x: str(x).upper() if isinstance(x, str) else x)
            elif text_std == "title":
                df[col] = df[col].apply(lambda x: str(x).title() if isinstance(x, str) else x)
        except Exception as e:
            print(f"[TEXT_WARN] Could not standardize text in '{col}': {e}")

    # Fix date columns
    default_date_fmt = cfg.get("date_format", "%Y-%m-%d")
    raw_date_cols = cfg.get("date_columns", [])
    configured_date_names = set()
    date_col_formats = {}

    for c in raw_date_cols:
        if isinstance(c, str):
            configured_date_names.add(c)
        elif isinstance(c, dict) and "name" in c:
            configured_date_names.add(c["name"])
            if "format" in c:
                date_col_formats[c["name"]] = c["format"]

    for col in df.columns:
        is_date_col = "date" in str(col).lower() or "time" in str(col).lower() or col in configured_date_names
        if is_date_col:
            specified_fmt = date_col_formats.get(col)
            try:
                parsed_dates = None
                if specified_fmt:
                    parsed_dates = pd.to_datetime(df[col], errors="coerce", format=specified_fmt)

                # Fallback to flexible parsing if no format or strict parsing produced only NaT
                if parsed_dates is None or (parsed_dates.isna().all() and df[col].notna().any()):
                    parsed_dates = pd.to_datetime(df[col], errors="coerce")

                mask = parsed_dates.notna()
                if mask.any():
                    df.loc[mask, col] = parsed_dates[mask].dt.strftime(default_date_fmt)
                    print(f"[DATE] Standardized '{col}' to format {default_date_fmt}")
            except Exception as e:
                print(f"[DATE_WARN] Could not standardize '{col}': {e}")

    print("[SUCCESS] Cleaning complete")
    return df
