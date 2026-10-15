import re
import pandas as pd

def validate_data(df: pd.DataFrame, rules: list) -> dict:
    """
    Validate data against rules defined in config.
    Returns a dictionary of issues found.
    """
    print("[VALIDATION] Starting data validation...")
    issues = {}

    if not isinstance(rules, list):
        return issues

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        col = rule.get("column")
        if not col or col not in df.columns:
            continue

        col_issues = []

        # Check min
        if "min" in rule:
            invalid_rows = df[df[col] < rule["min"]]
            if not invalid_rows.empty:
                col_issues.append(f"{len(invalid_rows)} values below {rule['min']}")

        # Check max
        if "max" in rule:
            invalid_rows = df[df[col] > rule["max"]]
            if not invalid_rows.empty:
                col_issues.append(f"{len(invalid_rows)} values above {rule['max']}")

        # Check regex
        if "regex" in rule:
            pattern = re.compile(rule["regex"])
            invalid_rows = df[~df[col].astype(str).str.match(pattern)]
            if not invalid_rows.empty:
                col_issues.append(f"{len(invalid_rows)} values do not match regex {rule['regex']}")

        # Check unique
        if rule.get("unique", False):
            duplicates = df[df.duplicated(col, keep=False)]
            if not duplicates.empty:
                col_issues.append(f"{len(duplicates)} duplicate values found")

        # Check allowed values
        if "allowed_values" in rule and isinstance(rule["allowed_values"], (list, set, tuple)):
            allowed = set(str(v).strip().lower() for v in rule["allowed_values"])
            # Validate non-null values
            non_null_series = df[col].dropna().astype(str).str.strip().str.lower()
            invalid_mask = ~non_null_series.isin(allowed)
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                col_issues.append(f"{invalid_count} values not in allowed set {rule['allowed_values']}")

        # Check required / not null
        if rule.get("required") or rule.get("not_null"):
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                col_issues.append(f"{null_count} null/missing values found")

        if col_issues:
            issues[col] = col_issues
            print(f"[VALIDATION_FAIL] Issues in '{col}': {col_issues}")

    if not issues:
        print("[VALIDATION_OK] No validation issues found")

    return issues
