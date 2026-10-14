import pandas as pd
import pytest
from src import validation


def test_validation_rules():
    df = pd.DataFrame({
        "age": [15, 25, 120],
        "code": ["P101", "INVALID", "P102"],
        "user_id": [1, 2, 2],
        "plan": ["basic", "pro", "enterprise"]
    })

    rules = [
        {"column": "age", "min": 18, "max": 100},
        {"column": "code", "regex": r"^P\d{3}$"},
        {"column": "user_id", "unique": True},
        {"column": "plan", "allowed_values": ["basic", "pro"]}
    ]

    issues = validation.validate_data(df, rules)

    assert "age" in issues
    assert any("below 18" in msg for msg in issues["age"])
    assert any("above 100" in msg for msg in issues["age"])

    assert "code" in issues
    assert any("regex" in msg for msg in issues["code"])

    assert "user_id" in issues
    assert any("duplicate" in msg for msg in issues["user_id"])

    assert "plan" in issues
    assert any("not in allowed set" in msg for msg in issues["plan"])
