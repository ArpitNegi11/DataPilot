import pandas as pd
import numpy as np
import pytest
from src import cleaning


def test_clean_data_duplicates_and_imputation():
    df = pd.DataFrame({
        "id": [1, 2, 2, 3],
        "sales": [10.0, np.nan, 20.0, 30.0],
        "category": ["electronics", "clothing", "clothing", np.nan],
    })

    config = {
        "drop_duplicates": True,
        "text_standardization": "title"
    }

    df_cleaned = cleaning.clean_data(df, config)

    # Duplicate row dropped
    assert len(df_cleaned) == 3
    # Category imputed with 'Unknown' and capitalized to 'Unknown'
    assert df_cleaned["category"].isna().sum() == 0
    assert "Unknown" in df_cleaned["category"].values
    # Sales NaN imputed with median (10 and 30 median is 20)
    assert df_cleaned["sales"].isna().sum() == 0
    # Text standardized to title case
    assert "Electronics" in df_cleaned["category"].values


def test_clean_data_custom_fillna():
    df = pd.DataFrame({
        "status": ["active", np.nan, "canceled"],
        "cost": [10, 20, np.nan]
    })

    config = {
        "fillna": {
            "status": "Not Canceled"
        }
    }

    df_cleaned = cleaning.clean_data(df, config)
    assert df_cleaned["status"].iloc[1] == "Not Canceled"


def test_clean_data_date_standardization():
    df = pd.DataFrame({
        "order_date": ["2023/01/15", "2023-05-20", "12-10-2022"],
        "price": [100, 200, 300]
    })

    config = {
        "date_format": "%Y-%m-%d"
    }

    df_cleaned = cleaning.clean_data(df, config)
    assert df_cleaned["order_date"].iloc[0] == "2023-01-15"
    assert df_cleaned["order_date"].iloc[1] == "2023-05-20"
