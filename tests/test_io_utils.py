import os
import tempfile
import pandas as pd
import pytest
from src import io_utils


def test_load_csv_and_excel(tmp_path):
    df_orig = pd.DataFrame({
        "Product": ["Widget A", "Widget B"],
        "Sales": [100.5, 250.0]
    })

    csv_file = str(tmp_path / "test.csv")
    xlsx_file = str(tmp_path / "test.xlsx")

    df_orig.to_csv(csv_file, index=False)
    df_orig.to_excel(xlsx_file, index=False)

    df_csv = io_utils.load_file(csv_file)
    assert df_csv.shape == (2, 2)
    assert list(df_csv.columns) == ["Product", "Sales"]

    df_xlsx = io_utils.load_file(xlsx_file)
    assert df_xlsx.shape == (2, 2)
    assert list(df_xlsx.columns) == ["Product", "Sales"]


def test_detect_and_clean_header_with_banner():
    # Simulate a raw spreadsheet with metadata title rows and empty columns
    raw_data = {
        "Unnamed: 0": [None, "COMPANY REPORT - Q1", None, "OrderID", 101, 102],
        "Unnamed: 1": [None, None, None, "Amount", 50, 75],
        "Unnamed: 2": [None, None, None, None, None, None]  # completely empty column
    }
    df_messy = pd.DataFrame(raw_data)
    cleaned = io_utils.detect_and_clean_header(df_messy)

    assert "OrderID" in cleaned.columns
    assert "Amount" in cleaned.columns
    assert cleaned.shape[0] == 2
    assert "Unnamed: 2" not in cleaned.columns


def test_save_file(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    out_csv = str(tmp_path / "output.csv")
    out_xlsx = str(tmp_path / "output.xlsx")

    io_utils.save_file(df, out_csv)
    assert os.path.exists(out_csv)

    io_utils.save_file(df, out_xlsx)
    assert os.path.exists(out_xlsx)
