import os
import pytest
from src.main import run_pipeline


def test_cli_pipeline_csv_and_excel(tmp_path):
    outdir_csv = str(tmp_path / "out_csv")
    outdir_xlsx = str(tmp_path / "out_xlsx")

    # Run on sample CSV
    run_pipeline(
        input_file="sample_data/DataPilot_Sales_Demo.csv",
        output_dir=outdir_csv,
        config_path="config/config.yaml"
    )

    assert os.path.exists(os.path.join(outdir_csv, "cleaned.csv"))
    assert os.path.exists(os.path.join(outdir_csv, "report.html"))

    # Run on sample Excel
    run_pipeline(
        input_file="sample_data/sample.xlsx",
        output_dir=outdir_xlsx,
        config_path="config/config.yaml"
    )

    assert os.path.exists(os.path.join(outdir_xlsx, "cleaned.xlsx"))
    assert os.path.exists(os.path.join(outdir_xlsx, "report.html"))
