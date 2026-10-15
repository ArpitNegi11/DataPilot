import pandas as pd
import numpy as np
import pytest
from src import anomalies


def test_detect_anomalies():
    # Normal distribution with extreme outlier
    np.random.seed(42)
    normal_values = np.random.normal(loc=50, scale=5, size=100).tolist()
    normal_values.append(500.0)  # Extreme outlier

    df = pd.DataFrame({
        "metric": normal_values,
        "constant_col": [10.0] * 101
    })

    anoms = anomalies.detect_anomalies(df, z_thresh=3.0)

    assert "metric" in anoms
    assert 500.0 in anoms["metric"]
    # Constant column should not crash or trigger anomalies
    assert "constant_col" not in anoms
