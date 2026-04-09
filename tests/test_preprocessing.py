import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocessing import load_series, check_stationarity, train_test_split_ts, compute_baseline_metrics

DATA = Path(__file__).parent.parent / "data" / "monthly_sales.csv"


@pytest.fixture(scope="module")
def df():
    return load_series(DATA)


# TC-TS-001
def test_load_returns_dataframe(df):
    assert isinstance(df, pd.DataFrame)


# TC-TS-002
def test_index_is_period(df):
    assert str(type(df.index)) == "<class 'pandas.core.indexes.period.PeriodIndex'>"


# TC-TS-003
def test_no_nulls(df):
    assert df.isnull().sum().sum() == 0


# TC-TS-004
def test_revenue_positive(df):
    assert (df["revenue"] > 0).all()


# TC-TS-005
def test_sorted_ascending(df):
    assert df.index.is_monotonic_increasing


# TC-TS-006
def test_stationarity_returns_dict(df):
    result = check_stationarity(df["revenue"])
    for k in ["adf_statistic", "p_value", "is_stationary"]:
        assert k in result


# TC-TS-007
def test_stationarity_p_value_range(df):
    result = check_stationarity(df["revenue"])
    assert 0 <= result["p_value"] <= 1


# TC-TS-008
def test_train_test_split_sizes(df):
    train, test = train_test_split_ts(df, test_months=6)
    assert len(train) + len(test) == len(df)
    assert len(test) == 6


# TC-TS-009
def test_train_before_test(df):
    train, test = train_test_split_ts(df, test_months=6)
    assert train.index[-1] < test.index[0]


# TC-TS-010
def test_baseline_metrics_keys(df):
    import numpy as np
    actual = df["revenue"].iloc[:6]
    predicted = pd.Series(actual.values * 1.05)
    metrics = compute_baseline_metrics(actual, predicted)
    for k in ["mae", "rmse", "mape_pct"]:
        assert k in metrics


# TC-TS-011
def test_mae_non_negative(df):
    import numpy as np
    actual = df["revenue"].iloc[:6]
    predicted = pd.Series(actual.values * 1.05)
    metrics = compute_baseline_metrics(actual, predicted)
    assert metrics["mae"] >= 0


# TC-TS-012
def test_rmse_geq_mae(df):
    import numpy as np
    actual = df["revenue"].iloc[:6]
    predicted = pd.Series(actual.values * 1.05)
    metrics = compute_baseline_metrics(actual, predicted)
    assert metrics["rmse"] >= metrics["mae"]
