import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from preprocessing import load_series, train_test_split_ts, compute_baseline_metrics
from model import fit_arima, forecast, residual_stats

DATA = Path(__file__).parent.parent / "data" / "monthly_sales.csv"


@pytest.fixture(scope="module")
def fitted():
    df = load_series(DATA)
    train, test = train_test_split_ts(df, test_months=6)
    model = fit_arima(train["revenue"], order=(1, 1, 1))
    fc = forecast(model, steps=6)
    return {"model": model, "fc": fc, "train": train, "test": test}


# TC-MOD-TS-001
def test_forecast_length(fitted):
    assert len(fitted["fc"]) == 6


# TC-MOD-TS-002
def test_forecast_positive(fitted):
    assert (fitted["fc"] > 0).all()


# TC-MOD-TS-003
def test_mape_below_threshold(fitted):
    metrics = compute_baseline_metrics(fitted["test"]["revenue"], fitted["fc"])
    assert metrics["mape_pct"] < 20


# TC-MOD-TS-004
def test_residuals_dict_keys(fitted):
    stats = residual_stats(fitted["model"])
    for k in ["mean", "std", "max_abs"]:
        assert k in stats


# TC-MOD-TS-005
def test_residuals_mean_near_zero(fitted):
    stats = residual_stats(fitted["model"])
    assert abs(stats["mean"]) < 5000


# TC-MOD-TS-006
def test_aic_is_finite(fitted):
    import math
    assert math.isfinite(fitted["model"].aic)


# TC-MOD-TS-007
def test_different_orders_produce_different_aic():
    df = load_series(DATA)
    train, _ = train_test_split_ts(df, test_months=6)
    m1 = fit_arima(train["revenue"], order=(1, 1, 1))
    m2 = fit_arima(train["revenue"], order=(2, 1, 2))
    assert m1.aic != m2.aic


# TC-MOD-TS-008
def test_rmse_reasonable(fitted):
    metrics = compute_baseline_metrics(fitted["test"]["revenue"], fitted["fc"])
    avg_revenue = fitted["test"]["revenue"].mean()
    assert metrics["rmse"] < avg_revenue * 0.3
