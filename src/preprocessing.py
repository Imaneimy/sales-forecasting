"""
Load and prepare monthly sales data for time series modeling.
"""

import pandas as pd
import numpy as np


def load_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["month"])
    df = df.sort_values("month").reset_index(drop=True)
    df = df.set_index("month")
    df.index = df.index.to_period("M")
    return df


def check_stationarity(series: pd.Series) -> dict:
    """Augmented Dickey-Fuller test — ADF stat < critical value means stationary."""
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series.dropna())
    return {
        "adf_statistic": round(result[0], 4),
        "p_value": round(result[1], 4),
        "is_stationary": result[1] < 0.05,
        "critical_values": {k: round(v, 4) for k, v in result[4].items()},
    }


def decompose(series: pd.Series, period: int = 12):
    from statsmodels.tsa.seasonal import seasonal_decompose
    return seasonal_decompose(series, model="additive", period=period)


def train_test_split_ts(df: pd.DataFrame, test_months: int = 6):
    train = df.iloc[:-test_months]
    test = df.iloc[-test_months:]
    return train, test


def compute_baseline_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    errors = actual.values - predicted.values
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs(errors / actual.values)) * 100
    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_pct": round(mape, 2),
    }
