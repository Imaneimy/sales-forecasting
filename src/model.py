"""
ARIMA model for monthly revenue forecasting.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def fit_arima(series: pd.Series, order: tuple = (1, 1, 1)):
    model = ARIMA(series, order=order)
    return model.fit()


def forecast(fitted_model, steps: int) -> pd.Series:
    pred = fitted_model.forecast(steps=steps)
    return pred


def auto_select_order(series: pd.Series, max_p: int = 3, max_q: int = 3) -> tuple:
    """Select ARIMA order by minimizing AIC over a small grid."""
    best_aic = float("inf")
    best_order = (1, 1, 1)
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(series, order=(p, 1, q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, 1, q)
            except Exception:
                continue
    return best_order, round(best_aic, 2)


def residual_stats(fitted_model) -> dict:
    resid = fitted_model.resid
    return {
        "mean": round(float(resid.mean()), 2),
        "std": round(float(resid.std()), 2),
        "max_abs": round(float(resid.abs().max()), 2),
    }
