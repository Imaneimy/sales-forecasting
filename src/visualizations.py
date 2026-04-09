"""
Charts for the sales forecasting project.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_raw_series(df, out="reports/raw_series.png"):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.index.astype(str), df["revenue"], marker="o", markersize=4, color="#4C72B0")
    ax.set_title("Monthly Revenue — 3 Years", fontsize=14, pad=12)
    ax.set_ylabel("Revenue (€)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


def plot_decomposition(decomp, out="reports/decomposition.png"):
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    components = [
        (decomp.observed, "Observed"),
        (decomp.trend, "Trend"),
        (decomp.seasonal, "Seasonality"),
        (decomp.resid, "Residuals"),
    ]
    for ax, (data, label) in zip(axes, components):
        ax.plot(data.index.astype(str), data.values, color="#4C72B0")
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[-1].tick_params(axis="x", rotation=45, labelsize=7)
    fig.suptitle("Seasonal Decomposition (Additive)", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, out)


def plot_forecast(train, test, forecast_values, out="reports/forecast.png"):
    fig, ax = plt.subplots(figsize=(12, 5))
    train_idx = [str(i) for i in train.index]
    test_idx = [str(i) for i in test.index]
    ax.plot(train_idx, train["revenue"].values, label="Train", color="#4C72B0", marker="o", markersize=3)
    ax.plot(test_idx, test["revenue"].values, label="Actual", color="#55A868", marker="o", markersize=4)
    ax.plot(test_idx, forecast_values, label="Forecast", color="#C44E52", linestyle="--", marker="x", markersize=5)
    ax.set_title("ARIMA Forecast vs Actual", fontsize=14, pad=12)
    ax.set_ylabel("Revenue (€)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    _save(fig, out)


def plot_residuals(fitted_model, out="reports/residuals.png"):
    resid = fitted_model.resid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(resid.values, color="#4C72B0", marker="o", markersize=3)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_title("Residuals over time")
    ax1.set_ylabel("Residual")
    ax2.hist(resid.values, bins=15, color="#4C72B0", edgecolor="white")
    ax2.set_title("Residuals distribution")
    ax2.set_xlabel("Residual")
    plt.tight_layout()
    _save(fig, out)
