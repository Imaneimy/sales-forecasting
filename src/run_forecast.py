"""
Entry point: load data, fit ARIMA, evaluate, generate charts.
"""

from pathlib import Path
from preprocessing import load_series, check_stationarity, decompose, train_test_split_ts, compute_baseline_metrics
from model import fit_arima, forecast, auto_select_order, residual_stats
from visualizations import plot_raw_series, plot_decomposition, plot_forecast, plot_residuals

DATA = Path(__file__).parent.parent / "data" / "monthly_sales.csv"
REPORTS = Path(__file__).parent.parent / "reports"
REPORTS.mkdir(exist_ok=True)


def main():
    df = load_series(DATA)
    print(f"Series: {len(df)} months from {df.index[0]} to {df.index[-1]}\n")

    stat = check_stationarity(df["revenue"])
    print(f"Stationarity (ADF): p={stat['p_value']} → {'stationary' if stat['is_stationary'] else 'not stationary — will difference'}")

    decomp = decompose(df["revenue"])

    train, test = train_test_split_ts(df, test_months=6)
    print(f"Train: {len(train)} months | Test: {len(test)} months")

    best_order, best_aic = auto_select_order(train["revenue"])
    print(f"\nBest ARIMA order: {best_order}  (AIC={best_aic})")

    fitted = fit_arima(train["revenue"], order=best_order)
    fc = forecast(fitted, steps=len(test))

    metrics = compute_baseline_metrics(test["revenue"], fc)
    print(f"\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    resid = residual_stats(fitted)
    print(f"\nResiduals: mean={resid['mean']}, std={resid['std']}")

    plot_raw_series(df, str(REPORTS / "raw_series.png"))
    plot_decomposition(decomp, str(REPORTS / "decomposition.png"))
    plot_forecast(train, test, fc.values, str(REPORTS / "forecast.png"))
    plot_residuals(fitted, str(REPORTS / "residuals.png"))

    print(f"\nCharts saved to {REPORTS}/")


if __name__ == "__main__":
    main()
