# sales-forecasting
![Tests](https://github.com/Imaneimy/sales-forecasting/actions/workflows/tests.yml/badge.svg)

At Orange Maroc I was regularly asked to project next month's procurement spend based on historical trends. The forecasting was done manually in Excel. This project is a Python implementation of that process — an ARIMA model trained on 30 months of monthly revenue data and evaluated on a 6-month holdout.

The dataset is 36 months of synthetic monthly sales (2022-2024) with an upward trend and clear December/summer seasonality, similar in shape to the actual procurement data I worked with. The model pipeline covers stationarity testing (ADF), seasonal decomposition, ARIMA order selection by AIC, forecasting, and residual diagnostics.

## Structure

```
src/
  preprocessing.py    # load, stationarity test (ADF), decomposition, train/test split, metrics
  model.py            # ARIMA fit, forecast, auto order selection by AIC, residual stats
  visualizations.py   # 4 charts: raw series, decomposition, forecast vs actual, residuals
  run_forecast.py     # entry point

tests/
  test_preprocessing.py   # 12 unit tests TC-TS-001→012
  test_model.py           # 8 unit tests TC-MOD-TS-001→008

data/
  monthly_sales.csv   # 36 months, 2022-2024
  generate_data.py    # script that generated the data

reports/              # generated charts (git-ignored)
```

## Running it

```bash
pip install -r requirements.txt
cd src
python run_forecast.py
```

Prints stationarity result, selected ARIMA order, test MAE/RMSE/MAPE, residual stats, and saves 4 charts to `reports/`.

```bash
pytest tests/ -v
```

## How the model is selected

`auto_select_order()` loops over p ∈ [0,3] and q ∈ [0,3] with d=1 (first difference) and picks the combination with the lowest AIC. On this dataset the winning order is usually (1,1,1) or (2,1,1).

## What I would add with real data

With actual procurement data from SAP I'd add external regressors (budget cycles, supplier contracts) using ARIMAX, and wrap the forecast in a confidence interval plot so stakeholders can see the uncertainty range — which is what procurement teams actually need for decision-making.

## Stack

Python, Pandas, Statsmodels, Matplotlib, Pytest
