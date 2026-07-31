"""
src/metrics.py

Shared evaluation helpers used by every forecasting notebook from Phase 2
onward, so every method (moving average, exponential smoothing, AR/MA/ARIMA,
SARIMA) is scored the same way and results are directly comparable in
Phase 5's final table.
"""
import numpy as np
import pandas as pd


def mae(y_true, y_pred):
    """Mean Absolute Error - average size of the miss, in the original units
    (e.g. 'off by 6.3 people' or 'off by 0.8 percentage points')."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Root Mean Squared Error - like MAE but squares errors before averaging,
    so a few big misses hurt the score more than many small ones."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error - error as a % of the actual value, so
    it's comparable across series with different scales (Headcount vs a %
    rate). Unstable when y_true is near zero (Attrition_Rate_Pct hits 0.0 in
    a few months) - those months are excluded from the denominator."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred, label=None):
    """Return a one-row dict of all three metrics, ready to append into a
    comparison DataFrame."""
    row = {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE_%": mape(y_true, y_pred)}
    if label is not None:
        row = {"Method": label, **row}
    print(f"{label or '':>28s}  MAE={row['MAE']:.4f}  RMSE={row['RMSE']:.4f}  MAPE={row['MAPE_%']:.2f}%")
    return row


def comparison_table(rows):
    """rows: list of dicts from evaluate(). Returns a DataFrame sorted best
    (lowest RMSE) first."""
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
