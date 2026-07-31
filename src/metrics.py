"""
Shared forecast-accuracy metrics, used from Phase 2 onward so every method
(AR, MA, ARIMA, and anything added later) is scored the same way and results
are directly comparable in results/model_comparison.csv.
"""
import numpy as np


def mae(y_true, y_pred):
    """Mean Absolute Error - average size of the miss, in the original units
    (e.g. 'on average we're off by 1.3 percentage points of attrition')."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Root Mean Squared Error - same units as MAE, but squares errors before
    averaging, so big misses are punished disproportionately more than small
    ones. RMSE >> MAE for the same model usually means a few large misses
    (e.g. right at a shock) rather than uniformly-mediocre misses."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error - MAE expressed as a % of the actual
    value, so it's comparable across series of different scale (e.g. compare
    error on Headcount, in the hundreds, against error on Attrition_Rate_Pct,
    in single digits). Undefined / unstable if y_true has values near 0."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def summarize(y_true, y_pred, label=""):
    m = {"model": label, "MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE_%": mape(y_true, y_pred)}
    print(f"{label:>20s}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  MAPE={m['MAPE_%']:.2f}%")
    return m
