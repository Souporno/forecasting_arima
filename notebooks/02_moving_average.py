# %%
"""
Phase 2 - Baselines: naive forecast, simple moving average, weighted moving
average.

These are the floor every later model (exponential smoothing, ARIMA/SARIMA)
needs to beat. If Holt-Winters or SARIMA can't clearly outperform a plain
moving average, that's a sign they're not adding real value on this data.

Two different evaluation modes are used here, on purpose - see the two halves
of this notebook:

  A) ROLLING one-step-ahead (cells 3-6): at every month in the test period,
     forecast just the NEXT month using real, actual past values. This is how
     these methods are normally used in practice - you always feed them the
     latest real data.

  B) STATIC multi-step (cell 7): compute one forecast using ONLY the training
     data, then hold it flat and use it for the entire 24-month test horizon
     with no updates. This is a harsher, "what if we planned two years out
     today and never looked back" test - and it's specifically designed to
     expose moving average's blindness to trend.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

sys.path.insert(0, os.path.join(BASE_DIR, "..", "src"))
from metrics import evaluate, comparison_table  # noqa: E402

FIG_DIR = os.path.join(BASE_DIR, "..", "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TARGETS = ["Headcount", "Attrition_Rate_Pct"]
TEST_MONTHS = 24  # hold out the last 2 years to evaluate on

# %% Load data, train/test split
df = pd.read_csv(
    os.path.join(BASE_DIR, "..", "data", "people_analytics_monthly.csv"),
    parse_dates=["Date"],
)
df = df.set_index("Date").asfreq("MS")

train = df.iloc[:-TEST_MONTHS]
test = df.iloc[-TEST_MONTHS:]
print(f"train: {train.index.min().date()} to {train.index.max().date()}  ({len(train)} months)")
print(f"test:  {test.index.min().date()} to {test.index.max().date()}  ({len(test)} months)")

# %% Method 1 - Naive forecast (rolling one-step): predict next month = this month
# df[col].shift(1) at row t holds the actual value from t-1 - i.e. "yesterday's
# value" used to predict "today". This is the simplest possible baseline: a
# model beaten by naive isn't learning anything useful from the data.
naive_preds = {col: df[col].shift(1) for col in TARGETS}

print("Naive forecast preview - the forecast column is just each column shifted down 1 row:")
for col in TARGETS:
    print(f"\n{col}:")
    print(pd.DataFrame({
        f"{col} (actual)": df[col],
        "Naive forecast (= prior month's actual)": naive_preds[col],
    }).head(6))

# %% Method 2 - Simple moving average (rolling one-step), window sweep
# shift(1) first (don't peek at the value we're trying to predict), then
# .rolling(w).mean() averages the w actual values before that.
#
# Window choice: 3/6/12 months - quarterly, half-yearly, annual. These are
# the natural reporting cadences a People Analytics team would already be
# looking at (quarterly business reviews, half-year/annual HR cycles), so
# they double as a sanity check on whether "the cadence you'd naturally
# report on" also happens to forecast well.
SMA_WINDOWS = [3, 6, 12]
sma_preds = {
    col: {w: df[col].shift(1).rolling(window=w).mean() for w in SMA_WINDOWS}
    for col in TARGETS
}

print("SMA preview (rows 10-17) - small window (3) vs large window (12), side by side:")
for col in TARGETS:
    print(f"\n{col}:")
    print(pd.DataFrame({
        f"{col} (actual)": df[col],
        "SMA(3) forecast": sma_preds[col][3],
        "SMA(12) forecast": sma_preds[col][12],
    }).iloc[10:18])

# %% Method 3 - Weighted moving average (rolling one-step)
# Same idea as SMA, but recent months count more. Weights [1, 2, ..., w]
# normalized to sum to 1, so the most recent of the w months gets the most
# influence instead of every month counting equally.
#
# Built at the SAME window sizes as SMA (3/6/12), on purpose: that way every
# SMA(w) has a matching WMA(w) at an identical window, isolating exactly one
# variable - equal weighting vs. recency weighting - at each window size,
# instead of getting just one data point on whether weighting helps.
WMA_WINDOWS = SMA_WINDOWS  # [3, 6, 12] - deliberately matched to SMA


def weighted_moving_average(series, window):
    weights = np.arange(1, window + 1)

    def _wavg(x):
        return np.dot(x, weights) / weights.sum()

    return series.shift(1).rolling(window=window).apply(_wavg, raw=True)


wma_preds = {
    col: {w: weighted_moving_average(df[col], w) for w in WMA_WINDOWS}
    for col in TARGETS
}

print("WMA preview (rows 10-17) - one example window (6) shown; also computed at 3 and 12:")
for col in TARGETS:
    print(f"\n{col}:")
    print(pd.DataFrame({
        f"{col} (actual)": df[col],
        "SMA(6) forecast (equal weights, for contrast)": sma_preds[col][6],
        "WMA(6) forecast (recent-weighted)": wma_preds[col][6],
    }).iloc[10:18])

# %% Evaluate every method on the test period
# Note what "evaluate" actually does: for each method, compare its forecast
# to the REAL test-period actual values and score how far off it was
# (MAE/RMSE/MAPE - smaller = closer to the truth). Nothing gets compared to
# the naive forecast directly here; naive is just one more row scored the
# same way, so it acts as a floor other methods should beat.
rows = []
for col in TARGETS:
    y_true = test[col]

    rows.append({"Target": col, **evaluate(y_true, naive_preds[col].loc[test.index], f"{col} - Naive")})
    for w in SMA_WINDOWS:
        rows.append({"Target": col, **evaluate(y_true, sma_preds[col][w].loc[test.index], f"{col} - SMA({w})")})
    for w in WMA_WINDOWS:
        rows.append({"Target": col, **evaluate(y_true, wma_preds[col][w].loc[test.index], f"{col} - WMA({w})")})

results = comparison_table(rows)  # sorted by Target, then RMSE (best first) within each Target
print("\nFull comparison, best (lowest RMSE) first within each target:")
print(results.to_string(index=False))

out_csv = os.path.join(BASE_DIR, "..", "results", "metrics_phase2_baselines.csv")
results.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv}")

# %% Does recency-weighting help consistently across window sizes, or just at one?
# Head-to-head: SMA(w) vs WMA(w) at each matching window, same target.
print("\nSMA vs WMA head-to-head, same window sizes:")
head_to_head = []
for col in TARGETS:
    print(f"\n{col}:")
    for w in SMA_WINDOWS:
        sma_rmse = results.loc[results["Method"] == f"{col} - SMA({w})", "RMSE"].iloc[0]
        wma_rmse = results.loc[results["Method"] == f"{col} - WMA({w})", "RMSE"].iloc[0]
        winner = "WMA" if wma_rmse < sma_rmse else "SMA"
        print(f"  window={w:>2}:  SMA RMSE={sma_rmse:.3f}   WMA RMSE={wma_rmse:.3f}   -> {winner} wins")
        head_to_head.append({"Target": col, "Window": w, "SMA_RMSE": sma_rmse, "WMA_RMSE": wma_rmse, "Winner": winner})
head_to_head = pd.DataFrame(head_to_head)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.arange(len(SMA_WINDOWS))
for ax, col in zip(axes, TARGETS):
    sub = head_to_head[head_to_head["Target"] == col]
    ax.bar(x - 0.18, sub["SMA_RMSE"], width=0.36, label="SMA")
    ax.bar(x + 0.18, sub["WMA_RMSE"], width=0.36, label="WMA")
    ax.set_xticks(x)
    ax.set_xticklabels([f"window={w}" for w in SMA_WINDOWS])
    ax.set_ylabel("RMSE (lower = better)")
    ax.set_title(f"{col}: SMA vs WMA at each window")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_sma_vs_wma.png"), dpi=150)
plt.close()
print("\nSaved 06_sma_vs_wma.png")

# %% Plot: actual vs each rolling one-step forecast, over the test period
best_sma_window = {col: min(SMA_WINDOWS, key=lambda w: np.abs(sma_preds[col][w].loc[test.index] - test[col]).mean()) for col in TARGETS}

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
for ax, col in zip(axes, TARGETS):
    ax.plot(test.index, test[col], label="Actual", color="black", linewidth=2)
    ax.plot(test.index, naive_preds[col].loc[test.index], label="Naive", linestyle="--")
    w = best_sma_window[col]
    ax.plot(test.index, sma_preds[col][w].loc[test.index], label=f"SMA({w})", linestyle="--")
    ax.plot(test.index, wma_preds[col][w].loc[test.index], label=f"WMA({w})", linestyle="--")
    ax.set_title(f"{col} - rolling one-step forecasts vs actual (test period)")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_baseline_rolling_forecasts.png"), dpi=150)
plt.close()
print("Saved 04_baseline_rolling_forecasts.png")

# %% Illustrative: STATIC multi-step forecast for Headcount (mode B, see docstring)
# Compute ONE forecast using only the last 12 months of TRAINING data, then
# hold it flat across the entire 24-month test horizon - no peeking at any
# test-period actuals, ever. This simulates "plan two years out today and
# don't revisit it" and is designed to make moving average's core weakness
# undeniable: it has no concept of trend, so it can only ever predict
# "more of the recent average" - never "keeps growing".
static_forecast_value = train["Headcount"].iloc[-12:].mean()
static_forecast = pd.Series(static_forecast_value, index=test.index)

static_row = evaluate(test["Headcount"], static_forecast, "Headcount - STATIC flat SMA(12)")
rolling_row = [r for r in rows if r["Method"] == "Headcount - SMA(12)"][0]
print(f"\nFor comparison, the ROLLING SMA(12) RMSE on Headcount was {rolling_row['RMSE']:.2f} "
      f"vs the STATIC flat forecast's RMSE of {static_row['RMSE']:.2f} - "
      "rolling 'cheats' by getting fed real recent data every month; static shows what "
      "moving average is actually capable of when forecasting further ahead without updates.")

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(df.index, df["Headcount"], label="Actual (full history)", color="black")
ax.axvline(test.index[0], color="gray", linestyle=":", label="Train/test split")
ax.plot(test.index, static_forecast, label="Static flat forecast (from train only)", color="firebrick", linewidth=2)
ax.set_title("Headcount - why a flat moving-average forecast fails on a trending series")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_static_forecast_vs_trend.png"), dpi=150)
plt.close()
print("Saved 05_static_forecast_vs_trend.png")

print("\nDone. Report back: which method won on Headcount vs on Attrition_Rate_Pct "
      "(they may differ - Headcount has strong trend, Attrition doesn't), and what the "
      "static-forecast plot looks like.")
