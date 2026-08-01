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
# Window choice: rather than hand-picking "round" numbers like 3/6/12
# (quarter/half-year/year) and hoping they're good, sweep every window from
# 2 to 18 months and let cell 6's evaluation find the actual best one - see
# the RMSE-vs-window plot later in this notebook for the data-driven answer.
SMA_WINDOWS = list(range(2, 19))
sma_preds = {
    col: {w: df[col].shift(1).rolling(window=w).mean() for w in SMA_WINDOWS}
    for col in TARGETS
}

print(f"Computed SMA forecasts for {len(SMA_WINDOWS)} window sizes ({min(SMA_WINDOWS)}-{max(SMA_WINDOWS)} months) per target.")
print("Preview (rows 10-17) - just 2 of those windows shown side by side, small vs large:")
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
WMA_WINDOW = 6


def weighted_moving_average(series, window):
    weights = np.arange(1, window + 1)

    def _wavg(x):
        return np.dot(x, weights) / weights.sum()

    return series.shift(1).rolling(window=window).apply(_wavg, raw=True)


wma_preds = {col: weighted_moving_average(df[col], WMA_WINDOW) for col in TARGETS}

print(f"WMA preview (rows 10-17) - recent months of the last {WMA_WINDOW} count more than older ones:")
for col in TARGETS:
    print(f"\n{col}:")
    print(pd.DataFrame({
        f"{col} (actual)": df[col],
        f"SMA({WMA_WINDOW}) forecast (equal weights, for contrast)": sma_preds[col][6],
        f"WMA({WMA_WINDOW}) forecast (recent-weighted)": wma_preds[col],
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
        row = {"Target": col, "Window": w, **evaluate(y_true, sma_preds[col][w].loc[test.index], f"{col} - SMA({w})")}
        rows.append(row)
    rows.append({"Target": col, **evaluate(y_true, wma_preds[col].loc[test.index], f"{col} - WMA({WMA_WINDOW})")})

results = comparison_table(rows)  # sorted by Target, then RMSE (best first) within each Target
print("\nFull comparison, best (lowest RMSE) first within each target:")
print(results.drop(columns="Window").to_string(index=False))

out_csv = os.path.join(BASE_DIR, "..", "results", "metrics_phase2_baselines.csv")
results.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv}")

# %% Answering "why these window sizes" properly: RMSE vs. window size, swept
sma_results = results[results["Window"].notna()].copy()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, col in zip(axes, TARGETS):
    sub = sma_results[sma_results["Target"] == col].sort_values("Window")
    ax.plot(sub["Window"], sub["RMSE"], marker="o")
    best = sub.loc[sub["RMSE"].idxmin()]
    ax.scatter([best["Window"]], [best["RMSE"]], color="red", zorder=5, label=f"best: window={int(best['Window'])}")
    ax.set_xlabel("SMA window (months)")
    ax.set_ylabel("RMSE (lower = better)")
    ax.set_title(f"{col}: RMSE vs. SMA window size")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_sma_window_sweep.png"), dpi=150)
plt.close()

for col in TARGETS:
    sub = sma_results[sma_results["Target"] == col]
    best = sub.loc[sub["RMSE"].idxmin()]
    print(f"{col}: data-driven best SMA window = {int(best['Window'])} months (RMSE={best['RMSE']:.3f}) "
          f"out of {sub['Window'].min():.0f}-{sub['Window'].max():.0f} swept")
print("Saved 06_sma_window_sweep.png")

# %% Plot: actual vs each rolling one-step forecast, over the test period
# Reuse the RMSE-minimizing window found by the sweep above, so this plot is
# showing the same "best" window the numbers actually picked.
best_sma_window = {
    col: int(sma_results[sma_results["Target"] == col].loc[sma_results[sma_results["Target"] == col]["RMSE"].idxmin(), "Window"])
    for col in TARGETS
}

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)
for ax, col in zip(axes, TARGETS):
    ax.plot(test.index, test[col], label="Actual", color="black", linewidth=2)
    ax.plot(test.index, naive_preds[col].loc[test.index], label="Naive", linestyle="--")
    w = best_sma_window[col]
    ax.plot(test.index, sma_preds[col][w].loc[test.index], label=f"SMA({w})", linestyle="--")
    ax.plot(test.index, wma_preds[col].loc[test.index], label=f"WMA({WMA_WINDOW})", linestyle="--")
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
