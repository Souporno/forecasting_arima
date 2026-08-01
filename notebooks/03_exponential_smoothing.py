# %%
"""
Phase 3 - Exponential Smoothing: SES -> Holt's linear trend -> Holt-Winters
seasonal.

IMPORTANT evaluation-mode change from Phase 2: naive/SMA/WMA were cheap
arithmetic, recomputed fresh every month (ROLLING one-step-ahead), so they
always had real, up-to-date data feeding them. The models here are properly
FIT (parameters estimated via optimization), which is normally done ONCE on
the training data - then used to forecast the entire 24-month test horizon
in a single shot via .forecast(24). That's the same "commit once, never
update" setup as Phase 2 cell 9's STATIC demo, not the rolling cells before
it.

That makes Phase 2's static flat forecast for Headcount (RMSE = 34.43) the
fair number to beat here - not the rolling SMA/naive numbers, since those
got fed fresh data every month and these models don't. (Phase 5 will
properly backtest everything, including these models, with rolling re-fits,
for the final apples-to-apples comparison across every phase.)

Method summary (each one adds exactly one new capability):
  SES           - smoothed LEVEL only, nothing else. Same blind spot as
                  moving average: forecasts a flat line forever. Included on
                  purpose to prove "exponential smoothing" per se isn't the
                  fix for trend-blindness - it's specifically the TREND term
                  Holt adds next.
  Holt          - LEVEL + TREND. Forecasts level + h*trend for h months
                  ahead - literally "notice the slope, keep going", the
                  capability plain averaging structurally cannot have.
  Holt-Winters  - LEVEL + TREND + SEASONAL. Adds the repeating 12-month wave
                  Phase 1 confirmed is real in both target series.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

sys.path.insert(0, os.path.join(BASE_DIR, "..", "src"))
from metrics import evaluate, comparison_table  # noqa: E402

FIG_DIR = os.path.join(BASE_DIR, "..", "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TARGETS = ["Headcount", "Attrition_Rate_Pct"]
TEST_MONTHS = 24  # same held-out window as every other phase, for fair comparison

# %% Load data, train/test split (identical setup to Phase 2)
df = pd.read_csv(
    os.path.join(BASE_DIR, "..", "data", "people_analytics_monthly.csv"),
    parse_dates=["Date"],
)
df = df.set_index("Date").asfreq("MS")

train = {col: df[col].iloc[:-TEST_MONTHS] for col in TARGETS}
test = {col: df[col].iloc[-TEST_MONTHS:] for col in TARGETS}
print(f"train: {train['Headcount'].index.min().date()} to {train['Headcount'].index.max().date()} "
      f"({len(train['Headcount'])} months)")
print(f"test:  {test['Headcount'].index.min().date()} to {test['Headcount'].index.max().date()} "
      f"({len(test['Headcount'])} months)")

# %% Method 1 - Simple Exponential Smoothing (level only)
# Fit ONCE on training data, forecast the whole 24-month horizon in one shot.
# alpha ("smoothing_level") controls how much weight recent observations get
# vs. older ones when estimating the current level - closer to 1 means
# "trust the most recent value almost entirely", closer to 0 means "barely
# update the level from what it's always been".
ses_forecasts, ses_models = {}, {}
for col in TARGETS:
    model = SimpleExpSmoothing(train[col], initialization_method="estimated").fit(optimized=True)
    ses_models[col] = model
    fc = model.forecast(TEST_MONTHS)
    fc.index = test[col].index
    ses_forecasts[col] = fc
    print(f"{col}: SES alpha (smoothing_level) = {model.params['smoothing_level']:.4f}")

print("\nSES preview - first 6 months of the forecast horizon:")
for col in TARGETS:
    print(f"\n{col}:")
    print(pd.DataFrame({f"{col} (actual)": test[col], "SES forecast": ses_forecasts[col]}).head(6))

# %% Method 2 - Holt's linear trend (level + trend)
holt_forecasts, holt_models = {}, {}
for col in TARGETS:
    model = Holt(train[col], initialization_method="estimated").fit(optimized=True)
    holt_models[col] = model
    fc = model.forecast(TEST_MONTHS)
    fc.index = test[col].index
    holt_forecasts[col] = fc
    print(f"{col}: Holt alpha={model.params['smoothing_level']:.4f}  beta (trend)={model.params['smoothing_trend']:.4f}")

print("\nHolt preview - first 6 months (compare to SES above - does adding trend change the shape?):")
for col in TARGETS:
    print(f"\n{col}:")
    print(pd.DataFrame({
        f"{col} (actual)": test[col],
        "SES forecast": ses_forecasts[col],
        "Holt forecast": holt_forecasts[col],
    }).head(6))

# %% Method 3 - Holt-Winters seasonal (level + trend + seasonality, period=12)
hw_forecasts, hw_models = {}, {}
for col in TARGETS:
    model = ExponentialSmoothing(
        train[col], trend="add", seasonal="add", seasonal_periods=12, initialization_method="estimated"
    ).fit(optimized=True)
    hw_models[col] = model
    fc = model.forecast(TEST_MONTHS)
    fc.index = test[col].index
    hw_forecasts[col] = fc
    print(f"{col}: Holt-Winters alpha={model.params['smoothing_level']:.4f}  "
          f"beta={model.params['smoothing_trend']:.4f}  gamma (seasonal)={model.params['smoothing_seasonal']:.4f}")

# %% Evaluate all three against the test period (static, single-fit mode - see docstring)
rows = []
for col in TARGETS:
    y_true = test[col]
    rows.append({"Target": col, **evaluate(y_true, ses_forecasts[col], f"{col} - SES")})
    rows.append({"Target": col, **evaluate(y_true, holt_forecasts[col], f"{col} - Holt (level+trend)")})
    rows.append({"Target": col, **evaluate(y_true, hw_forecasts[col], f"{col} - Holt-Winters (level+trend+seasonal)")})

results = comparison_table(rows)
print("\nExponential smoothing comparison, best (lowest RMSE) first within each target:")
print(results.to_string(index=False))

out_csv = os.path.join(BASE_DIR, "..", "results", "metrics_phase3_exp_smoothing.csv")
results.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv}")

# %% Full-circle comparison: best exp-smoothing method vs. a static moving-average baseline, BOTH targets
# Recomputed fresh here (not hardcoded from Phase 2's notebook) so this stays correct even if Phase 2
# changes later, and so Attrition gets the same fair, apples-to-apples baseline Headcount already had -
# same "fit once on train, hold flat for the whole test horizon" rule as every model in this notebook,
# same idea as Phase 2 cell 9's static demo, just computed for both series instead of only Headcount.
static_sma12 = {col: pd.Series(train[col].iloc[-12:].mean(), index=test[col].index) for col in TARGETS}

print("Full-circle comparison: best exponential smoothing method vs. a static SMA(12) baseline:")
for col in TARGETS:
    static_row = evaluate(test[col], static_sma12[col], f"{col} - STATIC flat SMA(12)")
    best = results[results["Target"] == col].iloc[0]
    verdict = "beats" if best["RMSE"] < static_row["RMSE"] else "does NOT beat"
    print(f"{col}: static SMA(12) RMSE={static_row['RMSE']:.3f}  vs.  best exp-smoothing "
          f"({best['Method'].split(' - ')[1]}) RMSE={best['RMSE']:.3f}  -> exp-smoothing {verdict} the static baseline\n")

# %% Plot: actual vs each exponential smoothing forecast
# Zoomed to the last 12 months of training + all 24 test months, so the
# forecasts are visible in detail rather than squashed at the edge of a
# 240-month chart.
fig, axes = plt.subplots(2, 1, figsize=(11, 8))
for ax, col in zip(axes, TARGETS):
    zoom = df[col].iloc[-(TEST_MONTHS + 12):]
    ax.plot(zoom.index, zoom.values, label="Actual", color="black", linewidth=2)
    ax.axvline(test[col].index[0], color="gray", linestyle=":", label="Train/test split")
    ax.plot(test[col].index, ses_forecasts[col], label="SES", linestyle="--")
    ax.plot(test[col].index, holt_forecasts[col], label="Holt", linestyle="--")
    ax.plot(test[col].index, hw_forecasts[col], label="Holt-Winters", linestyle="--")
    ax.set_title(f"{col} - exponential smoothing (fit once on train, forecast {TEST_MONTHS} months ahead)")
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "07_exp_smoothing_forecasts.png"), dpi=150)
plt.close()
print("Saved 07_exp_smoothing_forecasts.png")

print("\nDone. Report back: did Holt actually beat SES on Headcount (does adding a trend term help "
      "when forecasting a trending series 24 months out)? Did Holt-Winters beat Holt on either series "
      "(does adding seasonality help)? And how does the best method here compare to Phase 2's static "
      "baseline?")
