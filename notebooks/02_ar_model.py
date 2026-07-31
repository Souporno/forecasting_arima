# %%
"""
Phase 2 - pure AR (AutoRegressive) model on Attrition_Rate_Pct.

Recall the mental model from Phase 1: Z(t) = Phi_1*Z(t-1) + Phi_2*Z(t-2) + ... + E(t)
No MA term yet (that's Phase 3), no differencing (Attrition_Rate_Pct passed
ADF without it - though remember, it still has seasonality, which a plain AR
model has no way to represent; watch for that in the results below).

From the Phase 1 ACF/PACF: PACF cut off sharply after lag ~2-3, ACF decayed
slowly - the textbook signature of an AR process, with AR(2) or AR(3) as the
starting guess. This notebook fits AR(1) through AR(4), lets statsmodels'
AIC-based selector pick its own order independently, and checks which one
actually forecasts best on held-out data (AIC and "forecasts best" don't
always agree - worth seeing directly).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg, ar_select_order

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

sys.path.insert(0, os.path.join(BASE_DIR, "..", "src"))
from metrics import summarize  # noqa: E402

FIG_DIR = os.path.join(BASE_DIR, "..", "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TARGET = "Attrition_Rate_Pct"
TEST_MONTHS = 24  # held-out test window - kept the same across every phase
                   # from here on, so AR vs MA vs ARIMA are scored apples-to-apples

# %% Load + train/test split
df = pd.read_csv(
    os.path.join(BASE_DIR, "..", "data", "people_analytics_monthly.csv"),
    parse_dates=["Date"],
).set_index("Date").asfreq("MS")

y = df[TARGET]
train, test = y.iloc[:-TEST_MONTHS], y.iloc[-TEST_MONTHS:]
print(f"train: {train.index.min().date()} to {train.index.max().date()} ({len(train)} months)")
print(f"test:  {test.index.min().date()} to {test.index.max().date()} ({len(test)} months)")

# %% Let statsmodels pick its own AR order via AIC, for comparison
selected = ar_select_order(train, maxlag=12, ic="aic", old_names=False)
print(f"\nAIC-selected lags: {selected.ar_lags}")

# %% Fit AR(1) .. AR(4) plus the AIC-selected order, forecast the test window
candidates = {"AR(1)": [1], "AR(2)": [1, 2], "AR(3)": [1, 2, 3], "AR(4)": [1, 2, 3, 4]}
if selected.ar_lags and list(selected.ar_lags) not in candidates.values():
    candidates[f"AR(AIC={list(selected.ar_lags)})"] = list(selected.ar_lags)

results = []
forecasts = {}
for name, lags in candidates.items():
    model = AutoReg(train, lags=lags, old_names=False).fit()
    fc = model.forecast(steps=TEST_MONTHS)
    forecasts[name] = fc
    print(f"\n{name}  (AIC={model.aic:.2f}, BIC={model.bic:.2f})")
    print("  coefficients (const, then phi_1, phi_2, ...):")
    print("  " + ", ".join(f"{v:.4f}" for v in model.params.values))
    m = summarize(test.values, fc.values, label=name)
    m["AIC"] = model.aic
    m["BIC"] = model.bic
    results.append(m)

results_df = pd.DataFrame(results).set_index("model")
print("\n" + "=" * 60)
print("Comparison (lower is better for all columns)")
print("=" * 60)
print(results_df.round(3))

# %% Plot actual vs each candidate's forecast over the test window
fig, ax = plt.subplots(figsize=(11, 5))
train.iloc[-36:].plot(ax=ax, label="train (last 3y shown)", color="black")
test.plot(ax=ax, label="actual (test)", color="black", linewidth=2.5)
for name, fc in forecasts.items():
    fc.plot(ax=ax, label=name, linestyle="--", alpha=0.8)
ax.axvline(test.index[0], color="gray", linestyle=":")
ax.set_title(f"AR model forecasts vs actual - {TARGET}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_ar_forecasts.png"), dpi=150)
plt.close()
print("\nSaved 04_ar_forecasts.png")

results_df.to_csv(os.path.join(BASE_DIR, "..", "results", "ar_model_comparison.csv"))
print("Saved results/ar_model_comparison.csv")
print("\nReport back: which AR order won on test MAE/RMSE, whether it matches the AIC-selected "
      "order, and what the forecast plot looks like - does the AR forecast flatten out / miss the "
      "seasonal wiggle in the actual test data?")
