# %%
"""
Phase 4 - AR -> MA -> ARIMA -> SARIMA, built up one piece at a time.

Same evaluation convention as Phase 3: every model here is FIT ONCE on the
216-month training set, then forecasts the entire 24-month test horizon in a
single shot (`.forecast(24)`, or `.get_forecast(24)` for SARIMAX) - no
re-fitting, no peeking at test data. That keeps this phase directly
comparable to Phase 3's numbers and to Phase 2's static baseline.

Order selection (p, d, q) is done here from scratch on the TRAINING split
only, not read off Phase 1's plots. Phase 1's ACF/PACF/ADF were exploratory
(looking at the whole series is fine for "what does this data look like?").
Choosing (p, d, q) is a modeling decision, and modeling decisions - like
train/test splits - should never be informed by data the model will later be
graded on. In practice the conclusions end up matching Phase 1 anyway (same
data, same shape), which is itself a useful cross-check.

Build-up plan, one capability at a time:
  1. AR(p)     - ARIMA(p, d, 0): today depends on p of its own past values.
  2. MA(q)     - ARIMA(0, d, q): today depends on q of its own past shocks
                 (forecast errors), not past values. Same "MA" name as Phase
                 2's moving average baseline, unrelated math - see README.
  3. ARIMA(p,d,q) - both at once. Does combining actually help, or was one
                 half doing all the work?
  4. SARIMAX(p,d,q)x(P,D,Q,12) - adds a SEASONAL AR/MA/differencing layer on
                 top, since regular (non-seasonal) differencing removes trend
                 but does nothing about a repeating 12-month wave.
  5. pmdarima.auto_arima - an automated search, as a cross-check against the
                 manually chosen orders above.
"""
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf

warnings.filterwarnings("ignore")  # auto_arima's stepwise search fits and discards many candidate
                                    # models internally, some of which don't converge on purpose -
                                    # this only hides that internal chatter, not our own models below
                                    # (every model fit here was separately checked to converge cleanly,
                                    # via SARIMAX's maxiter=500 fix - see that cell)

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

sys.path.insert(0, os.path.join(BASE_DIR, "..", "src"))
from metrics import evaluate, comparison_table  # noqa: E402

FIG_DIR = os.path.join(BASE_DIR, "..", "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

TARGETS = ["Headcount", "Attrition_Rate_Pct"]
TEST_MONTHS = 24

# %% Load data, train/test split (identical setup to Phase 2 & 3)
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

# d (non-seasonal differencing order) - carried over from Phase 1's ADF findings,
# re-stated here rather than re-derived, since it's the same train-independent
# question either way: does this series need differencing to remove a unit root?
#   Headcount: raw ADF p=0.9976 (unit root) -> 1st diff ADF p=0.0000 (stationary) -> d=1
#   Attrition_Rate_Pct: raw ADF p=0.0026 (already stationary, trend-stationary per KPSS) -> d=0
D_ORDER = {"Headcount": 1, "Attrition_Rate_Pct": 0}

# %% Pick (p, q) from ACF/PACF on the TRAINING split, at the differencing order above
# Box-Jenkins heuristic: p = last lag before PACF drops inside the 95% confidence
# band (AR "cuts off" after lag p); q = same idea on ACF for MA. Computed here
# with statsmodels' own confidence intervals rather than eyeballing a plot, so
# it's reproducible and train-only.
def suggest_order(series, nlags=12, alpha=0.05):
    p_vals, p_ci = pacf(series, nlags=nlags, alpha=alpha)
    q_vals, q_ci = acf(series, nlags=nlags, alpha=alpha)

    def first_insignificant_lag(ci):
        for k in range(1, len(ci)):
            lower, upper = ci[k]
            if lower <= 0 <= upper:
                return k
        return len(ci) - 1

    p = max(first_insignificant_lag(p_ci) - 1, 0)
    q = max(first_insignificant_lag(q_ci) - 1, 0)
    return p, q


orders = {}
for col in TARGETS:
    stationary_train = train[col].diff(D_ORDER[col]).dropna() if D_ORDER[col] else train[col]
    p, q = suggest_order(stationary_train)
    orders[col] = {"p": p, "d": D_ORDER[col], "q": q}
    print(f"{col}: suggested (p, d, q) = ({p}, {D_ORDER[col]}, {q})  "
          f"[from PACF/ACF on {'the 1st-differenced' if D_ORDER[col] else 'the raw'} training series]")

# %% Method 1 - Pure AR(p): ARIMA(p, d, 0)
ar_forecasts, ar_models = {}, {}
for col in TARGETS:
    o = orders[col]
    model = ARIMA(train[col], order=(o["p"], o["d"], 0)).fit()
    ar_models[col] = model
    fc = model.forecast(TEST_MONTHS)
    fc.index = test[col].index
    ar_forecasts[col] = fc
    print(f"{col}: AR({o['p']}) fit, AIC={model.aic:.2f}")
    print(model.params.round(4).to_string())
    print()

# %% Method 2 - Pure MA(q): ARIMA(0, d, q)
ma_forecasts, ma_models = {}, {}
for col in TARGETS:
    o = orders[col]
    model = ARIMA(train[col], order=(0, o["d"], o["q"])).fit()
    ma_models[col] = model
    fc = model.forecast(TEST_MONTHS)
    fc.index = test[col].index
    ma_forecasts[col] = fc
    print(f"{col}: MA({o['q']}) fit, AIC={model.aic:.2f}")
    print(model.params.round(4).to_string())
    print()

# %% Method 3 - Combined ARIMA(p, d, q) - does combining AR and MA help?
arima_forecasts, arima_models = {}, {}
for col in TARGETS:
    o = orders[col]
    model = ARIMA(train[col], order=(o["p"], o["d"], o["q"])).fit()
    arima_models[col] = model
    fc = model.forecast(TEST_MONTHS)
    fc.index = test[col].index
    arima_forecasts[col] = fc
    print(f"{col}: ARIMA({o['p']},{o['d']},{o['q']}) fit, AIC={model.aic:.2f}")

print("\nAIC comparison (lower = better fit, penalized for extra parameters):")
for col in TARGETS:
    print(f"{col}: AR={ar_models[col].aic:.2f}  MA={ma_models[col].aic:.2f}  ARIMA={arima_models[col].aic:.2f}")

# %% Method 4 - SARIMAX: add a seasonal layer (period=12), since Phase 1 confirmed
# real yearly seasonality that non-seasonal differencing doesn't touch.
# Seasonal order (P, D, Q, 12) kept simple - one seasonal AR term, one seasonal
# difference, one seasonal MA term - rather than searched, so it can be compared
# against pmdarima's automated search in the next cell.
SEASONAL_ORDER = (1, 1, 1, 12)

sarimax_forecasts, sarimax_models = {}, {}
for col in TARGETS:
    o = orders[col]
    model = SARIMAX(
        train[col], order=(o["p"], o["d"], o["q"]), seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False, maxiter=500)  # default maxiter=50 wasn't enough for this many parameters to converge
    sarimax_models[col] = model
    fc = model.get_forecast(TEST_MONTHS).predicted_mean
    fc.index = test[col].index
    sarimax_forecasts[col] = fc
    print(f"{col}: SARIMAX({o['p']},{o['d']},{o['q']})x{SEASONAL_ORDER} fit, AIC={model.aic:.2f}")

# %% Method 5 - Cross-check: pmdarima.auto_arima's own search, vs. the manual picks above
auto_forecasts, auto_models = {}, {}
for col in TARGETS:
    model = pm.auto_arima(
        train[col], seasonal=True, m=12,
        max_p=4, max_q=4, max_P=2, max_Q=2,
        stepwise=True, suppress_warnings=True, error_action="ignore",
    )
    auto_models[col] = model
    fc = pd.Series(model.predict(TEST_MONTHS).values, index=test[col].index)
    auto_forecasts[col] = fc
    print(f"{col}: auto_arima picked order={model.order}  seasonal_order={model.seasonal_order}  AIC={model.aic():.2f}")
    print(f"       (manual pick was order=({orders[col]['p']},{orders[col]['d']},{orders[col]['q']})  "
          f"seasonal_order={SEASONAL_ORDER})")

# %% Evaluate everything against the test period (static, single-fit mode - same convention as Phase 3)
rows = []
for col in TARGETS:
    y_true = test[col]
    o = orders[col]
    rows.append({"Target": col, **evaluate(y_true, ar_forecasts[col], f"{col} - AR({o['p']})")})
    rows.append({"Target": col, **evaluate(y_true, ma_forecasts[col], f"{col} - MA({o['q']})")})
    rows.append({"Target": col, **evaluate(y_true, arima_forecasts[col], f"{col} - ARIMA({o['p']},{o['d']},{o['q']})")})
    rows.append({"Target": col, **evaluate(y_true, sarimax_forecasts[col], f"{col} - SARIMAX{SEASONAL_ORDER}")})
    rows.append({"Target": col, **evaluate(y_true, auto_forecasts[col], f"{col} - auto_arima")})

results = comparison_table(rows)
print("\nPhase 4 comparison, best (lowest RMSE) first within each target:")
print(results.to_string(index=False))

out_csv = os.path.join(BASE_DIR, "..", "results", "metrics_phase4_arima.csv")
results.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv}")

# %% Full-circle comparison: best Phase 4 method vs. Phase 2's static baseline and Phase 3's best, both targets
static_sma12 = {col: pd.Series(train[col].iloc[-12:].mean(), index=test[col].index) for col in TARGETS}
phase3_best_rmse = {"Headcount": 13.569907, "Attrition_Rate_Pct": 0.665794}  # from results/metrics_phase3_exp_smoothing.csv

print("Full-circle comparison across all phases so far:")
for col in TARGETS:
    static_row = evaluate(test[col], static_sma12[col], f"{col} - STATIC flat SMA(12)")
    best4 = results[results["Target"] == col].iloc[0]
    print(f"{col}: Phase 2 static={static_row['RMSE']:.3f}  Phase 3 best={phase3_best_rmse[col]:.3f}  "
          f"Phase 4 best={best4['RMSE']:.3f} ({best4['Method'].split(' - ')[1]})\n")

# %% Plot: actual vs each Phase 4 forecast, zoomed to last 12 train + 24 test months
fig, axes = plt.subplots(2, 1, figsize=(11, 8))
for ax, col in zip(axes, TARGETS):
    zoom = df[col].iloc[-(TEST_MONTHS + 12):]
    ax.plot(zoom.index, zoom.values, label="Actual", color="black", linewidth=2)
    ax.axvline(test[col].index[0], color="gray", linestyle=":", label="Train/test split")
    ax.plot(test[col].index, ar_forecasts[col], label=f"AR({orders[col]['p']})", linestyle="--")
    ax.plot(test[col].index, ma_forecasts[col], label=f"MA({orders[col]['q']})", linestyle="--")
    ax.plot(test[col].index, arima_forecasts[col], label="ARIMA", linestyle="--")
    ax.plot(test[col].index, sarimax_forecasts[col], label="SARIMAX", linestyle="--")
    ax.set_title(f"{col} - AR/MA/ARIMA/SARIMAX forecasts vs actual")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "08_arima_forecasts.png"), dpi=150)
plt.close()
print("Saved 08_arima_forecasts.png")

print("\nDone. Report back: did combining AR+MA into ARIMA beat the pure AR and pure MA versions? "
      "Did SARIMAX's seasonal layer beat plain ARIMA? Did auto_arima agree with the manual (p,d,q) picks? "
      "And how does the best Phase 4 method compare to Phase 3's best and Phase 2's static baseline?")
