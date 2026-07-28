# %%
"""
Phase 1 - EDA & stationarity.

Run this directly (`python notebooks/01_eda.py` from the repo root) or open it
in VS Code with the Jupyter extension and run cell-by-cell (the `# %%` markers
define cells). Either way it saves every plot to results/figures/ and prints
the numeric tests to the console, so you don't need an interactive window to
see the output.

What this phase answers, before touching any model:
  1. What does the series actually look like - trend? seasonality? shocks?
  2. Is it stationary? (ARIMA needs to know this to pick `d`)
  3. What do ACF/PACF suggest about AR/MA order?
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Works both as a script (__file__ defined) and inside a Jupyter kernel
# (__file__ undefined - falls back to the notebook's working directory,
# which Jupyter sets to the folder the .ipynb lives in, i.e. notebooks/).
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

FIG_DIR = os.path.join(BASE_DIR, "..", "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# %% Load
df = pd.read_csv(
    os.path.join(BASE_DIR, "..", "data", "people_analytics_monthly.csv"),
    parse_dates=["Date"],
)
df = df.set_index("Date").asfreq("MS")
print(df.shape)
print(df.head())

# %% Raw series plot
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
df["Headcount"].plot(ax=axes[0], title="Monthly Headcount")
df["Attrition_Rate_Pct"].plot(ax=axes[1], title="Monthly Attrition Rate (%)", color="firebrick")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_raw_series.png"), dpi=150)
plt.close()
print("Saved 01_raw_series.png - look for: overall trend, visible dips (2009 / 2020), any yearly wobble.")

# %% Seasonal decomposition
for col in ["Headcount", "Attrition_Rate_Pct"]:
    decomp = seasonal_decompose(df[col], model="additive", period=12)
    fig = decomp.plot()
    fig.set_size_inches(10, 7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"02_decomposition_{col}.png"), dpi=150)
    plt.close()
print("Saved decomposition plots - check the 'seasonal' panel: is there a repeating yearly pattern? "
      "check 'resid': is what's left over roughly noise, or does it still show the 2009/2020 dips?")


# %% Stationarity: Augmented Dickey-Fuller
def run_adf(series, label):
    result = adfuller(series.dropna())
    print(f"\nADF test - {label}")
    print(f"  ADF statistic: {result[0]:.4f}")
    print(f"  p-value:       {result[1]:.4f}")
    print(f"  -> {'STATIONARY (reject H0)' if result[1] < 0.05 else 'NOT stationary (fail to reject H0)'} at 5%")


run_adf(df["Headcount"], "Headcount (raw)")
run_adf(df["Headcount"].diff(), "Headcount (1st difference)")
run_adf(df["Attrition_Rate_Pct"], "Attrition_Rate_Pct (raw)")

# %% ACF / PACF - guides ARIMA (p, d, q) later
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
plot_acf(df["Attrition_Rate_Pct"].dropna(), ax=axes[0, 0], lags=36, title="ACF - Attrition Rate (raw)")
plot_pacf(df["Attrition_Rate_Pct"].dropna(), ax=axes[0, 1], lags=36, title="PACF - Attrition Rate (raw)")
plot_acf(df["Headcount"].diff().dropna(), ax=axes[1, 0], lags=36, title="ACF - Headcount (1st diff)")
plot_pacf(df["Headcount"].diff().dropna(), ax=axes[1, 1], lags=36, title="PACF - Headcount (1st diff)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_acf_pacf.png"), dpi=150)
plt.close()
print("\nSaved 03_acf_pacf.png - look for: how many lags stick out above the confidence band before "
      "cutting off, and whether there's a spike around lag 12 (seasonal signal).")

print("\nDone. Figures are in results/figures/. Report back: the two ADF p-values above, "
      "and what you see in 01_raw_series.png / 03_acf_pacf.png (trend? seasonality? spikes at which lags?).")
