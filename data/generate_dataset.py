"""
generate_dataset.py

Generates `people_analytics_monthly.csv`: a synthetic but realistically-structured
monthly People Analytics time series for one mid-size company, Jan 2006 - Dec 2025
(240 months).

Why synthetic: real, ready-to-use TIME SERIES people-analytics datasets are scarce.
Public HR datasets (IBM HR Attrition, "HRDataset_v14", UCI Absenteeism-at-Work) are
almost all CROSS-SECTIONAL (one row per employee), not date-indexed series suitable
for ARIMA/exponential smoothing/moving-average forecasting. This script builds a
monthly panel with a realistic trend, yearly seasonality, and two macro shocks
(2009 recession, 2020 pandemic + 2021-22 "Great Resignation"), so the resulting
series has properties worth modeling: trend + seasonality + noise + regime shifts.

Fully reproducible: fixed random seed, no external calls.

Columns:
    Date                    first-of-month date (YYYY-MM-DD)
    Headcount               active employees at month end
    Hires                   new hires during the month
    Voluntary_Terminations  voluntary exits during the month
    Involuntary_Terminations involuntary exits (layoffs/performance) during the month
    Total_Terminations      Voluntary_Terminations + Involuntary_Terminations
    Attrition_Rate_Pct      Total_Terminations / avg(Headcount_prev, Headcount) * 100
    Avg_Engagement_Score    mean engagement survey score, 1-5 scale
    Avg_Satisfaction_Score  mean employee satisfaction score, 1-5 scale
    Open_Requisitions       open job requisitions at month end (recruiting pipeline)
"""

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

N_MONTHS = 240  # Jan 2006 - Dec 2025
dates = pd.date_range("2006-01-01", periods=N_MONTHS, freq="MS")
t = np.arange(N_MONTHS)
month_of_year = dates.month

# ---------------------------------------------------------------------------
# 1. Target headcount trajectory: long-run growth + two shocks
# ---------------------------------------------------------------------------
base_growth = 120 + 0.85 * t + 0.0035 * t**2  # gentle accelerating growth

# 2009 recession dip (months ~36-52), 2020 pandemic dip (months ~168-176),
# 2021-22 rehire surge (months ~180-204)
recession_2009 = -48 * np.exp(-0.5 * ((t - 44) / 6) ** 2)
pandemic_2020 = -42 * np.exp(-0.5 * ((t - 172) / 4.5) ** 2)
rehire_surge = 30 * np.exp(-0.5 * ((t - 196) / 10) ** 2)

target_headcount = base_growth + recession_2009 + pandemic_2020 + rehire_surge
target_headcount = np.clip(target_headcount, 60, None)

# ---------------------------------------------------------------------------
# 2. Voluntary / involuntary attrition rate paths (monthly, as a fraction)
# ---------------------------------------------------------------------------
# Seasonality: voluntary attrition higher in Jan (new-year job search) and
# June (post-bonus/mid-year), lower around Nov/Dec.
seasonal_vol = 0.0035 * np.cos(2 * np.pi * (month_of_year - 1) / 12) + \
               0.0020 * np.cos(2 * np.pi * (month_of_year - 6) / 12)

secular_vol = 0.0078 + 0.00004 * t  # slow secular rise in baseline turnover

# "Great Resignation" bump centered ~2021-22 (months ~184-204)
great_resignation = 0.008 * np.exp(-0.5 * ((t - 194) / 10) ** 2)

vol_rate = np.clip(secular_vol + seasonal_vol + great_resignation, 0.002, None)

# Involuntary terminations: small and fairly flat, elevated during the two shocks
# (layoffs), which is also what drives the visible headcount contractions.
invol_rate = 0.0018 + 0.014 * np.exp(-0.5 * ((t - 44) / 6) ** 2) + \
             0.016 * np.exp(-0.5 * ((t - 172) / 4.5) ** 2)

# ---------------------------------------------------------------------------
# 3. Simulate month by month so Headcount stays internally consistent
# ---------------------------------------------------------------------------
headcount = np.zeros(N_MONTHS, dtype=int)
hires = np.zeros(N_MONTHS, dtype=int)
vol_terms = np.zeros(N_MONTHS, dtype=int)
invol_terms = np.zeros(N_MONTHS, dtype=int)

prev_hc = int(round(target_headcount[0])) - rng.integers(0, 5)

for i in range(N_MONTHS):
    vt = rng.poisson(max(prev_hc, 1) * vol_rate[i])
    it = rng.poisson(max(prev_hc, 1) * invol_rate[i])
    remaining = prev_hc - vt - it
    needed_hires = int(round(target_headcount[i] - remaining))
    h = max(0, needed_hires + rng.integers(-2, 3))

    hc = remaining + h
    headcount[i] = hc
    hires[i] = h
    vol_terms[i] = vt
    invol_terms[i] = it
    prev_hc = hc

total_terms = vol_terms + invol_terms
prev_headcount = np.concatenate(([headcount[0] - hires[0] + total_terms[0]], headcount[:-1]))
avg_hc = (prev_headcount + headcount) / 2.0
attrition_rate = np.round(total_terms / np.maximum(avg_hc, 1) * 100, 2)

# ---------------------------------------------------------------------------
# 4. Engagement & satisfaction: inversely related to attrition, slow drift, noise
# ---------------------------------------------------------------------------
z_attr = (attrition_rate - attrition_rate.mean()) / attrition_rate.std()
engagement = 3.6 - 0.18 * z_attr + 0.10 * np.sin(2 * np.pi * (month_of_year - 3) / 12)
engagement += rng.normal(0, 0.12, N_MONTHS)
engagement = np.clip(engagement, 1.0, 5.0).round(2)

satisfaction = 3.5 - 0.22 * z_attr + 0.08 * np.sin(2 * np.pi * (month_of_year - 4) / 12)
satisfaction += rng.normal(0, 0.13, N_MONTHS)
satisfaction = np.clip(satisfaction, 1.0, 5.0).round(2)

# ---------------------------------------------------------------------------
# 5. Open requisitions: proxy for near-term hiring pipeline pressure
# ---------------------------------------------------------------------------
next_hires = np.concatenate((hires[1:], [hires[-1]]))
open_reqs = np.clip((0.6 * hires + 0.6 * next_hires + rng.normal(0, 2, N_MONTHS)).round(), 0, None).astype(int)

# ---------------------------------------------------------------------------
# 6. Assemble & save
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "Date": dates.strftime("%Y-%m-%d"),
    "Headcount": headcount,
    "Hires": hires,
    "Voluntary_Terminations": vol_terms,
    "Involuntary_Terminations": invol_terms,
    "Total_Terminations": total_terms,
    "Attrition_Rate_Pct": attrition_rate,
    "Avg_Engagement_Score": engagement,
    "Avg_Satisfaction_Score": satisfaction,
    "Open_Requisitions": open_reqs,
})

out_path = "people_analytics_monthly.csv"
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} rows to {out_path}")
print(df.head())
print(df.tail())
print(df.describe())
