# People Analytics Forecasting: Headcount & Attrition with Moving Averages, Exponential Smoothing, and ARIMA

Forecasting monthly workforce headcount and attrition rate for a company using classical time series methods, and comparing them head-to-head on the same data. This README doubles as the project plan.

## 1. Does this already exist?

Searched GitHub/web for projects at the intersection of time series forecasting and people analytics. The short answer: not really, which is what makes this a good project to build.

What exists instead falls into two separate buckets that never overlap:

- **People analytics repos** (e.g. `nilakshiGogoi/Employee-turnover-Analytics`, `rohitkrishnanm/Employee-Turnover-Analytics`, the various `IBM-HR-Analytics-Employee-Attrition` forks) all frame attrition as a **classification problem** on cross-sectional data — one row per employee, predict "will this person leave," using logistic regression / random forest / XGBoost. None of them treat attrition or headcount as a time-indexed series.
- **Time series forecasting repos** (the many `arima-forecasting` / `time-series-forecasting` GitHub Topics results) almost universally use stock prices, sales, or sensor data as the example domain. Moving average, exponential smoothing, and ARIMA tutorials are common — just never on HR data.
- The closest thing to real time-series HR data is the UCI "Absenteeism at Work" dataset (`ytnvj2/employee-absenteeism`, `UBC-MDS/Absenteeism_at_Work`), but it's 740 individual absence *events* for 36 workers with no continuous year field, not a clean date-indexed series — it's typically used for regression/clustering on absence hours, not forecasting.

So a project that takes classic forecasting techniques (moving average → exponential smoothing → ARIMA/SARIMA) and applies them to a genuinely time-indexed people analytics series (monthly headcount / attrition) fills a real gap rather than duplicating an existing repo.

## 2. Project concept

**Framing:** "How many people will we have, and how many will we lose, over the next 3-12 months?" — the kind of question a People Analytics or Workforce Planning team answers to size hiring budgets and recruiting capacity.

**Target variables (pick one as primary, the other as a stretch/secondary series):**
- `Headcount` — active employee count at month end
- `Attrition_Rate_Pct` — monthly attrition rate (more stationary, arguably the more interesting forecasting target since Headcount has a strong trend)

**Core comparison:** implement all three methods the user already knows on the same train/test split and the same evaluation metric, so the project's payoff is a clear, defensible answer to "which method actually works best here, and why":
1. Moving average (simple + weighted) as the naive baseline
2. Exponential smoothing (SES → Holt's linear trend → Holt-Winters seasonal)
3. ARIMA / SARIMA (manually identified via ACF/PACF + `auto_arima`)

Because the dataset has trend, yearly seasonality, and two shock periods, moving average should visibly underperform, Holt-Winters should handle trend+seasonality well, and SARIMA should be competitive or best — a natural narrative for a README or blog writeup.

## 3. Dataset

**File:** `data/people_analytics_monthly.csv` — 240 rows, monthly, Jan 2006 – Dec 2025, 10 columns, no missing values.

No existing public dataset was a good fit as-is: real people analytics datasets on GitHub/Kaggle (IBM HR Attrition, "HRDataset_v14", the UCI Absenteeism set) are all cross-sectional — one row per employee, not per time period — so they'd need non-trivial, hard-to-verify reconstruction to become a time series. Rather than force-fit one of those, `data/generate_dataset.py` builds a monthly panel from scratch with realistic structure: a growth trend, yearly seasonality in hiring/attrition, a 2009-recession dip, a 2020-pandemic dip, and a 2021-22 "Great Resignation" attrition spike, all built from a single seeded random generator (`seed=42`) so it's 100% reproducible — re-run the script and you get byte-identical output. It also means you can dial the parameters (growth rate, seasonality strength, shock timing) up or down if you want a different flavor of the same problem.

**Data dictionary:**

| Column | Description |
|---|---|
| `Date` | First of month, `YYYY-MM-DD` |
| `Headcount` | Active employees at month end |
| `Hires` | New hires during the month |
| `Voluntary_Terminations` | Voluntary exits during the month |
| `Involuntary_Terminations` | Layoffs / performance-related exits during the month |
| `Total_Terminations` | Voluntary + Involuntary |
| `Attrition_Rate_Pct` | Total_Terminations / avg(prior, current Headcount) × 100 |
| `Avg_Engagement_Score` | Mean engagement survey score that month, 1-5 scale |
| `Avg_Satisfaction_Score` | Mean employee satisfaction score that month, 1-5 scale |
| `Open_Requisitions` | Open job requisitions at month end |

Sanity-checked: no negative values, headcount = prior headcount + hires − terminations every row, dates are complete and consecutive, mean annualized attrition ≈ 20% (in line with typical real-world benchmarks, spiking during the simulated 2021-22 period).

**If you'd rather use real data:** the closest usable substitute is [`pouyasattari/HR-Dataset-Analysis`](https://github.com/pouyasattari/HR-Dataset-Analysis) (the well-known "HRDataset_v14", ~252 fictional-company employees with real hire/termination dates spanning ~2006-2018). It's not a time series as-is, but grouping by month on `DateofHire` and `DateofTermination` would derive a real monthly hires/terminations/headcount series with the same shape as this synthetic one — a natural "swap in real data" extension once the pipeline works. Other reference datasets worth knowing about: [IBM HR Analytics Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (cross-sectional, 1,470 rows) and the [UCI Absenteeism at Work](https://github.com/ytnvj2/employee-absenteeism) set.

## 4. Methodology / roadmap

**Phase 1 — EDA & stationarity**
Plot the raw series, seasonal decomposition (`statsmodels.tsa.seasonal_decompose`), Augmented Dickey-Fuller test, ACF/PACF plots, note the trend/seasonality/shock structure visible in the data.

**Phase 2 — Baselines**
Naive forecast (last value), simple moving average (window sweep, e.g. 3/6/12 month), weighted moving average. These are the floor every later model needs to beat.

**Phase 3 — Exponential smoothing**
Simple exponential smoothing → Holt's linear trend method → Holt-Winters seasonal (additive and multiplicative), via `statsmodels.tsa.holtwinters.ExponentialSmoothing`.

**Phase 4 — ARIMA / SARIMA**
Difference to stationarity (guided by the ADF test from Phase 1), identify (p,d,q) from ACF/PACF, fit with `statsmodels.tsa.arima.model.ARIMA`, then a seasonal `SARIMAX` (seasonal period = 12). Cross-check manual order selection against `pmdarima.auto_arima`. Stretch: refit SARIMAX with `Avg_Engagement_Score` or `Open_Requisitions` as an exogenous regressor.

**Phase 5 — Evaluation**
Rolling-origin (walk-forward) backtesting rather than a single train/test split, since 240 points is small enough to make backtesting cheap. Report MAE, RMSE, and MAPE for every method side by side, plus a plot of forecasts vs. actuals over the last 24 held-out months.

**Phase 6 — Stretch goals**
Prophet or a simple LSTM as an outside comparison; anomaly/changepoint detection to auto-flag the 2009/2020 shocks; a small Streamlit app to interactively forecast N months ahead; GitHub Actions to re-run the notebook on push.

## 5. Suggested repo structure

```
ARIMA/
├── README.md
├── requirements.txt
├── data/
│   ├── generate_dataset.py
│   └── people_analytics_monthly.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_moving_average.ipynb
│   ├── 03_exponential_smoothing.ipynb
│   └── 04_arima_sarima.ipynb
├── src/
│   ├── metrics.py          # MAE / RMSE / MAPE, backtesting helper
│   └── plotting.py         # shared forecast-vs-actual plot helper
└── results/
    └── model_comparison.csv
```

## 6. Next steps checklist

- [ ] `git init`, push `README.md` + `data/` as the first commit
- [ ] Phase 1 EDA notebook
- [ ] Baselines (Phase 2)
- [ ] Exponential smoothing (Phase 3)
- [ ] ARIMA/SARIMA (Phase 4)
- [ ] Backtesting + comparison table (Phase 5)
- [ ] Write up findings in README ("which model won, and why")
- [ ] Pick a stretch goal (Phase 6) if there's time left

## Tech stack

Python, pandas, numpy, statsmodels, pmdarima, scikit-learn (metrics), matplotlib/plotly, jupyter. See `requirements.txt`.
