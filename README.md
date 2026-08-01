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
- `Attrition_Rate_Pct` — monthly attrition rate

**Core comparison:** implement all three method families on the same train/test split and the same evaluation metric, so the project's payoff is a clear, defensible answer to "which method actually works best here, and why":
1. Moving average (simple + weighted) as the naive baseline
2. Exponential smoothing (SES → Holt's linear trend → Holt-Winters seasonal)
3. AR, MA, ARIMA, and SARIMA, built up one piece at a time

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

## 4. Key concepts, explained simply

This project leans on a handful of core time-series ideas. If you're new to this, read this section before the notebooks — everything below was worked out interactively while building Phase 1 (`notebooks/01_eda.ipynb`) and Phase 2 (`notebooks/02_moving_average.ipynb`), so it doubles as a walkthrough of *why* those notebooks do what they do.

### Stationarity

A series is **stationary** if its statistical behavior doesn't depend on *when* you look at it: constant mean, constant variance, and no seasonality (no systematic dependence on calendar position). Most forecasting theory (AR, MA, ARIMA) is built assuming stationarity — fit those models directly on a non-stationary series and the math becomes unreliable.

**Concrete example:** an ice cream shop's sales are $2k every January and $15k every July, identically, every year, forever — no growth, just a repeating wave. Is that stationary? No — the average depends on which month you pick ($2k vs. $15k), so "constant mean" is already broken. Seasonality alone is enough to disqualify a series from being stationary, regardless of whether it has a long-term trend on top.

### ADF and KPSS: two different questions, not two versions of the same test

- **ADF** (Augmented Dickey-Fuller): H₀ = "this series has a unit root" (a shock to it persists *forever*, like a random walk that never comes back). Rejecting H₀ (p < 0.05) → no unit root → often reported as "stationary."
- **KPSS**: H₀ = "this series is stationary" — the opposite null. Rejecting H₀ (p < 0.05) → non-stationary.

Run both, because they can disagree, and the disagreement is informative. If ADF says stationary but KPSS says not, that's the signature of a **trend-stationary** series: noisy and mean-reverting month-to-month, but riding on a slow-moving deterministic drift (exactly what happened with `Attrition_Rate_Pct` here).

**Important limitation:** neither test checks for seasonality. They only test for a *unit root* (a permanent, non-decaying shock). A series can be perfectly, obviously seasonal (like the ice cream example) and still pass both tests, because "the mean cycles predictably with the calendar" and "the series has a unit root" are different failure modes. Seasonality is diagnosed separately — by seasonal decomposition or by looking for repeated spikes in the ACF at multiples of the seasonal period (see below) — not by ADF/KPSS.

**Why this connects to "shocks":** a unit root literally means shocks are permanent. In this dataset, `Headcount` never "bounces back" after the 2009 recession — it just keeps growing from the new, lower level — which is exactly why it tests as non-stationary. A stationary series, by contrast, absorbs a shock and reverts back toward its usual range over time.

### Seasonal decomposition — what `seasonal_decompose(df[col], model="additive", period=12)` actually does

Splits `observed = trend + seasonal + resid` in four mechanical steps:

1. **Trend**: a centered moving average with a 12-month window (`period=12`, because the data is monthly and a year = 12 rows) smooths out both noise and the yearly wiggle, leaving just the slow-moving level.
2. **Detrend**: subtract that trend from the observed series. What's left is seasonality and noise, still tangled together.
3. **Isolate the seasonal shape**: average every January's detrended value together (all 20 Januaries → 1 number), every February's, etc. — 12 numbers total. Averaging is what separates signal from noise: a real seasonal push shows up in every year so it survives averaging, while random noise is independently positive or negative each year so it mostly cancels out.
4. **Stamp that identical 12-number shape onto every year** in the dataset. This is why the seasonal panel looks like a perfectly repeating sawtooth with unchanging amplitude across 20 years — it's mechanically built from only 12 numbers on a loop, not something that's free to vary year to year.

`resid = observed − trend − seasonal` — whatever's left. Ideally it looks like plain noise; in `Headcount`'s case it isn't quite, showing clear spikes right at 2009 and 2020, because the moving-average trend line is symmetric (uses data from both before *and* after each point) and so lags behind sudden real shocks — the gap between what really happened and what the smoothed trend "knew" gets dumped into the residual.

### ACF and PACF — reading the bars, and what they're for

Both measure correlation between a series and a lagged (shifted-back-in-time) version of itself, using the *entire* dataset for every lag — "lag 12" doesn't mean "only look at year 1," it means "compare every month in the whole 20-year range to the same month one year earlier." They're plotted out to lag 36 (3 years) here, not because the data before that is ignored, but because correlation estimates get noisy past a few years (fewer non-overlapping pairs left to compute them from).

- **ACF** (autocorrelation): correlation at each lag directly.
- **PACF** (partial autocorrelation): correlation at each lag *after* removing the effect already explained by shorter lags.

The classic reading: **ACF decays gradually over many lags + PACF cuts off sharply after lag p → AR(p)**. A genuinely seasonal series should also show repeated bumps at lag 12, 24, 36 (multiples of the period) — a single blip at lag 12 alone is weaker evidence than a repeating echo at every multiple.

### AR, MA, ARIMA, SARIMA — and a naming trap

All of these are fundamentally driven by random shocks (`E(t)`); they differ in **how long a shock's influence lasts**:

- **AR(p)** — `Z(t) = φ₁·Z(t-1) + ... + φₚ·Z(t-p) + E(t)`. Today depends on past *values*, which themselves already embed all earlier shocks. A shock's influence never fully vanishes — it decays geometrically, smaller each period, but technically lingers forever (as long as the process is stationary; if it doesn't decay at all, that's the unit-root case again).
- **MA(q)** — `Z(t) = E(t) + θ₁·E(t-1) + ... + θq·E(t-q)`. Today depends directly on a *fixed, finite* list of the last `q` shocks. Past the q-th lag, a shock's influence isn't small — it's exactly zero. A hard cutoff, not a decay.
- **ARMA(p,q)** combines both, e.g. `Z(t) = φ·Z(t-1) + θ·E(t-1) + E(t)` for ARMA(1,1).
- **ARIMA(p,d,q)** — the "I" (Integrated) just means "differenced `d` times first." Difference the raw series until it's stationary, *then* fit ARMA(p,q) on the differenced series. ARIMA doesn't replace AR/MA, it's what makes fitting them valid in the first place.
- **SARIMA** — adds a *seasonal* difference and seasonal AR/MA terms on top, because regular (non-seasonal) differencing removes trend but does nothing about a repeating 12-month wave. Given both `Headcount` and `Attrition_Rate_Pct` show real seasonality here, SARIMA rather than plain ARIMA is the realistic target for this dataset.

**Naming trap:** "moving average" as a simple forecasting baseline (Phase 2 below — literally averaging the last N actual values to predict the next one) and the "MA" inside ARMA/ARIMA (modeling today's value from past *forecast errors*, not past actual values) are unrelated ideas that happen to share a name. Different math, different purpose — don't conflate them.

### Data leakage, and why every forecast column gets `.shift(1)`

A "forecast" for May built from May's own actual value isn't a forecast — it's May compared to itself, which scores a suspicious, meaningless zero error. `.shift(1)` is what prevents that: it guarantees the number sitting in May's forecast slot is actually April's real value, i.e. only information that would genuinely have existed *before* May happened. This is the general rule against **data leakage** — a prediction for time `t` may only be built from data up through `t-1` — and it's why every baseline in `02_moving_average.ipynb` is built on a shifted column, never the raw one.

### Two unrelated "which past matters" questions: PACF vs. train/test split

Easy to conflate, genuinely separate: **PACF** decides the *structure* of an equation — how many lag terms to include (e.g. "use the last 2 months as predictors"). The **train/test split** decides which *time periods* a model is allowed to see while fitting vs. which are held back to honestly grade it on afterward (216 months to train on, the most recent 24 held out as an exam it can't study from). You'd need the train/test split regardless of how many lags your model uses, and PACF would recommend the same lag count regardless of whether you hold out any test data at all. Independent decisions.

### Trailing vs. centered moving averages — CMA (Phase 1) vs. SMA (Phase 2)

Two different tools that both get called "moving average," easy to conflate: Phase 1's decomposition used a **centered** moving average (data from both *before and after* each point) to describe a smooth trend line after the fact — useless for real forecasting, since on the day you're actually predicting, "after" data doesn't exist yet. Phase 2's SMA is **trailing** (`.shift(1).rolling(w).mean()`) — it only ever looks backward, which is what makes it usable as an actual forecast rather than just a descriptive summary.

### How weights work in SMA vs. WMA

SMA ("simple") has no differentiated weighting at all — every month inside the window counts identically, `1/w` each, and everything outside the window counts `0`. WMA assigns weights `[1, 2, ..., w]` (normalized to sum to 1), so the most recent month in the window counts the most and the oldest counts the least — literally `np.dot(values, weights) / weights.sum()`. Built at matching window sizes on purpose (3/6/12 for both), so that comparing `SMA(w)` to `WMA(w)` isolates exactly one variable — equal weighting vs. recency weighting — instead of mixing that question up with "is this window size any good." Result on this data, cleanly consistent at every window: WMA beats SMA on trending `Headcount` every time (recent = more accurate when trending up), SMA beats WMA on noisy `Attrition_Rate_Pct` every time (no direction to weight toward, so equal weighting cancels more noise).

### MAE vs. RMSE — why `sqrt(mean(x²))` isn't `mean(|x|)`

`sqrt(x²) = |x|` for any single number, so it's natural to expect RMSE and MAE to agree. They don't, because the averaging happens at a different point: RMSE squares every error, **averages the squared values together**, then takes one square root at the end; MAE takes the absolute value of every error and averages those directly. Example — errors `[1, 1, 1, 1, 5]`: MAE = (1+1+1+1+5)/5 = **1.8**. RMSE = √((1+1+1+1+25)/5) = √5.8 = **2.41**. That one error of 5 got squared to 25 — a hugely disproportionate contribution — and once it's mixed into the average with the others, the final square root can only shrink the whole mixture back down, not undo the lopsided contribution from that one term. Rule: **RMSE ≥ MAE always**, equal only when every error is exactly the same size. A bigger RMSE-vs-MAE gap means a few large misses are driving the error, not uniformly mediocre ones.

### Rolling (one-step) vs. static (multi-step) evaluation

Two very different tests of the same method. **Rolling** (Phase 2 cells 3-8): at every point in the test period, forecast just the next month using real, up-to-date actual data — the method gets "refed" fresh information every single month, so even a lagging method can slowly climb along with a real trend. **Static** (Phase 2 cell 9): fit or compute *once*, using only training data, then hold that forecast flat/frozen for the entire test horizon with no updates at all — simulating "build a plan today, never revisit it." On trending `Headcount`, this is brutal: rolling `SMA(12)` scored RMSE 13.39 (still climbing, just lagged); the same method held static scored RMSE 34.43 (2.5× worse — a single frozen number sitting motionless for 2 years while reality climbed 60+ points away from it). Exponential smoothing and ARIMA/SARIMA models (Phases 3-4) are properly *fit* rather than cheaply recomputed, so by convention they're evaluated the static way — fit once on training data, forecast the full horizon in one shot — making Phase 2's static baseline (not the rolling numbers) the fair comparison point for those later phases.

### Why a moving average can never capture trend

Not a tuning problem — a structural one. Averaging discards the *order* values arrived in, keeping only their center: the flat sequence `15.5, 15.5, 15.5, 15.5, 15.5, 15.5` and the steadily climbing sequence `13, 14, 15, 16, 17, 18` both average to exactly **15.5** — indistinguishable to a moving average, even though one is going nowhere and the other should obviously be forecast to hit 19 next. Noticing "this is clearly rising, so it'll keep rising" requires estimating a *slope* (rate of change per month) and projecting it forward — a fundamentally different operation from "add up recent values and divide," which a plain average has no mechanism to do, no matter which window size is chosen. Holt's method (Phase 3) fixes this directly by estimating a level **and** a trend, then forecasting `level + h × trend` — literally the "notice the slope, keep going" logic done properly.

### Exponential smoothing — how the level updates, one step at a time

`SimpleExpSmoothing.fit(optimized=True)` doesn't hand-pick alpha (the smoothing constant) — it numerically searches for whichever alpha minimizes forecast error *on the training data itself*, the same kind of optimization a regression uses to pick its coefficients, just for one parameter instead of many. Once fit, the update rule is: `level(t) = alpha × actual(t) + (1 − alpha) × level(t−1)` — each new level is built only from this month's real value and the previous level, nothing further back gets re-touched. Concretely, with alpha = 0.3 and a starting level of 100: if this month's actual comes in at 130, the new level becomes `0.3 × 130 + 0.7 × 100 = 109` — nudged toward the new observation, not jumped to it. A higher alpha nudges harder (more trust in the newest point, faster-reacting, noisier); a lower alpha barely moves (smoother, slower to react). SES forecasts by freezing that final level and repeating it flat for every future month — which is exactly why it has the same trend-blindness as a moving average, just computed differently. That blindness is the point of including SES here: it proves "exponential smoothing" per se isn't what fixes trend-blindness, the *trend term* Holt adds next is.

On this data, SES's fitted alpha differs sharply by target: `Headcount` alpha = **1.0000** (mathematically collapses to the naive forecast — trust only the most recent value, ignore everything older, because Headcount barely wobbles month to month once you're inside a trend, so "the most recent point" is already the best available estimate), `Attrition_Rate_Pct` alpha = **0.1753** (barely react at all — because Attrition is noisy, over-trusting any single recent point would just chase noise, so the optimizer settled on relying mostly on a slow-moving historical average instead).

### Holt's linear trend method — adding a second moving part

Holt adds a second smoothed quantity, the trend, updated the same recursive way as the level: `trend(t) = beta × (level(t) − level(t−1)) + (1 − beta) × trend(t−1)`. The forecast for `h` months ahead is then `level + h × trend` — a straight line extrapolated forward from the last fitted level and slope. On `Headcount`, alpha = beta = 0.6596 and Holt clearly beats SES: RMSE 14.32 vs. 30.03, MAE 11.73 vs. 25.42 — adding the ability to keep climbing instead of freezing flat matters a lot on a genuinely trending series. On `Attrition_Rate_Pct`, beta fit to **0.0000** — the optimizer decided a trend term adds nothing (there's no real slope to a series that oscillates without ever committing to a direction), and with the trend term switched off, Holt collapses back to being SES with slightly different rounding, which is exactly why Holt only barely underperforms SES there (RMSE 0.685 vs. 0.666) rather than being dramatically better or worse — it isn't really doing anything different.

### Holt-Winters — why gamma fit to 0 even though seasonality is real

Adding `trend="add", seasonal="add", seasonal_periods=12` gives Holt-Winters a third smoothed quantity, gamma, meant to track a repeating 12-month shape the same way alpha tracks the level and beta tracks the trend. On both targets here, gamma fit to **0.0000** — surprising at first, since Phase 1's decomposition confirmed both series really do have yearly seasonality. The resolution: gamma near 0 doesn't mean "no seasonality exists," it means "the seasonal component isn't worth actively re-estimating every period." `initialization_method="estimated"` already bakes a reasonable starting seasonal shape into the model from the training data before optimization even begins — if that starting shape is already a decent fit, the optimizer has little incentive to keep adjusting it, so gamma gets pushed toward 0 and the initial seasonal estimate rides along mostly frozen for the whole forecast. That's a difference from beta on Attrition, where beta = 0 meant "there really is no trend to model" — here gamma = 0 means "the seasonal pattern is real, but already captured once at initialization; continuously re-fitting it during the optimization isn't earning its keep."

For `Headcount`, this is mostly harmless — the frozen seasonal shape from initialization is small and the trend term is doing the heavy lifting anyway, so Holt-Winters (RMSE 13.57) still edges out plain Holt (RMSE 14.32) with a modest assist from the seasonal shape. For `Attrition_Rate_Pct`, it's actively harmful: RMSE goes from SES's 0.666 up to Holt-Winters's **0.830** — the worst of all three methods. Here's why gamma freezing at 0 doesn't save it: the frozen seasonal shape is *fixed*, but the model still forecasts `level + seasonal[month]` — a rigid extra offset it can never adjust away, even if the exact position of the real spikes and dips shifts slightly year to year, which they do on a genuinely noisy series. So Holt-Winters is worse not because it's actively re-chasing noise (gamma = 0 means it isn't), but because it committed to one specific seasonal offset pattern up front and now can't undo it when reality doesn't line up with that exact shape.

### What this looks like when overfitting happens — a concrete example from this data

Overfitting: a model spends its limited "fitting power" matching patterns in the training data that don't generalize, and ends up *worse* on new data than a simpler model that didn't try. Attrition's Holt-Winters result is a clean, visual example. It has more moving parts than SES (level + trend + season vs. just level), so it can match more of what it saw during training. The seasonal shape it locked onto is real in a weak statistical sense — Phase 1 confirmed genuine yearly seasonality — but it's a small effect sitting on top of a lot of month-to-month noise, and the shape estimated from 18 years of training data doesn't line up precisely with where the swings land in the 24 held-out test months. Plotted out (`07_exp_smoothing_forecasts.png`), Holt-Winters is visibly the most "active" line — wiggling up and down chasing that seasonal cycle — while SES just sits flat. The wiggle looks more sophisticated, like it's trying harder to track the real swings (which whip between roughly 1.0% and 3.6%). But the numbers say the opposite: RMSE went from 0.666 (SES, flat) to 0.830 (Holt-Winters, wiggling) — worse, not better. The general tell to watch for going forward: whenever a fancier, more-parameterized model scores worse than a simpler one *on the test set* (not the training set), that's the signature to check for overfitting — especially on a series like Attrition where noise dominates whatever real signal exists.

### Fit-once-and-freeze vs. rolling: why Phase 3 changes the evaluation rule

Phase 2's naive/SMA/WMA are cheap arithmetic, recomputed fresh every month using real, up-to-date test-period data (rolling one-step-ahead). Phase 3's models are properly *fit* — parameters like alpha/beta/gamma are estimated once via numerical optimization on the training data, then `.forecast(24)` projects the entire test horizon in a single shot with no updates. That's the same "commit once, never revisit" setup as Phase 2's static demo, not its rolling cells — which makes Phase 2's **static** flat-SMA(12) baseline (Headcount RMSE 34.43, Attrition RMSE 0.665) the fair number for Phase 3 to beat, not the rolling numbers.

The final scorecard: on `Headcount`, Holt-Winters (RMSE 13.57) clearly beats the static baseline (34.43) — a real, decisive win, exponential smoothing earns its complexity here. On `Attrition_Rate_Pct`, the best method (SES, RMSE 0.666) is statistically tied with the static baseline (0.665) — essentially no improvement at all. Consistent with everything above: when there's a real trend/seasonal signal to exploit (Headcount), more sophisticated methods pay off; when a series is dominated by noise (Attrition), no amount of extra modeling machinery manufactures signal that isn't there, and the honest, useful lesson is that a dumb flat average was already about as good as it gets.

## 5. Methodology / roadmap

**Phase 1 — EDA & stationarity** (`notebooks/01_eda.ipynb`, done)
Raw series plots, seasonal decomposition, ADF + KPSS stationarity tests, ACF/PACF plots. Findings: both `Headcount` and `Attrition_Rate_Pct` are non-stationary (Headcount from trend + seasonality; Attrition from seasonality + a slow trend-stationary drift that KPSS catches and ADF misses). PACF cuts off around lag 2-3 for both, ACF decays gradually — points toward AR(2)/AR(3) as a starting order for Phase 4.

**Phase 2 — Baselines** (`notebooks/02_moving_average.ipynb`, done)
Naive forecast, simple moving average (3/6/12 month, matching quarterly/half-yearly/annual reporting cadences), weighted moving average (same 3/6/12 windows, recency-weighted). Evaluated two ways: rolling one-step-ahead, and a static multi-step demo. Findings: on trending `Headcount`, less smoothing always wins — Naive beats every SMA/WMA outright, and WMA beats SMA at every matching window (recency-weighting helps when there's a real direction to weight toward). On noisy `Attrition_Rate_Pct`, it inverts — SMA(12) wins overall, and SMA beats WMA at every matching window (no direction to weight toward, so equal weighting cancels more noise). The static demo confirmed this isn't a tuning issue: even the best moving average, frozen and never updated, cannot represent a trend at all (RMSE 13.39 rolling → 34.43 static on Headcount) — a structural ceiling, not something a better window size could fix.

**Phase 3 — Exponential smoothing** (`notebooks/03_exponential_smoothing.ipynb`, done)
Simple exponential smoothing → Holt's linear trend method → Holt-Winters seasonal, via `statsmodels.tsa.holtwinters`. Evaluated in fit-once-forecast-24-months-ahead mode (not rolling — see "Fit-once-and-freeze vs. rolling" above), against Phase 2's static baseline. Findings: on `Headcount`, each added capability helped — SES (alpha=1.0, collapses to naive) RMSE 30.03 → Holt (alpha=beta=0.66, adds trend) RMSE 14.32 → Holt-Winters (adds a small seasonal assist) RMSE 13.57 — comfortably beating Phase 2's static baseline of 34.43. On `Attrition_Rate_Pct`, the opposite: SES (alpha=0.175, mostly ignores recent noise) RMSE 0.666 was already about as good as Phase 2's static baseline (0.665), Holt's trend term fit to beta=0 (no real slope to model) barely changed anything, and Holt-Winters's frozen seasonal offset (gamma=0) actively hurt, RMSE 0.830 — a clean example of overfitting: more model complexity chasing a real-but-small seasonal signal, scoring worse on held-out data than the simplest method.

**Phase 4 — AR → MA → ARIMA → SARIMA** (`notebooks/04_arima_sarima.ipynb`, done)
Built up one piece at a time rather than jumping straight to the combined model, with (p, d, q) chosen from PACF/ACF cutoffs computed fresh on the training split (not read off Phase 1's plots — order selection is a modeling decision and shouldn't be informed by data used for anything else): `Headcount` → (p,d,q)=(3,1,5), `Attrition_Rate_Pct` → (3,0,6).
1. Pure **AR(p)** (`ARIMA(p, d, 0)`).
2. Pure **MA(q)** (`ARIMA(0, d, q)`).
3. Combined **ARIMA(p, d, q)** — does combining actually help, or was one component doing all the work?
4. **SARIMAX** with seasonal order (1,1,1,12), since Phase 1 confirmed real seasonality in both targets.
5. Cross-check against `pmdarima.auto_arima`'s own automated search.

Findings, evaluated the same fit-once-forecast-24-months-ahead way as Phase 3: on `Headcount`, combining didn't help — pure AR(3) (RMSE 23.54) actually beat the combined ARIMA(3,1,5) (RMSE 25.83), which beat pure MA(5) (RMSE 28.89) — extra MA terms added complexity without adding accuracy. The seasonal layer, though, was decisive: SARIMAX RMSE **3.02**, by far the best result across every phase so far (vs. Phase 3's best of 13.57 and Phase 2's static baseline of 34.43). `auto_arima` picked a smaller seasonal order here and landed at RMSE 10.79 — worse than the manual pick on both AIC and test RMSE, because its search didn't apply seasonal differencing (D=0) the way the manual (1,1,1,12) order did.

On `Attrition_Rate_Pct`, the opposite pattern from Headcount: combining AR+MA *did* help (ARIMA RMSE 0.671, better than AR's 0.713 or MA's 0.727), but the seasonal layer hurt again — SARIMAX RMSE 0.798, worse than plain ARIMA — the same overfitting signature as Holt-Winters in Phase 3 (a real-but-small seasonal effect, over-fit to training data, actively wrong on test data). `auto_arima` won outright here (RMSE **0.643**, best of all Phase 4 methods and a first, if marginal, improvement over Phase 2/3's ~0.665 static-baseline plateau) by picking a deliberately smaller order (0,1,1)x(0,0,1,12) than the manual pick — parsimony winning over the manually-fit model's larger order on a noisy series.

**Phase 5 — Evaluation**
Rolling-origin (walk-forward) backtesting rather than a single train/test split, since 240 points is small enough to make backtesting cheap. Report MAE, RMSE, and MAPE for every method side by side (naive, moving average, SES/Holt/Holt-Winters, AR, MA, ARIMA, SARIMA), plus a plot of forecasts vs. actuals over the last 24 held-out months.

**Phase 6 — Stretch goals**
Prophet or a simple LSTM as an outside comparison; anomaly/changepoint detection to auto-flag the 2009/2020 shocks; a small Streamlit app to interactively forecast N months ahead; GitHub Actions to re-run the notebook on push.

## 6. Repo structure

```
ARIMA/
├── README.md
├── requirements.txt
├── data/
│   ├── generate_dataset.py
│   └── people_analytics_monthly.csv
├── notebooks/
│   ├── 01_eda.ipynb                    # done
│   ├── 02_moving_average.ipynb         # done
│   ├── 03_exponential_smoothing.ipynb  # done
│   └── 04_arima_sarima.ipynb           # done
├── src/
│   └── metrics.py           # MAE / RMSE / MAPE, comparison_table helper
└── results/
    ├── figures/              # PNGs saved from each notebook
    └── metrics_phase*.csv    # per-phase comparison tables
```

## 7. Next steps checklist

- [x] `git init`, push `README.md` + `data/` as the first commit
- [x] Phase 1 EDA notebook
- [x] Baselines (Phase 2)
- [x] Exponential smoothing (Phase 3)
- [x] AR → MA → ARIMA → SARIMA (Phase 4)
- [ ] Backtesting + comparison table (Phase 5)
- [ ] Write up findings in README ("which model won, and why")
- [ ] Pick a stretch goal (Phase 6) if there's time left

## Tech stack

Python, pandas, numpy, statsmodels, pmdarima, scikit-learn (metrics), matplotlib/plotly, jupyter. See `requirements.txt`.
