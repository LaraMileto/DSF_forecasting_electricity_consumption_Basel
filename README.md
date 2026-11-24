# DSF_forecasting_electricity_consumption_Basel
# Forecasting Daily Electricity Consumption in Basel-Stadt

**University of St. Gallen | Data Science Fundamentals | Group Project**

## 1. Project Overview

Electricity consumption varies over time due to factors such as weather conditions, seasonal patterns, and differences between weekdays and weekends. Accurately forecasting electricity demand is important for electricity procurement and operational planning.

This project investigates whether historical electricity consumption, weather forecasts, and calendar information can be used to predict **next-day electricity consumption in the canton of Basel-Stadt, Switzerland**.

We develop and evaluate machine learning models using Python, with a particular focus on generating forecasts that would realistically be available to an electricity procurement decision-maker.

### Research Question

**How accurately can we predict the total electricity consumption (MWh/day) in Basel-Stadt for the following day using information available by 12:00 noon on the preceding day?**

The research question emphasizes two aspects:

1. **Predictive accuracy:** How closely can our models estimate actual next-day electricity consumption?
2. **Real-world applicability:** Can these predictions be generated using only information that would have been available at the time the forecasting decision was made?

The second aspect is particularly important because using information that becomes available only after the prediction deadline would introduce *data leakage* and produce unrealistically optimistic results.

## 2. Decision-Making Context

We consider a hypothetical electricity procurement team responsible for planning electricity purchases for the following day.

The team must prepare its demand forecast by **12:00 noon on day \(t\)** to support purchasing decisions for **day \(t+1\)**.

The forecast must therefore be generated without knowing the actual electricity consumption or weather conditions of the following day.

The predicted quantity is:

**Total electricity consumption in Basel-Stadt during the next calendar day, measured in MWh/day.**

The project focuses on forecasting demand rather than directly optimizing electricity purchases. An economic evaluation of forecasting errors may be added if time permits.

## 3. Data Sources

The project uses two main data sources, together with calendar information.

### Electricity Consumption

**Source:** Open Data Basel-Stadt — electricity consumption dataset (ID: 100233).

The original data contain electricity consumption measurements at 15-minute intervals.

We aggregate these measurements to calculate total electricity consumption for each calendar day.

The resulting daily electricity consumption is the target variable of our forecasting models.

**Daylight-saving time and data completeness:** Daily consumption is aggregated using Swiss local calendar dates. Ordinary days contain 96 quarter-hour observations, while spring daylight-saving transition days contain 92 observations, as expected. However, the ten autumn transition days between 2017 and 2026 contain only 96 observations instead of the expected 100. The second occurrence of the hour between 02:00 and 03:00 CET is absent from the frozen dataset. Consequently, the aggregated consumption totals for these dates may not fully represent the 25-hour calendar day. We retain the original observations without imputing missing intervals and document this limitation in the exploratory data analysis.

Historical electricity consumption may also be used to construct lagged predictors, provided those observations would have been available at the forecasting deadline.

**Electricity data availability:** Because the exact historical publication times of the Basel-Stadt electricity observations cannot be verified, we adopt a conservative availability assumption: only consumption observations from at least five calendar days before the target date (`lag_5` and older) are used as predictors. This rule is applied consistently throughout the project. Missing lagged observations are handled using explicitly documented fallback or preprocessing procedures.

### Historical Weather Forecasts

**Source:** Open-Meteo Historical Forecast / Previous Runs API, using the JMA GSM weather model.

Weather forecasts are relevant because electricity demand may be related to temperature and precipitation.

We collect historical weather forecast information for Basel-Stadt and aggregate it into daily variables, including:

- Mean forecast temperature
- Minimum forecast temperature
- Maximum forecast temperature
- Forecast precipitation

We use historical forecasts rather than realized weather observations to better approximate the information available when making a real prediction.

**Availability at the 12:00 cutoff:** we use the `previous_day2` forecasts, i.e. the forecast issued about 48 hours before each hour. For every hour of day t+1 this forecast was made on day t−1, so it is always available at 12:00 on day t. (`previous_day1`, issued ~24 hours before, would include forecasts made after the cutoff for the afternoon and evening of day t+1.)

For 318 hours between 13 and 26 April 2026 the archive has no `previous_day2` (or `previous_day3`) precipitation. For those hours only, we use the `previous_day4` forecast, which is older and therefore also available at the cutoff. The column `precip_fallback_hours` in `weather_daily.csv` records how many hours of each day use this fallback.

### Calendar Information

Calendar variables can be derived directly from the forecast date, including:

- Day of the week
- Month of the year
- Weekend indicators
- Other calendar indicators where appropriate

These variables allow the models to capture recurring patterns in electricity demand.

## 4. Methodology

The project follows a structured forecasting workflow.

### Step 1 — Data Collection and Preprocessing

Download the electricity consumption and historical weather forecast datasets.

Clean the data, aggregate observations to daily frequency, inspect missing values, and align the datasets by date.

### Step 2 — Exploratory Data Analysis

Investigate the main characteristics of electricity consumption in Basel-Stadt.

We examine:

- Long-term changes in electricity consumption
- Annual and weekly seasonality
- Potential structural changes over time
- Relationships between electricity consumption and weather conditions
- Data completeness and quality

The results help determine which predictors and historical training periods are appropriate.

### Step 3 — Feature Engineering

Construct explanatory variables from information available before the forecasting deadline.

Potential features include historical consumption lags, weather forecasts, and calendar variables.

All features must be checked for availability at the time of prediction to avoid data leakage.

**Implemented in `code/04_prepare_features.py`** → `data/processed/features.csv` (one row per target day t+1, only information available at 12:00 on day t):

| Group | Columns | Reason |
|---|---|---|
| Target | `consumption_mwh` | Value to predict |
| Baseline | `baseline` (`lag_7`, with fallback to `lag_14`, `lag_21`, `lag_28`, then a historical same-weekday average and, if necessary, an eligible historical overall average), `baseline_uses_fallback` | Same-weekday historical benchmark with a predefined fallback for missing consumption observations |
| Calendar | `weekday`, `is_weekend`, `month`, `is_holiday` (Basel-Stadt), `is_christmas_period` (24 Dec–6 Jan) | Weekends and holidays are 13–20% lower; Christmas-period workdays ~10% lower |
| Holiday in lags | `lag_7_is_holiday`, `lag_14_is_holiday` | If day t+1−7 was a holiday, `lag_7` is abnormally low (baseline underestimates by 554 MWh on average in 2018–2024) |
| Weather (forecast) | mean/min/max temperature, precipitation, `heating_degrees` = max(0, 15 − T), `cooling_degrees` = max(0, T − 15) | U-shaped temperature effect, minimum around 11–16 °C |
| Consumption lags | `lag_5`, `lag_6`, `lag_7`, `lag_14`, `lag_21`, `lag_28`, `lag_mean_5_11` | Historical electricity consumption predictors using observations from at least five calendar days before the target date, consistent with our conservative data-availability assumption |
| Period | `period` = train (2018–2024), validation (2025), test (Jan–Oct 2026) | Methodology split. For the monthly validation folds, select training rows by date. |

Scaling/standardization is deliberately not done here: it must be fitted on each fold's training data only.

**Missing-data handling:** Missing daily electricity consumption targets are excluded from model fitting and evaluation, but missing calendar dates remain represented during lag construction so that lags always correspond to the correct calendar-day offsets. Missing predictor values are preserved in the feature table and handled through preprocessing fitted exclusively on each model's eligible training data. The same-weekday baseline first uses consumption from seven days earlier, then from 14, 21 and 28 days earlier. If none of these observations is available, it uses the historical average for the same weekday based only on eligible earlier observations, followed by an eligible historical overall average if needed. These rules avoid using future consumption information.

### Step 4 — Baseline and Forecasting Models

We begin with a simple benchmark:

**Same-weekday-last-week baseline:** Predict tomorrow's electricity consumption using the consumption observed seven days earlier.

We then evaluate whether regression-based forecasting models can improve upon this benchmark.

The final selection of models and features will depend on the exploratory analysis and the available data.

### Step 5 — Model Evaluation

Because electricity consumption is a time series, observations must be evaluated in chronological order.

We plan to use:

- **Expanding-window validation:** Evaluate models on successive future periods while increasing the amount of historical training data.
- **Final holdout test set:** Reserve the most recent period for a final evaluation that is not used for model selection.

The main evaluation metric is **Mean Absolute Error (MAE)**, measured in MWh/day.

Root Mean Squared Error (RMSE) provides an additional measure that gives greater weight to large forecasting errors.

We compare all forecasting models against the baseline using the same evaluation periods.

## 5. Expected Outcome

The project aims to determine whether machine learning models can provide more accurate next-day electricity consumption forecasts than a simple historical benchmark.

The final deliverables will include:

- A reproducible Python forecasting workflow
- Exploratory analysis of Basel-Stadt electricity demand
- A comparison of forecasting models and their predictive accuracy
- An assessment of model limitations and practical applicability
- A discussion of the usefulness of the forecasts for electricity procurement planning

The objective is not to guarantee perfect forecasts, but to evaluate which methods provide reliable and practically useful predictions using information that could realistically have been available at the forecasting deadline.

---

## 6. Getting Started

The following sections explain how to set up the Python environment, install dependencies, execute scripts and notebooks, and collaborate through GitHub.


### 6.1 Create a virtual environment

Every team member must create their own local virtual environment.

A virtual environment isolates the project's Python packages from those installed elsewhere on your computer.


#### macOS / Linux

Create the environment:

```bash
python3.11 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

#### Windows (PowerShell)

Create the environment:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

#### Verify activation

Your terminal should display `(.venv)` at the beginning of the prompt.

Check the Python version:

```bash
python --version
```

It should show Python 3.11.x.

**Note:** You must reactivate the virtual environment whenever you open a new terminal that does not automatically activate it.

To deactivate it:

```bash
deactivate
```

---

### 6.2 Install project dependencies

With the virtual environment activated, run:

```bash
python -m pip install -r requirements.txt
```


---

### 6.3 Configure Python in VS Code

After installing the dependencies:

1. Open the project in VS Code.
2. Press `Cmd + Shift + P` (macOS) or `Ctrl + Shift + P` (Windows).
3. Search for **Python: Select Interpreter**.
4. Select the Python interpreter inside the project's `.venv` folder.

The interpreter path should resemble:

**macOS / Linux:**

```text
.venv/bin/python
```

**Windows:**

```text
.venv\Scripts\python.exe
```

This ensures VS Code uses the project's Python environment.

---

### 6.4 Run Jupyter notebooks

The project uses Jupyter notebooks (`.ipynb`) for exploratory data analysis and modeling.

To run a notebook:

1. Open the desired `.ipynb` file in VS Code.
2. Click **Select Kernel** in the top-right corner.
3. Choose **Python Environments**.
4. Select the interpreter from the project's `.venv`.
5. Run individual cells using the play button, or select **Run All**.

To verify that the correct environment is being used, execute:

```python
import sys

print(sys.executable)
print(sys.version)
```

The executable path should point to the project's `.venv` directory, and the Python version should be 3.11.x.

**Important:** Run notebook cells in the correct order, because later cells may depend on variables created earlier.

---

## 7. Project structure

The project is organized approximately as follows:

```text
DSF_forecasting_electricity_consumption_Basel/
│
├── code/
│   ├── 01_download_electricity.py
│   ├── 02_download_weather.py
│   ├── 03_data_exploration.ipynb
│   ├── 04_prepare_features.py
│   └── 05_ … 11_ (baselines, models, validation, final test)
│
├── data/
│   ├── raw/
│   │   ├── electricity_15min_2026-11-11.csv   (frozen snapshot)
│   │   ├── weather_hourly_2026-11-11.csv      (old snapshot, previous_day1, not used)
│   │   └── weather_hourly_2026-11-17.csv      (frozen snapshot, previous_day2)
│   │
│   └── processed/
│       ├── electricity_daily.csv
│       ├── weather_daily.csv
│       └── features.csv                       (feature table for all models)
│
├── output/
│   ├── figures/
│   └── tables/
│
├── .gitignore
├── .python-version
├── requirements.txt
└── README.md
```


### Folder descriptions

| Folder | Purpose |
|---|---|
| `code/` | Python scripts and Jupyter notebooks |
| `data/raw/` | Downloaded data before aggregation |
| `data/processed/` | Cleaned and aggregated datasets |
| `output/figures/` | Generated visualizations |
| `output/tables/` | Generated results and summary tables |

Some data files may need to be generated locally if they are not included in the repository.

---

## 8. Run Python scripts

Python scripts can be executed from the VS Code terminal.

Make sure the virtual environment is activated and the terminal is in the repository root.

To rebuild the processed daily datasets from the frozen raw snapshots (no internet needed):

```bash
python code/01_download_electricity.py
python code/02_download_weather.py
```

Then build the feature table used by all models:

```bash
python code/04_prepare_features.py
```

### Frozen data snapshot (group rule)

All team members use **the same dated raw files**, committed in `data/raw/`:

- `electricity_15min_2026-11-11.csv`
- `weather_hourly_2026-11-17.csv` (`previous_day2` forecasts; the older `weather_hourly_2026-11-11.csv` used `previous_day1` and is kept only for reference)

**Why two different dates?** The date in each filename indicates when the file was downloaded. Both snapshots cover the relevant historical period up to 8 November 2026. The weather snapshot is dated 17 November 2026 because the group switched from `previous_day1` to `previous_day2` forecasts after the initial electricity snapshot was saved. To investigate whether the weather archive had changed, we compared 3,552 hourly `previous_day1` observations from five sample periods between 2018 and 2026 across the downloads of 11 and 17 November. No differences were detected in these sampled observations. However, this does not establish that the entire historical forecast archive is immutable. Both frozen snapshots are retained to ensure reproducibility.

The source websites can change their data at any time (on 2026-11-12 a re-download of the Basel dataset was missing 44 days in 2026). Re-downloading would give each of us different data and different results. Therefore:

1. **Never re-download by default.** The scripts only read the frozen snapshot named by `SNAPSHOT_DATE` at the top of each script.
2. **Never edit or overwrite files in `data/raw/`.**
3. A new snapshot is created only with `--new-snapshot`, which saves a *new* file named with today's date and never overwrites anything:

   ```bash
   python code/01_download_electricity.py --new-snapshot
   ```

4. Switching to a new snapshot (changing `SNAPSHOT_DATE`) is a **group decision**, made through a pull request that states why and what changed (rows, date range, missing days).
5. If `git status` shows changes in `data/` that you did not intend, do **not** commit them; run `git restore data/`.

To explore the data, open:

```text
code/03_data_exploration.ipynb
```

Select the project's `.venv` kernel and execute the notebook cells.

---

## 9. Collaborating with GitHub

Because four people are working on this repository, everyone should follow the same Git workflow.

### Before starting work

Always update your local repository:

```bash
git pull
```

If you have uncommitted changes, commit or stash them before pulling when necessary.


### Save your changes

Check which files have changed:

```bash
git status
```

Stage only the files relevant to your task:

```bash
git add code/03_data_exploration.ipynb
```

Commit your changes:

```bash
git commit -m "Add monthly electricity seasonality analysis"
```

Push your branch:

```bash
git push -u origin feature/my-task
```

Then open a pull request on GitHub so another team member can review and merge the changes.

---

## Final Results

We compared a same-weekday historical baseline, linear regression models (OLS, Ridge and Lasso), and Random Forest models using chronological monthly validation in 2025. We also investigated whether expanding or rolling training windows improved forecasting performance.

The best-performing configuration on the 2025 validation data was **ordinary least squares (OLS) regression with historical electricity consumption, archived weather forecasts and calendar features, using a rolling three-year training window**.

| Evaluation metric | 2025 validation | 2026 final test |
|---|---:|---:|
| OLS rolling three-year MAE | 74.64 MWh/day | 72.59 MWh/day |
| OLS rolling three-year RMSE | 97.66 MWh/day | 94.63 MWh/day |
| Evaluation days | 365 | 304 |

The final test covered January 1–October 31, 2026. The historical baseline achieved a test MAE of 161.48 MWh/day, compared with 72.59 MWh/day for the selected OLS model. This corresponds to a **55.05% reduction in MAE**.

Model selection was based exclusively on 2025 validation performance. The 2026 test period was used for final evaluation, with monthly model refitting according to the previously established five-calendar-day consumption-data availability rule.

### Limitations

The forecasting exercise relies on conservative assumptions about historical consumption-data publication timing and the availability of archived weather forecasts. Exact historical publication times could not be fully reconstructed.

The frozen electricity dataset also does not explicitly represent the repeated hour on autumn daylight-saving transition days, potentially affecting the corresponding daily totals. Missing observations are not reconstructed using invented consumption values.

The reported results concern retrospective prediction accuracy on the frozen dataset. They do not establish the economic value of the forecasts or guarantee equivalent accuracy in future operational use.



