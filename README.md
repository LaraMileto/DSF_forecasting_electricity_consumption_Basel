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

Historical electricity consumption may also be used to construct lagged predictors, provided those observations would have been available at the forecasting deadline.

### Historical Weather Forecasts

**Source:** Open-Meteo Historical Forecast / Previous Runs API, using the JMA GSM weather model.

Weather forecasts are relevant because electricity demand may be related to temperature and precipitation.

We collect historical weather forecast information for Basel-Stadt and aggregate it into daily variables, including:

- Mean forecast temperature
- Minimum forecast temperature
- Maximum forecast temperature
- Forecast precipitation

We use historical forecasts rather than realized weather observations to better approximate the information available when making a real prediction.

**Important:** The availability of individual forecast values by the 12:00 cutoff must be verified before they are used in the final forecasting model.

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
│   └── 03_data_exploration.ipynb
│
├── data/
│   ├── raw/
│   │   ├── electricity_15min.csv
│   │   └── weather_hourly.csv
│   │
│   └── processed/
│       ├── electricity_daily.csv
│       └── weather_daily.csv
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

For example, to download electricity data:

```bash
python code/01_download_electricity.py
```

To download historical weather forecast data:

```bash
python code/02_download_weather.py
```

The scripts are intended to save data in the corresponding `data/` folders.

**Important:** Downloading data may require an internet connection and may take some time.

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



## 10. Quick-start checklist

For a new team member:

- [ ] Clone the GitHub repository.
- [ ] Open the repository in VS Code.
- [ ] Verify that Python 3.11 is installed.
- [ ] Create a local `.venv`.
- [ ] Activate the virtual environment.
- [ ] Install packages from `requirements.txt`.
- [ ] Select the `.venv` Python interpreter in VS Code.
- [ ] Select the `.venv` kernel in Jupyter.
- [ ] Verify that the required datasets are available.
- [ ] Run the notebook or script relevant to your task.

**After completing these steps, your local development environment should be ready for the project.**