"""
04 - Build the feature table used by all models (05-11).

    python code/04_prepare_features.py

Output: data/processed/features.csv

One row = one TARGET day d (= day t+1). Every feature column only uses
information available at 12:00 on day t = d-1 (our forecasting cutoff):

- Calendar features: known in advance.
- Weather: Open-Meteo "previous_day2" forecasts, issued ~48 h before
  each hour, i.e. before the cutoff (see 02_download_weather.py).
- Consumption lags: lag_k = consumption on day d-k. The Basel data are
  published with a delay of about 3 days (at noon on day t the last
  complete day is t-3). We use lag_5 (= day t-4) and older, which keeps
  a one-day safety margin.

Note for the model notebooks: the "period" column gives the three
methodology periods. For the 12 monthly validation folds in 2025, select
the training rows BY DATE (all days before the predicted month), not with
"period", otherwise earlier 2025 months would never be used for training.

Not done here on purpose: scaling / standardization. It must be fitted
on the training data of each fold only, so it belongs in the model
notebooks (otherwise information from later periods would leak in).
"""

from pathlib import Path

import holidays
import pandas as pd


# 1. SETTINGS

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

# Periods from our methodology
PERIODS = {
    "train": ("2018-01-01", "2024-12-31"),
    "validation": ("2025-01-01", "2025-12-31"),
    "test": ("2026-01-01", "2026-10-31"),
}

# Temperature separating heating from cooling. In the training data,
# workday consumption is flat between about 11 and 16 degC and rises on
# both sides (see 03_data_exploration, section 8).
BASE_TEMP = 15.0

LAGS = [5, 6, 7, 14, 21, 28]  # only lags available at the cutoff


# 2. LOAD DATA (built from the frozen snapshots by scripts 01 and 02)

electricity = pd.read_csv(
    PROCESSED / "electricity_daily.csv",
    parse_dates=["date"]
)

weather = pd.read_csv(
    PROCESSED / "weather_daily.csv",
    parse_dates=["date"]
)

# Put consumption on a complete daily calendar, so that "shift by k rows"
# always means "k calendar days earlier". A missing day becomes NaN
# (it is never invented).
consumption = (
    electricity.set_index("date")["consumption_mwh"]
    .sort_index()
    .asfreq("D")
)

table = consumption.to_frame()
table.index.name = "date"


# 3. CONSUMPTION LAGS (only values published before the cutoff)

for k in LAGS:
    table[f"lag_{k}"] = consumption.shift(k)

# Average of the most recent known week (days d-11 ... d-5):
# gives the recent consumption level, less sensitive to one unusual day.
table["lag_mean_5_11"] = consumption.rolling(7).mean().shift(5)


# 4. BASELINE: same weekday last week, fallback = two weeks before

LAGS = [5, 6, 7, 14, 21, 28]

table["baseline"] = (
    table["lag_7"]
    .fillna(table["lag_14"])
    .fillna(table["lag_21"])
    .fillna(table["lag_28"])
)
table["baseline_uses_fallback"] = (
    table["lag_7"].isna() & table["baseline"].notna()
)


# 5. CALENDAR FEATURES

dates = table.index

table["weekday"] = dates.day_of_week          # 0 = Monday, 6 = Sunday
table["is_weekend"] = (table["weekday"] >= 5).astype(int)
table["month"] = dates.month

bs_holidays = holidays.Switzerland(
    subdiv="BS",
    years=range(dates.year.min(), dates.year.max() + 1)
)
table["is_holiday"] = [int(d in bs_holidays) for d in dates.date]

# Was the lagged day a holiday? Then lag_7 / lag_14 are abnormally low.
# Training data 2018-2024 (target day not a holiday): when day d-7 was a
# holiday, the lag_7 baseline underestimates by 554 MWh on average
# (56 days), vs. an average absolute error of 113 MWh otherwise.
# These flags let the models correct for it. The baseline itself stays
# plain lag_7, as defined in our methodology. Holidays are known in
# advance, so this is available at the cutoff.
table["lag_7_is_holiday"] = table["is_holiday"].shift(7)
table["lag_14_is_holiday"] = table["is_holiday"].shift(14)

# 24 Dec - 6 Jan: workdays about 10% lower than other Dec/Jan workdays
# (training data 2018-2024), mostly not covered by public holidays.
month_day = dates.strftime("%m-%d")
table["is_christmas_period"] = (
    (month_day >= "12-24") | (month_day <= "01-06")
).astype(int)


# 6. WEATHER FEATURES (forecasts for the target day)

weather_cols = [
    "forecast_temp_mean",
    "forecast_temp_min",
    "forecast_temp_max",
    "forecast_precipitation",
]

table = table.join(weather.set_index("date")[weather_cols])

# Turn the U-shaped temperature effect into two straight lines
table["heating_degrees"] = (BASE_TEMP - table["forecast_temp_mean"]).clip(lower=0)
table["cooling_degrees"] = (table["forecast_temp_mean"] - BASE_TEMP).clip(lower=0)


# 7. PERIOD LABELS (rows outside the three periods are dropped)

table["period"] = None

for name, (start, end) in PERIODS.items():
    table.loc[start:end, "period"] = name

table = table[table["period"].notna()].reset_index()


# 8. MISSING VALUES

# Missing actual consumption: excluded, never invented (methodology rule)
missing_target = table["consumption_mwh"].isna()
print("Rows removed because actual consumption is missing:", missing_target.sum())
table = table[~missing_target]

missing = table.isna().sum()
missing = missing[missing > 0]

if len(missing):
    print("\nMissing feature values per column:")
    print(missing)
    raise ValueError(
        "Some features are missing. Decide as a group how to handle them "
        "before modelling. features.csv was not saved."
    )


# 9. SANITY CHECKS

# lag_7 of a day must equal the consumption 7 calendar days before
check_day = pd.Timestamp("2022-06-15")
expected = consumption.loc[check_day - pd.Timedelta(days=7)]
actual = table.loc[table["date"] == check_day, "lag_7"].iloc[0]
assert abs(expected - actual) < 1e-9, "lag_7 is not 7 days before"

# The newest lag must be at least 5 days before the target day
assert min(LAGS) >= 5, "Lags younger than 5 days are not available at the cutoff"


# 10. SAVE

column_order = (
    ["date", "period", "consumption_mwh", "baseline", "baseline_uses_fallback"]
    + ["weekday", "is_weekend", "month", "is_holiday", "is_christmas_period",
       "lag_7_is_holiday", "lag_14_is_holiday"]
    + weather_cols + ["heating_degrees", "cooling_degrees"]
    + [f"lag_{k}" for k in LAGS] + ["lag_mean_5_11"]
)

table = table[column_order]
table["lag_7_is_holiday"] = table["lag_7_is_holiday"].astype(int)
table["lag_14_is_holiday"] = table["lag_14_is_holiday"].astype(int)
table.to_csv(PROCESSED / "features.csv", index=False)


# 11. SUMMARY

print("\nSaved:", PROCESSED / "features.csv")
print("Rows:", len(table), "| Columns:", len(table.columns))

print("\nRows per period:")
print(
    table.groupby("period", sort=False)["date"]
    .agg(first="min", last="max", days="count")
)

print("\nBaseline rows using a fallback:",
      table["baseline_uses_fallback"].sum())
print("Holidays:", table["is_holiday"].sum(),
      "| Christmas-period days:", table["is_christmas_period"].sum())
