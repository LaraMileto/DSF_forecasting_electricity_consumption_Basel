
from pathlib import Path
import pandas as pd
import requests

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

# Basel-Stadt Open Data: dataset 100233
url = (
    "https://data.bs.ch/api/explore/v2.1/catalog/"
    "datasets/100233/exports/csv"
)

response = requests.get(url, timeout=180)
response.raise_for_status()

raw_file = RAW / "electricity_15min.csv"
raw_file.write_bytes(response.content)

# Parse CSV and preserve the original data
df = pd.read_csv(raw_file, sep=";")

print("Columns:", df.columns.tolist())
print("Rows:", len(df))

# Use the original timestamp and electricity consumption
df["timestamp"] = pd.to_datetime(
    df["timestamp_interval_start"],
    utc=True
).dt.tz_convert("Europe/Zurich")

df["stromverbrauch_kwh"] = pd.to_numeric(
    df["stromverbrauch_kwh"], errors="coerce"
)

df = df.dropna(subset=[
    "timestamp", "stromverbrauch_kwh"
])

# Convert 15-minute electricity ENERGY to daily MWh
df["date"] = df["timestamp"].dt.date

daily = (
    df.groupby("date", as_index=False)
      .agg(
          consumption_mwh=(
              "stromverbrauch_kwh",
              lambda x: x.sum() / 1000
          ),
          intervals=("stromverbrauch_kwh", "size")
      )
)

daily["date"] = pd.to_datetime(daily["date"])
daily = daily[daily["date"] >= "2017-01-01"]

daily.to_csv(
    PROCESSED / "electricity_daily.csv",
    index=False
)

print(daily.head())
print("Date range:", daily.date.min(), daily.date.max())
print("Interval counts:", daily.intervals.value_counts())


# Identify days with unusual numbers of observations
unusual_days = daily[daily["intervals"] != 96]

print("\nDays with unusual interval counts:")
print(unusual_days.to_string(index=False))


# Check whether any calendar dates are missing
expected_dates = pd.date_range(
    daily["date"].min(),
    daily["date"].max(),
    freq="D"
)

missing_dates = expected_dates.difference(daily["date"])

print("\nMissing dates:")
print(missing_dates)

print("\nTotal daily observations:", len(daily))