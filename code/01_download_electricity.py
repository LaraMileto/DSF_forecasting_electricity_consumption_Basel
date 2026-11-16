"""
Electricity data: frozen snapshot + daily aggregation.

Default (what everyone should run):
    python code/01_download_electricity.py
    -> reads the FROZEN raw snapshot below and rebuilds
       data/processed/electricity_daily.csv. No internet access.

Only after the group agrees to change the snapshot:
    python code/01_download_electricity.py --new-snapshot
    -> downloads a NEW file data/raw/electricity_15min_<today>.csv.
       It never overwrites an existing file and does not change the
       processed data. To adopt it, update SNAPSHOT_DATE below.
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests


# 1. PATHS AND FROZEN SNAPSHOT

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

# Date on which the group's shared raw file was downloaded.
SNAPSHOT_DATE = "2026-11-11"
SNAPSHOT_FILE = RAW / f"electricity_15min_{SNAPSHOT_DATE}.csv"

# Basel-Stadt Open Data: dataset 100233
URL = (
    "https://data.bs.ch/api/explore/v2.1/catalog/"
    "datasets/100233/exports/csv"
)


# 2. OPTIONAL: DOWNLOAD A NEW SNAPSHOT (never overwrites)

if "--new-snapshot" in sys.argv:
    new_file = RAW / f"electricity_15min_{date.today().isoformat()}.csv"

    if new_file.exists():
        sys.exit(f"{new_file.name} already exists. Nothing downloaded.")

    response = requests.get(URL, timeout=180)
    response.raise_for_status()
    new_file.write_bytes(response.content)

    print(f"New snapshot saved: {new_file}")
    print(
        "The processed data still uses the frozen snapshot "
        f"({SNAPSHOT_DATE}). Check the new file for gaps, and only "
        "update SNAPSHOT_DATE after agreeing with the group."
    )
    sys.exit(0)


# 3. LOAD THE FROZEN SNAPSHOT

if not SNAPSHOT_FILE.exists():
    sys.exit(
        f"Frozen snapshot not found: {SNAPSHOT_FILE}\n"
        "Run `git pull` to get it from the repository."
    )

df = pd.read_csv(SNAPSHOT_FILE, sep=";")

print("Snapshot:", SNAPSHOT_FILE.name)
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


# 4. AGGREGATE 15-MINUTE ENERGY TO DAILY MWh

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


# 5. QUALITY CHECKS

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
