"""
Weather forecast data (Open-Meteo, JMA GSM): frozen snapshot + daily aggregation.

Default (what everyone should run):
    python code/02_download_weather.py
    -> reads the FROZEN raw snapshot below and rebuilds
       data/processed/weather_daily.csv. No internet access.

Only after the group agrees to change the snapshot:
    python code/02_download_weather.py --new-snapshot
    -> downloads a NEW file data/raw/weather_hourly_<today>.csv.
       It never overwrites an existing file and does not change the
       processed data. To adopt it, update SNAPSHOT_DATE below.
"""

import sys
import time
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
SNAPSHOT_DATE = "2026-11-17"
SNAPSHOT_FILE = RAW / f"weather_hourly_{SNAPSHOT_DATE}.csv"


# 2. JMA API CONFIGURATION

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

START_DATE = "2018-01-01"
END_DATE = "2026-11-08"

# "previous_day2" = forecast issued about 48 hours before each hour.
# For every hour of day t+1 it was made on day t-1, so it is always
# available at our 12:00 cutoff on day t. ("previous_day1", ~24 h before,
# would use forecasts made after the cutoff for the afternoon of t+1.)
TEMP = "temperature_2m_previous_day2"
PRECIP = "precipitation_previous_day2"

# Fallback for hours where the archive has no previous_day2 precipitation
# (13-26 April 2026: also missing for previous_day3). The ~96 h old
# forecast was made even earlier, so it is also available at the cutoff.
PRECIP_FALLBACK = "precipitation_previous_day4"


def download_weather():
    """Download hourly JMA forecasts month by month."""
    months = pd.period_range(START_DATE, END_DATE, freq="M")
    all_months = []
    session = requests.Session()

    for month in months:
        start = max(
            month.start_time.date(),
            pd.Timestamp(START_DATE).date()
        )
        end = min(
            month.end_time.date(),
            pd.Timestamp(END_DATE).date()
        )

        params = {
            "latitude": 47.56,
            "longitude": 7.59,
            "start_date": str(start),
            "end_date": str(end),
            "hourly": f"{TEMP},{PRECIP},{PRECIP_FALLBACK}",
            "models": "jma_gsm",
            "timezone": "Europe/Zurich"
        }

        print(f"Downloading {start} to {end}...")

        for attempt in range(1, 4):
            try:
                response = session.get(URL, params=params, timeout=120)
                response.raise_for_status()

                data = pd.DataFrame(response.json()["hourly"])
                required = ["time", TEMP, PRECIP, PRECIP_FALLBACK]

                if not set(required).issubset(data.columns):
                    raise ValueError(f"Missing API columns for {month}")

                all_months.append(data[required])
                break

            except (requests.RequestException, KeyError, ValueError) as error:
                print(f"Attempt {attempt} failed: {error}")

                if attempt == 3:
                    raise RuntimeError(f"Download failed for {month}") from error

                time.sleep(3 * attempt)

        time.sleep(0.5)

    session.close()
    return pd.concat(all_months, ignore_index=True)


# 3. OPTIONAL: DOWNLOAD A NEW SNAPSHOT (never overwrites)

if "--new-snapshot" in sys.argv:
    new_file = RAW / f"weather_hourly_{date.today().isoformat()}.csv"

    if new_file.exists():
        sys.exit(f"{new_file.name} already exists. Nothing downloaded.")

    download_weather().to_csv(new_file, index=False)

    print(f"\nNew snapshot saved: {new_file}")
    print(
        "The processed data still uses the frozen snapshot "
        f"({SNAPSHOT_DATE}). Check the new file, and only update "
        "SNAPSHOT_DATE after agreeing with the group."
    )
    sys.exit(0)


# 4. LOAD THE FROZEN SNAPSHOT

if not SNAPSHOT_FILE.exists():
    sys.exit(
        f"Frozen snapshot not found: {SNAPSHOT_FILE}\n"
        "Run `git pull` to get it from the repository."
    )

weather = pd.read_csv(SNAPSHOT_FILE)
print("Snapshot:", SNAPSHOT_FILE.name)

weather["time"] = pd.to_datetime(weather["time"])
weather = weather.sort_values("time")

duplicates = weather["time"].duplicated().sum()

print("\nDuplicate hourly timestamps:", duplicates)

if duplicates:
    raise ValueError("Duplicate weather timestamps detected.")

# Fill missing previous_day2 precipitation with the older previous_day4
# forecast (still known before the cutoff) and record where this happened.
weather["precip_from_fallback"] = (
    weather[PRECIP].isna() & weather[PRECIP_FALLBACK].notna()
)
weather[PRECIP] = weather[PRECIP].fillna(weather[PRECIP_FALLBACK])

print("Precipitation hours filled from previous_day4:",
      weather["precip_from_fallback"].sum())


# 5. AGGREGATE TO DAILY WEATHER

weather["date"] = weather["time"].dt.date

daily = (
    weather.groupby("date", as_index=False)
    .agg(
        forecast_temp_mean=(TEMP, "mean"),
        forecast_temp_min=(TEMP, "min"),
        forecast_temp_max=(TEMP, "max"),
        forecast_precipitation=(PRECIP, "sum"),
        hourly_observations=("time", "size"),
        valid_temp_hours=(TEMP, "count"),
        valid_precip_hours=(PRECIP, "count"),
        precip_fallback_hours=("precip_from_fallback", "sum")
    )
)

daily["date"] = pd.to_datetime(daily["date"])


# 6. CHECK COMPLETENESS

daily["complete"] = (
    (daily["valid_temp_hours"] == daily["hourly_observations"]) &
    (daily["valid_precip_hours"] == daily["hourly_observations"]) &
    (daily["valid_temp_hours"] > 0)
)

expected_dates = pd.date_range(START_DATE, END_DATE, freq="D")
missing_dates = expected_dates.difference(daily["date"])

print("\nJMA WEATHER QUALITY CHECK")
print("Total days:", len(daily))
print("Complete days:", daily["complete"].sum())
print("Incomplete days:", (~daily["complete"]).sum())
print("Missing calendar dates:", len(missing_dates))

print("\nHourly observations per day:")
print(daily["hourly_observations"].value_counts())

print("\nFirst complete day:")
print(daily.loc[daily["complete"], "date"].min())

print("\nLast complete day:")
print(daily.loc[daily["complete"], "date"].max())

if len(missing_dates):
    print("\nMissing dates:")
    print(missing_dates)

if not daily["complete"].all() or len(missing_dates):
    raise ValueError(
        "Weather snapshot has missing or incomplete days. "
        "Processed CSV has not been overwritten."
    )


# 7. SAVE PROCESSED DATA

daily.to_csv(PROCESSED / "weather_daily.csv", index=False)

print("\nProcessed weather file saved successfully.")
