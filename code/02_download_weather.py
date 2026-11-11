
from pathlib import Path
from datetime import timedelta
import time

import pandas as pd
import requests


# 1. PATHS

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)


# 2. JMA API CONFIGURATION

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

START_DATE = "2018-01-01"
END_DATE = "2026-11-08"

TEMP = "temperature_2m_previous_day1"
PRECIP = "precipitation_previous_day1"


# 3. DOWNLOAD MONTH BY MONTH

months = pd.period_range(
    START_DATE,
    END_DATE,
    freq="M"
)

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
        "hourly": f"{TEMP},{PRECIP}",
        "models": "jma_gsm",
        "timezone": "Europe/Zurich"
    }

    print(f"Downloading {start} to {end}...")

    for attempt in range(1, 4):
        try:
            response = session.get(
                URL,
                params=params,
                timeout=120
            )
            response.raise_for_status()

            data = pd.DataFrame(
                response.json()["hourly"]
            )

            required = ["time", TEMP, PRECIP]

            if not set(required).issubset(data.columns):
                raise ValueError(
                    f"Missing API columns for {month}"
                )

            all_months.append(data[required])
            break

        except (
            requests.RequestException,
            KeyError,
            ValueError
        ) as error:

            print(
                f"Attempt {attempt} failed: {error}"
            )

            if attempt == 3:
                raise RuntimeError(
                    f"Download failed for {month}"
                ) from error

            time.sleep(3 * attempt)

    time.sleep(0.5)

session.close()


# 4. COMBINE HOURLY DATA

weather = pd.concat(
    all_months,
    ignore_index=True
)

weather["time"] = pd.to_datetime(
    weather["time"]
)

weather = weather.sort_values("time")

duplicates = weather["time"].duplicated().sum()

print("\nDuplicate hourly timestamps:", duplicates)

if duplicates:
    raise ValueError(
        "Duplicate weather timestamps detected."
    )


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
        valid_precip_hours=(PRECIP, "count")
    )
)

daily["date"] = pd.to_datetime(daily["date"])


# 6. CHECK COMPLETENESS

daily["complete"] = (
    (daily["valid_temp_hours"] ==
     daily["hourly_observations"]) &
    (daily["valid_precip_hours"] ==
     daily["hourly_observations"]) &
    (daily["valid_temp_hours"] > 0)
)

expected_dates = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

missing_dates = expected_dates.difference(
    daily["date"]
)

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
        "Weather archive has missing or incomplete days. "
        "Existing CSV files have not been overwritten."
    )


# 7. SAVE DATA AFTER VALIDATION

weather.drop(columns="date").to_csv(
    RAW / "weather_hourly.csv",
    index=False
)

daily.to_csv(
    PROCESSED / "weather_daily.csv",
    index=False
)

print("\nJMA weather files saved successfully.")
