
from pathlib import Path
import pandas as pd
import requests


# 1. Folder paths

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)



# 2. API configuration

url = "https://previous-runs-api.open-meteo.com/v1/forecast"

params = {
    "latitude": 47.56,
    "longitude": 7.59,
    "start_date": "2024-01-01",
    "end_date": "2026-11-08",
    "hourly": (
        "temperature_2m_previous_day1,"
        "precipitation_previous_day1"
    ),
    "timezone": "Europe/Zurich"
}


# 3. Download historical forecasts

print("Downloading historical weather forecasts...")

response = requests.get(
    url,
    params=params,
    timeout=180
)

response.raise_for_status()
data = response.json()

if "hourly" not in data:
    raise ValueError(
        f"Missing hourly weather data: {data}"
    )

raw_file = RAW / "weather_hourly.csv"

weather = pd.DataFrame(data["hourly"])
weather.to_csv(raw_file, index=False)

print("Downloaded rows:", len(weather))
print("Columns:", weather.columns.tolist())



# 4. Convert hourly data to daily features

weather["time"] = pd.to_datetime(weather["time"])
weather["date"] = weather["time"].dt.date

daily = (
    weather.groupby("date", as_index=False)
    .agg(
        forecast_temp_mean=(
            "temperature_2m_previous_day1",
            "mean"
        ),
        forecast_temp_min=(
            "temperature_2m_previous_day1",
            "min"
        ),
        forecast_temp_max=(
            "temperature_2m_previous_day1",
            "max"
        ),
        forecast_precipitation=(
            "precipitation_previous_day1",
            "sum"
        ),
        hourly_observations=("time", "size"),
        valid_temp_hours=(
            "temperature_2m_previous_day1",
            "count"
        ),
        valid_precip_hours=(
            "precipitation_previous_day1",
            "count"
        )
    )
)


# 5. Save processed data

daily["date"] = pd.to_datetime(daily["date"])

daily.to_csv(
    PROCESSED / "weather_daily.csv",
    index=False
)


# 6. Quality checks

print("\nFirst five rows:")
print(daily.head())

print("\nDate range:")
print(daily["date"].min(), daily["date"].max())

print("\nTotal daily observations:", len(daily))

print("\nMissing values:")
print(daily.isna().sum())

print("\nHourly observation counts:")
print(daily["hourly_observations"].value_counts())

print("\nDays with incomplete forecast data:")
incomplete = daily[
    (daily["valid_temp_hours"] <
     daily["hourly_observations"]) |
    (daily["valid_precip_hours"] <
     daily["hourly_observations"])
]
print(incomplete.head(20).to_string(index=False))
print("Total incomplete days:", len(incomplete))



# 7. Check availability of historical forecasts

valid_weather = daily.dropna(
    subset=["forecast_temp_mean"]
)

print("\nWEATHER DATA AVAILABILITY")

print("Total days:", len(daily))

print("Days with valid temperature forecasts:",
      len(valid_weather))

print("Days without valid temperature forecasts:",
      daily["forecast_temp_mean"].isna().sum())

if not valid_weather.empty:
    print("\nFirst valid forecast date:",
          valid_weather["date"].min())

    print("Last valid forecast date:",
          valid_weather["date"].max())

    print("\nFirst five valid observations:")
    print(valid_weather.head())
else:
    print("\nNo valid temperature forecasts found.")


# 8. Check for fully complete weather days
complete_weather = daily[
    (daily["valid_temp_hours"] == daily["hourly_observations"]) &
    (daily["valid_precip_hours"] == daily["hourly_observations"]) &
    (daily["valid_temp_hours"] > 0)
].copy()

print("\nFINAL WEATHER QUALITY CHECK")
print("Complete weather days:", len(complete_weather))
print("First complete day:", complete_weather["date"].min())
print("Last complete day:", complete_weather["date"].max())