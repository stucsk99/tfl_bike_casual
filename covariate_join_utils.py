# build_causal_covariates_functional.py
#
# Notebook-friendly, callable version (no argparse required).
# See the bottom for an example call.

# build_causal_covariates_from_panel.py
#
# Notebook-friendly, callable pipeline that assumes your *bike panel already contains*
# lat/lon (and optionally station_name), like:
#   station_id, ts, trips_start, strike_exposed, lat, lon, station_name
#
# It will:
# - derive a station_id -> weather_cell_id mapping from the panel itself (no stations file)
# - cluster into lat/lon grid cells (to limit weather API calls)
# - download & cache hourly historical weather per grid cell (Open-Meteo archive API)
# - add calendar + bank holiday features
# - optionally add sunrise/sunset + is_daylight (Astral)
# - join weather on (weather_cell_id, trips_start)
# - write parquet if out_path is provided

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set

import pandas as pd
import polars as pl
import requests
import random


# -----------------------------
# Grid clustering helpers
# -----------------------------

def km_to_deg_lat(km: float) -> float:
    return km / 111.32  # ~111.32 km per degree latitude

def km_to_deg_lon(km: float, lat_deg: float) -> float:
    return km / (111.32 * max(1e-6, math.cos(math.radians(lat_deg))))

def grid_cell_id(lat: float, lon: float, grid_km: float) -> str:
    """Stable rectangular grid bin id; lon-step varies with latitude."""
    lat_step = km_to_deg_lat(grid_km)
    lon_step = km_to_deg_lon(grid_km, lat)
    lat_idx = math.floor(lat / lat_step)
    lon_idx = math.floor(lon / lon_step)
    return f"g{grid_km:.3f}_lat{lat_idx}_lon{lon_idx}"

def grid_cell_center(lat: float, lon: float, grid_km: float) -> Tuple[float, float]:
    lat_step = km_to_deg_lat(grid_km)
    lon_step = km_to_deg_lon(grid_km, lat)
    lat_idx = math.floor(lat / lat_step)
    lon_idx = math.floor(lon / lon_step)
    return (lat_idx + 0.5) * lat_step, (lon_idx + 0.5) * lon_step


# -----------------------------
# Bank holidays (UK)
# -----------------------------

def fetch_uk_bank_holidays() -> pd.DataFrame:
    """England & Wales bank holidays from GOV.UK JSON endpoint."""
    url = "https://www.gov.uk/bank-holidays.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    payload = r.json()
    events = pd.DataFrame(payload["england-and-wales"]["events"])
    events["date"] = pd.to_datetime(events["date"]).dt.date
    return events[["date", "title"]]


# -----------------------------
# Weather (Open-Meteo archive)
# -----------------------------

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DEFAULT_HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "cloud_cover",
    "wind_speed_10m",
    "weather_code",
]

@dataclass
class WeatherRequest:
    lat: float
    lon: float
    start_date: str
    end_date: str
    timezone: str
    hourly_vars: List[str]

def fetch_open_meteo_hourly_with_retry(req, max_retries=8, base_sleep=2.0, jitter=0.25):
    """
    Robust Open-Meteo fetch that handles 429 rate limiting via exponential backoff.
    - max_retries: number of attempts after the first
    - base_sleep: initial wait (seconds)
    - jitter: random extra sleep fraction to desync
    """
    params = {
        "latitude": req.lat,
        "longitude": req.lon,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "timezone": req.timezone,
        "hourly": ",".join(req.hourly_vars),
    }

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=90)

            # If rate-limited, back off and retry
            if r.status_code == 429:
                # If server provides Retry-After, prefer it
                retry_after = r.headers.get("Retry-After")
                if retry_after is not None:
                    sleep_s = float(retry_after)
                else:
                    sleep_s = base_sleep * (2 ** attempt)

                # add jitter
                sleep_s *= (1.0 + random.uniform(0, jitter))
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            js = r.json()

            if "hourly" not in js or "time" not in js["hourly"]:
                raise RuntimeError(f"Unexpected Open-Meteo response: {json.dumps(js)[:600]}")

            w = pd.DataFrame(js["hourly"])
            w["trips_start"] = pd.to_datetime(w["time"])
            w = w.drop(columns=["time"], errors="ignore")
            return w

        except Exception as e:
            last_err = e
            # backoff on other transient failures too
            sleep_s = base_sleep * (2 ** attempt)
            sleep_s *= (1.0 + random.uniform(0, jitter))
            time.sleep(sleep_s)

    raise last_err

def load_or_fetch_weather_for_cell(
    cell_id: str,
    cell_lat: float,
    cell_lon: float,
    start_date: str,
    end_date: str,
    timezone: str,
    cache_dir: str,
    hourly_vars: List[str],
    polite_sleep_s: float,
) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"weather_{cell_id}_{start_date}_{end_date}.parquet")

    if os.path.exists(cache_path):
        w = pd.read_parquet(cache_path)
        if "weather_cell_id" not in w.columns:
            w["weather_cell_id"] = cell_id
        return w

    req = WeatherRequest(
        lat=cell_lat,
        lon=cell_lon,
        start_date=start_date,
        end_date=end_date,
        timezone=timezone,
        hourly_vars=hourly_vars,
    )
    w = fetch_open_meteo_hourly_with_retry(req)
    w["weather_cell_id"] = cell_id
    tmp_path = cache_path + ".tmp"
    w.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, cache_path)

    time.sleep(polite_sleep_s)
    return w


# -----------------------------
# Sunrise/sunset (Astral, optional)
# -----------------------------

def build_daylight_daily_by_cell(
    cells_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    timezone: str,
) -> Optional[pd.DataFrame]:
    """
    Per-(weather_cell_id, date): sunrise, sunset.
    Returns None if astral isn't available.
    """
    try:
        from astral import LocationInfo
        from astral.sun import sun
        import pytz
    except Exception:
        return None

    tz = pytz.timezone(timezone)
    dates = pd.date_range(start_date, end_date, freq="D").date

    rows = []
    for _, row in cells_df.iterrows():
        cid = row["weather_cell_id"]
        lat = float(row["cell_lat"])
        lon = float(row["cell_lon"])
        loc = LocationInfo(name=str(cid), region="London", timezone=timezone, latitude=lat, longitude=lon)

        for d in dates:
            s = sun(loc.observer, date=d, tzinfo=tz)
            rows.append(
                {
                    "weather_cell_id": cid,
                    "date": d,
                    "sunrise": s["sunrise"],
                    "sunset": s["sunset"],
                }
            )
    return pd.DataFrame(rows)


# -----------------------------
# Main callable function
# -----------------------------

def build_bike_panel_with_covariates_from_panel(
    bike_path: str,
    out_path: Optional[str] = None,
    *,
    bike_format: str = "parquet",          # "parquet" or "csv"
    cache_dir: str = "cache_weather",
    start_date: str = "2016-01-01",
    end_date: str = "2018-12-31",
    timezone: str = "Europe/London",
    grid_km: float = 1.0,
    hourly_vars: Optional[List[str]] = None,
    include_daylight: bool = True,
    polite_sleep_s: float = 0.25,
    station_sample_limit: Optional[int] = None,
    streaming_collect: bool = True,
) -> pl.DataFrame:
    """
    Build a feature-enriched station-hour panel *using lat/lon already in the bike panel*.

    Required columns in bike data:
      - station_id
      - trips_start (hourly timestamp)
      - ts (trips count)
      - strike_exposed (0/1)
      - lat
      - lon
    Optional:
      - station_name, other station metadata

    Returns a Polars DataFrame (also writes parquet if out_path is provided).
    """

    if hourly_vars is None:
        hourly_vars = DEFAULT_HOURLY_VARS

    # ---- Load the bike panel lazily
    if bike_format == "parquet":
        bike = pl.scan_parquet(bike_path)
    elif bike_format == "csv":
        bike = pl.scan_csv(bike_path, try_parse_dates=True)
    else:
        raise ValueError("bike_format must be 'parquet' or 'csv'")

    # Ensure types for key columns
    bike = bike.with_columns([
        pl.col("station_id").cast(pl.Int32),
        pl.col("strike_exposed").cast(pl.Int8),
        pl.col("ts").cast(pl.Int32),
        pl.col("trips_start").cast(pl.Datetime),
        pl.col("lat").cast(pl.Float64),
        pl.col("lon").cast(pl.Float64),
    ])

    # ---- Extract unique station_id/lat/lon mapping
    # This collect is small (number of stations), not 7M rows.
    station_map = (
        bike
        .select(["station_id", "lat", "lon"])
        .unique()
        .collect()
        .to_pandas()
    )

    if station_sample_limit is not None:
        station_map = station_map.head(station_sample_limit)

    # ---- Compute weather_cell_id + representative cell center coords
    station_map["weather_cell_id"] = [
        grid_cell_id(lat, lon, grid_km) for lat, lon in zip(station_map["lat"], station_map["lon"])
    ]
    centers = [grid_cell_center(lat, lon, grid_km) for lat, lon in zip(station_map["lat"], station_map["lon"])]
    station_map["cell_lat"] = [c[0] for c in centers]
    station_map["cell_lon"] = [c[1] for c in centers]

    # ---- Unique cells
    cells = station_map[["weather_cell_id", "cell_lat", "cell_lon"]].drop_duplicates()

    # ---- Weather per cell (cached)
    weather_parts = []
    for _, r in cells.iterrows():
        cid = str(r["weather_cell_id"])
        w = load_or_fetch_weather_for_cell(
            cell_id=cid,
            cell_lat=float(r["cell_lat"]),
            cell_lon=float(r["cell_lon"]),
            start_date=start_date,
            end_date=end_date,
            timezone=timezone,
            cache_dir=cache_dir,
            hourly_vars=hourly_vars,
            polite_sleep_s=polite_sleep_s,
        )
        weather_parts.append(w)

    weather = pd.concat(weather_parts, ignore_index=True)
    weather["trips_start"] = pd.to_datetime(weather["trips_start"])

    # ---- Bank holidays
    bank_holidays = fetch_uk_bank_holidays()
    bh_set: Set = set(bank_holidays["date"].tolist())

    # ---- Daylight (optional)
    daylight_daily = None
    if include_daylight:
        daylight_daily = build_daylight_daily_by_cell(
            cells_df=cells,
            start_date=start_date,
            end_date=end_date,
            timezone=timezone,
        )
        if daylight_daily is None:
            include_daylight = False

    # ---- Convert lookup tables to Polars
    station_map_pl = pl.from_pandas(station_map[["station_id", "weather_cell_id"]])

    weather_pl = pl.from_pandas(weather).with_columns(pl.col("trips_start").cast(pl.Datetime))

    if include_daylight and daylight_daily is not None:
        daylight_pl = pl.from_pandas(daylight_daily).with_columns([
            pl.col("sunrise").cast(pl.Datetime),
            pl.col("sunset").cast(pl.Datetime),
        ])

    # ---- Join weather_cell_id onto the big panel by station_id (no lat/lon join!)
    df = bike.join(station_map_pl.lazy(), on="station_id", how="left")

    # ---- Join weather on (weather_cell_id, trips_start)
    df = df.join(weather_pl.lazy(), on=["weather_cell_id", "trips_start"], how="left")

    # ---- Calendar features
    df = df.with_columns([
        pl.col("trips_start").dt.date().alias("date"),
        pl.col("trips_start").dt.hour().alias("hour"),
        pl.col("trips_start").dt.weekday().alias("dow"),
        pl.col("trips_start").dt.month().alias("month"),
        pl.col("trips_start").dt.year().alias("year"),
        pl.col("trips_start").dt.ordinal_day().alias("doy"),

        (pl.col("trips_start").dt.weekday() >= 5).cast(pl.Int8).alias("is_weekend"),
        pl.col("trips_start").dt.hour().is_in([7, 8, 9]).cast(pl.Int8).alias("is_am_peak"),
        pl.col("trips_start").dt.hour().is_in([16, 17, 18, 19]).cast(pl.Int8).alias("is_pm_peak"),
        (pl.col("trips_start").dt.hour().cast(pl.Utf8) + "_" + pl.col("trips_start").dt.weekday().cast(pl.Utf8)).alias("hour_dow"),

        pl.col("ts").log1p().alias("y_log1p"),
    ])

    # ---- Bank holiday flag
    bh_dates = pl.Series("bh_dates", sorted(list(bh_set)))
    df = df.with_columns([
        pl.col("date").is_in(bh_dates).cast(pl.Int8).alias("is_bank_holiday")
    ])

    # ---- Daylight join + flag
    if include_daylight and daylight_daily is not None:
        df = df.join(daylight_pl.lazy(), on=["weather_cell_id", "date"], how="left").with_columns([
            ((pl.col("trips_start") >= pl.col("sunrise")) & (pl.col("trips_start") < pl.col("sunset")))
            .cast(pl.Int8).alias("is_daylight"),
        ])

    # ---- Select output columns (keep existing lat/lon/station_name from the panel)
    base_cols = [
        "station_id", "trips_start", "date", "ts", "y_log1p", "strike_exposed",
        "lat", "lon",
        "station_name",  # keep if present
        "weather_cell_id",
        "hour", "dow", "month", "year", "doy",
        "is_weekend", "is_am_peak", "is_pm_peak", "hour_dow",
        "is_bank_holiday",
        *hourly_vars,
    ]
    if include_daylight:
        base_cols += ["sunrise", "sunset", "is_daylight"]

    existing = set(df.schema.keys())
    select_cols = [c for c in base_cols if c in existing]

    out = df.select(select_cols).collect(streaming=streaming_collect)

    if out_path is not None:
        out.write_parquet(out_path)

    return out


# -----------------------------
# Example usage (copy into notebook)
# -----------------------------
# out = build_bike_panel_with_covariates_from_panel(
#     bike_path="bike_hourly.parquet",
#     out_path="bike_hourly_with_covariates.parquet",
#     grid_km=1.0,
#     include_daylight=True,  # pip install astral pytz
#     start_date="2016-01-01",
#     end_date="2018-12-31",
# )
# out.head()