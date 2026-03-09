# covariate_join_utils.py
#
# Station-hour panel enrichment pipeline for TfL bike causal analysis.
#
# Covariates added:
#   Weather    : temperature, humidity, precipitation, rain, cloud cover,
#                wind speed, weather code (Open-Meteo archive, cached)
#   Calendar   : hour, dow, month, year, doy, is_weekend, is_am_peak,
#                is_pm_peak, is_bank_holiday, is_school_holiday
#   Daylight   : sunrise, sunset, is_daylight  (optional; requires astral + pytz)
#   Tube proximity (station-level):
#                dist_nearest_tube_km, n_tube_within_500m, n_tube_within_1km
#   Strike context (date-level):
#                strike_severity_daily_frac, days_to_next_strike,
#                days_since_last_strike
#   Cycle infra (station-level, optional):
#                cycle_infra_score  (OSM; run fetch_osm_cycle_lanes() once first)

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import polars as pl
import requests


# ── Grid clustering ───────────────────────────────────────────────────────────

def _km_to_deg_lat(km: float) -> float:
    return km / 111.32

def _km_to_deg_lon(km: float, lat_deg: float) -> float:
    return km / (111.32 * max(1e-6, math.cos(math.radians(lat_deg))))

def grid_cell_id(lat: float, lon: float, grid_km: float) -> str:
    """Stable rectangular grid bin id; lon-step varies with latitude."""
    lat_step = _km_to_deg_lat(grid_km)
    lon_step = _km_to_deg_lon(grid_km, lat)
    lat_idx  = math.floor(lat / lat_step)
    lon_idx  = math.floor(lon / lon_step)
    return f"g{grid_km:.3f}_lat{lat_idx}_lon{lon_idx}"

def grid_cell_center(lat: float, lon: float, grid_km: float) -> Tuple[float, float]:
    lat_step = _km_to_deg_lat(grid_km)
    lon_step = _km_to_deg_lon(grid_km, lat)
    lat_idx  = math.floor(lat / lat_step)
    lon_idx  = math.floor(lon / lon_step)
    return (lat_idx + 0.5) * lat_step, (lon_idx + 0.5) * lon_step


# ── Haversine distance ────────────────────────────────────────────────────────

def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray,
) -> np.ndarray:
    """Vectorised haversine distance in kilometres."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return R * 2 * np.arcsin(np.sqrt(a))


# ── Tube station coordinates ──────────────────────────────────────────────────
# Central London tube stops most relevant to the Santander Bikes footprint.
# Extend this list if your panel covers outer zones.

_TUBE_STATIONS = pd.DataFrame([
    {"lat": 51.5143, "lon": -0.0755},   # Aldgate
    {"lat": 51.5322, "lon": -0.1058},   # Angel
    {"lat": 51.5226, "lon": -0.1571},   # Baker Street
    {"lat": 51.5133, "lon": -0.0886},   # Bank
    {"lat": 51.5204, "lon": -0.0979},   # Barbican
    {"lat": 51.5272, "lon": -0.0549},   # Bethnal Green
    {"lat": 51.5142, "lon": -0.1494},   # Bond Street
    {"lat": 51.5113, "lon": -0.0904},   # Cannon Street
    {"lat": 51.5051, "lon": -0.0209},   # Canary Wharf
    {"lat": 51.5185, "lon": -0.1111},   # Chancery Lane
    {"lat": 51.5080, "lon": -0.1247},   # Charing Cross
    {"lat": 51.5145, "lon": -0.1037},   # City Thameslink
    {"lat": 51.4943, "lon": -0.1001},   # Elephant & Castle
    {"lat": 51.5074, "lon": -0.1223},   # Embankment
    {"lat": 51.5282, "lon": -0.1337},   # Euston
    {"lat": 51.5261, "lon": -0.1350},   # Euston Square
    {"lat": 51.5203, "lon": -0.1050},   # Farringdon
    {"lat": 51.5199, "lon": -0.1344},   # Goodge Street
    {"lat": 51.5174, "lon": -0.1199},   # Holborn
    {"lat": 51.5027, "lon": -0.1527},   # Hyde Park Corner
    {"lat": 51.5308, "lon": -0.1238},   # Kings Cross
    {"lat": 51.5014, "lon": -0.1607},   # Knightsbridge
    {"lat": 51.4989, "lon": -0.1116},   # Lambeth North
    {"lat": 51.5113, "lon": -0.1281},   # Leicester Square
    {"lat": 51.5178, "lon": -0.0823},   # Liverpool Street
    {"lat": 51.5055, "lon": -0.0861},   # London Bridge
    {"lat": 51.5136, "lon": -0.1586},   # Marble Arch
    {"lat": 51.5186, "lon": -0.0886},   # Moorgate
    {"lat": 51.5107, "lon": -0.0863},   # Monument
    {"lat": 51.5263, "lon": -0.0873},   # Old Street
    {"lat": 51.5154, "lon": -0.1417},   # Oxford Circus
    {"lat": 51.5154, "lon": -0.1755},   # Paddington
    {"lat": 51.4893, "lon": -0.1334},   # Pimlico
    {"lat": 51.5108, "lon": -0.1873},   # Queensway
    {"lat": 51.5232, "lon": -0.1244},   # Russell Square
    {"lat": 51.4994, "lon": -0.1335},   # St James Park
    {"lat": 51.5146, "lon": -0.0973},   # St Pauls
    {"lat": 51.5238, "lon": -0.0755},   # Shoreditch High St
    {"lat": 51.4924, "lon": -0.1565},   # Sloane Square
    {"lat": 51.4941, "lon": -0.1738},   # South Kensington
    {"lat": 51.5041, "lon": -0.1052},   # Southwark
    {"lat": 51.5111, "lon": -0.1141},   # Temple
    {"lat": 51.5165, "lon": -0.1308},   # Tottenham Court Rd
    {"lat": 51.5098, "lon": -0.0766},   # Tower Hill
    {"lat": 51.4861, "lon": -0.1245},   # Vauxhall
    {"lat": 51.4965, "lon": -0.1447},   # Victoria
    {"lat": 51.5243, "lon": -0.1388},   # Warren Street
    {"lat": 51.5036, "lon": -0.1143},   # Waterloo
    {"lat": 51.5010, "lon": -0.1254},   # Westminster
])


def _build_tube_proximity_features(station_map: pd.DataFrame) -> pd.DataFrame:
    """
    Per bike station: distance to nearest tube stop + counts within 500m / 1km.

    Causal role: dist_nearest_tube_km is the dominant driver of positivity —
    stations far from any tube line have P(T=1|X) ≈ 0 by construction, making
    them uninformative for CATE estimation. It also proxies commuter demand
    density, a direct confounder of Y.

    Operates on the small unique-station map, not the full 7M-row panel.
    """
    tube_lats = _TUBE_STATIONS["lat"].values
    tube_lons = _TUBE_STATIONS["lon"].values

    records = []
    for _, row in station_map.iterrows():
        dists = _haversine_km(row["lat"], row["lon"], tube_lats, tube_lons)
        records.append({
            "station_id":           row["station_id"],
            "dist_nearest_tube_km": float(dists.min()),
            "n_tube_within_500m":   int((dists <= 0.5).sum()),
            "n_tube_within_1km":    int((dists <= 1.0).sum()),
        })
    return pd.DataFrame(records)


# ── School holidays ───────────────────────────────────────────────────────────
# Inner London state school holiday periods (GOV.UK term dates).
# Extend if your panel runs beyond 2020.

_LONDON_SCHOOL_HOLIDAYS: List[Tuple[str, str]] = [
    # 2016
    ("2016-01-01", "2016-01-05"), ("2016-03-28", "2016-04-08"),
    ("2016-05-30", "2016-06-03"), ("2016-07-22", "2016-09-05"),
    ("2016-10-24", "2016-10-28"), ("2016-12-19", "2016-12-31"),
    # 2017
    ("2017-01-01", "2017-01-03"), ("2017-02-13", "2017-02-17"),
    ("2017-04-03", "2017-04-21"), ("2017-05-29", "2017-06-02"),
    ("2017-07-21", "2017-09-04"), ("2017-10-23", "2017-10-27"),
    ("2017-12-18", "2017-12-31"),
    # 2018
    ("2018-01-01", "2018-01-01"), ("2018-02-12", "2018-02-16"),
    ("2018-03-30", "2018-04-13"), ("2018-05-28", "2018-06-01"),
    ("2018-07-24", "2018-09-03"), ("2018-10-29", "2018-11-02"),
    ("2018-12-17", "2018-12-31"),
    # 2019
    ("2019-01-01", "2019-01-01"), ("2019-02-18", "2019-02-22"),
    ("2019-04-12", "2019-04-26"), ("2019-05-27", "2019-05-31"),
    ("2019-07-23", "2019-09-02"), ("2019-10-28", "2019-11-01"),
    ("2019-12-23", "2019-12-31"),
    # 2020 (includes national lockdown)
    ("2020-01-01", "2020-01-01"), ("2020-02-17", "2020-02-21"),
    ("2020-03-23", "2020-06-01"), ("2020-07-22", "2020-09-02"),
    ("2020-10-26", "2020-10-30"), ("2020-12-18", "2020-12-31"),
]

def _build_school_holiday_set(start_date: str, end_date: str) -> Set:
    """
    Expand holiday ranges into a set of datetime.date objects within the panel window.

    Causal role: school holidays reshape commuter flows and are correlated with
    strike timing (unions sometimes target school holidays for maximum disruption).
    Omitting them violates unconfoundedness.
    """
    panel_start = pd.Timestamp(start_date).date()
    panel_end   = pd.Timestamp(end_date).date()
    out: Set = set()
    for s, e in _LONDON_SCHOOL_HOLIDAYS:
        for d in pd.date_range(s, e, freq="D"):
            d = d.date()
            if panel_start <= d <= panel_end:
                out.add(d)
    return out


# ── Strike severity + anticipation ────────────────────────────────────────────

def _build_strike_date_features(
    bike_lf:    pl.LazyFrame,
    start_date: str,
    end_date:   str,
) -> pd.DataFrame:
    """
    Build a date-level DataFrame with three strike context features:

    strike_severity_daily_frac
        Fraction of all stations that are strike-exposed on each date.
        Scales with the number of lines striking simultaneously; a 1-line
        strike affects ~10% of stations, a 4-line strike ~40%.
        Causal role: a proxy for treatment dose / hidden treatment versions,
        helping satisfy consistency (no hidden versions of treatment).

    days_to_next_strike  (capped at 30)
        Days until the next known strike date.
        Causal role: captures anticipation effects. Announced strikes cause
        behaviour change (e.g. pre-buying bike day passes) *before* T=1,
        contaminating nearby control observations. Including this variable
        makes the no-anticipation component of unconfoundedness more plausible.

    days_since_last_strike  (capped at 30)
        Days since the most recent strike.
        Causal role: captures recovery / habit-formation spillovers. Cyclists
        who discovered bikes during a strike may persist for several days after.
        Without this, post-strike control observations are contaminated.
    """
    # Daily severity — small collect (~1 row per day, not 7M rows)
    daily = (
        bike_lf
        .select(["trips_start", "strike_exposed"])
        .with_columns(pl.col("trips_start").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.col("strike_exposed").mean().alias("strike_severity_daily_frac"))
        .collect()
        .to_pandas()
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.date

    # Ensure every date in the panel window has a row (fill non-strike days with 0)
    all_dates = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D").date})
    daily = all_dates.merge(daily, on="date", how="left").fillna(
        {"strike_severity_daily_frac": 0.0}
    )

    strike_days = set(daily.loc[daily["strike_severity_daily_frac"] > 0, "date"])

    def _days_to_next(d):
        future = sorted(s for s in strike_days if s > d)
        return (future[0] - d).days if future else 30

    def _days_since_last(d):
        past = sorted((s for s in strike_days if s < d), reverse=True)
        return (d - past[0]).days if past else 30

    daily["days_to_next_strike"]    = daily["date"].map(_days_to_next).clip(upper=30)
    daily["days_since_last_strike"] = daily["date"].map(_days_since_last).clip(upper=30)

    return daily


# ── Cycle infrastructure (optional) ──────────────────────────────────────────

def fetch_osm_cycle_lanes(
    bbox: Tuple[float, float, float, float] = (51.28, -0.51, 51.70, 0.33),
    out_path: str = "osm_cycle_lanes.json",
) -> None:
    """
    Download OSM cycle lane data for London and save as JSON.
    Run once before calling the main pipeline with osm_json_path.

    Causal role of cycle_infra_score: areas with denser cycling infrastructure
    have higher substitution elasticity — commuters there are more likely to
    switch to bikes when the tube is disrupted. This is a confounder because
    cycle infrastructure is also denser in inner London where tube coverage
    (and strike exposure) is highest. Omitting it biases CATE estimates upward.
    """
    south, west, north, east = bbox
    query = f"""
    [out:json][timeout:120];
    (
      way["highway"="cycleway"]({south},{west},{north},{east});
      way["cycleway"~"lane|track|opposite_lane"]({south},{west},{north},{east});
      way["bicycle"="designated"]({south},{west},{north},{east});
    );
    out geom;
    """
    resp = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        timeout=180,
    )
    resp.raise_for_status()
    with open(out_path, "w") as f:
        json.dump(resp.json(), f)
    print(f"Saved {len(resp.json().get('elements', []))} OSM ways → {out_path}")


def _build_cycle_infra_features(
    station_map:   pd.DataFrame,
    osm_json_path: str,
    radius_km:     float = 0.5,
) -> pd.DataFrame:
    """
    Count OSM cycle lane geometry nodes within radius_km of each bike station
    using a KD-tree.  More nodes ≈ more / longer cycle lanes nearby.
    """
    from scipy.spatial import cKDTree

    with open(osm_json_path) as f:
        osm_data = json.load(f)

    nodes = [
        (node["lat"], node["lon"])
        for way in osm_data.get("elements", [])
        for node in way.get("geometry", [])
    ]

    if not nodes:
        return pd.DataFrame({
            "station_id":        station_map["station_id"].values,
            "cycle_infra_score": np.zeros(len(station_map), dtype=np.int32),
        })

    node_arr  = np.radians(np.array(nodes))
    tree      = cKDTree(node_arr)
    query_pts = np.radians(station_map[["lat", "lon"]].values)
    r_rad     = radius_km / 6371.0
    counts    = tree.query_ball_point(query_pts, r=r_rad, return_length=True)

    return pd.DataFrame({
        "station_id":        station_map["station_id"].values,
        "cycle_infra_score": counts.astype(np.int32),
    })


# ── Bank holidays ─────────────────────────────────────────────────────────────

def fetch_uk_bank_holidays() -> pd.DataFrame:
    """England & Wales bank holidays from the GOV.UK JSON endpoint."""
    r = requests.get("https://www.gov.uk/bank-holidays.json", timeout=30)
    r.raise_for_status()
    events = pd.DataFrame(r.json()["england-and-wales"]["events"])
    events["date"] = pd.to_datetime(events["date"]).dt.date
    return events[["date", "title"]]


# ── Weather (Open-Meteo archive) ──────────────────────────────────────────────

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DEFAULT_HOURLY_VARS: List[str] = [
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
    lat:         float
    lon:         float
    start_date:  str
    end_date:    str
    timezone:    str
    hourly_vars: List[str]


def fetch_open_meteo_hourly_with_retry(
    req:         WeatherRequest,
    max_retries: int   = 8,
    base_sleep:  float = 2.0,
    jitter:      float = 0.25,
) -> pd.DataFrame:
    """Fetch Open-Meteo hourly data with exponential backoff on 429 rate limits."""
    params = {
        "latitude":   req.lat,
        "longitude":  req.lon,
        "start_date": req.start_date,
        "end_date":   req.end_date,
        "timezone":   req.timezone,
        "hourly":     ",".join(req.hourly_vars),
    }
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=90)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after else base_sleep * (2 ** attempt)
                time.sleep(sleep_s * (1.0 + random.uniform(0, jitter)))
                continue
            r.raise_for_status()
            js = r.json()
            if "hourly" not in js or "time" not in js["hourly"]:
                raise RuntimeError(f"Unexpected Open-Meteo response: {json.dumps(js)[:600]}")
            w = pd.DataFrame(js["hourly"])
            w["trips_start"] = pd.to_datetime(w["time"])
            return w.drop(columns=["time"], errors="ignore")
        except Exception as e:
            last_err = e
            sleep_s = base_sleep * (2 ** attempt) * (1.0 + random.uniform(0, jitter))
            time.sleep(sleep_s)
    raise last_err


def load_or_fetch_weather_for_cell(
    cell_id:        str,
    cell_lat:       float,
    cell_lon:       float,
    start_date:     str,
    end_date:       str,
    timezone:       str,
    cache_dir:      str,
    hourly_vars:    List[str],
    polite_sleep_s: float,
) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"weather_{cell_id}_{start_date}_{end_date}.parquet"
    )

    if os.path.exists(cache_path):
        w = pd.read_parquet(cache_path)
        if "weather_cell_id" not in w.columns:
            w["weather_cell_id"] = cell_id
        return w

    req = WeatherRequest(
        lat=cell_lat, lon=cell_lon,
        start_date=start_date, end_date=end_date,
        timezone=timezone, hourly_vars=hourly_vars,
    )
    w = fetch_open_meteo_hourly_with_retry(req)
    w["weather_cell_id"] = cell_id

    tmp = cache_path + ".tmp"
    w.to_parquet(tmp, index=False)
    os.replace(tmp, cache_path)
    time.sleep(polite_sleep_s)
    return w


# ── Daylight (optional) ───────────────────────────────────────────────────────

def _build_daylight_daily_by_cell(
    cells_df:   pd.DataFrame,
    start_date: str,
    end_date:   str,
    timezone:   str,
) -> Optional[pd.DataFrame]:
    """
    Per (weather_cell_id, date): sunrise and sunset times.
    Returns None if astral / pytz are not installed.
    """
    try:
        from astral import LocationInfo
        from astral.sun import sun
        import pytz
    except ImportError:
        return None

    tz    = pytz.timezone(timezone)
    dates = pd.date_range(start_date, end_date, freq="D").date
    rows  = []

    for _, row in cells_df.iterrows():
        loc = LocationInfo(
            name=str(row["weather_cell_id"]),
            region="London",
            timezone=timezone,
            latitude=float(row["cell_lat"]),
            longitude=float(row["cell_lon"]),
        )
        for d in dates:
            s = sun(loc.observer, date=d, tzinfo=tz)
            rows.append({
                "weather_cell_id": row["weather_cell_id"],
                "date":            d,
                "sunrise":         s["sunrise"],
                "sunset":          s["sunset"],
            })
    return pd.DataFrame(rows)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_bike_panel_with_covariates_from_panel(
    bike_path:            str,
    out_path:             Optional[str]       = None,
    *,
    bike_format:          str                 = "parquet",
    cache_dir:            str                 = "cache_weather",
    start_date:           str                 = "2016-01-01",
    end_date:             str                 = "2018-12-31",
    timezone:             str                 = "Europe/London",
    grid_km:              float               = 1.0,
    hourly_vars:          Optional[List[str]] = None,
    include_daylight:     bool                = True,
    osm_json_path:        Optional[str]       = None,
    polite_sleep_s:       float               = 0.25,
    station_sample_limit: Optional[int]       = None,
) -> Optional[pl.DataFrame]:
    """
    Build a feature-enriched station-hour panel from a bike rides parquet/CSV.

    Required input columns:
        station_id, trips_start, ts, strike_exposed, lat, lon

    Optional input columns:
        station_name

    Parameters
    ----------
    bike_path            : Path to the input bike panel file.
    out_path             : If provided, streams the output to this parquet path
                           (recommended for large panels; returns None).
                           If None, returns a 100k-row in-memory sample.
    bike_format          : "parquet" or "csv".
    cache_dir            : Directory for cached weather parquet files.
    start_date / end_date: Panel date range (used for weather fetching and
                           strike feature computation).
    timezone             : IANA timezone string for weather and daylight.
    grid_km              : Grid resolution for weather cell clustering (km).
    hourly_vars          : Open-Meteo variables to fetch. Defaults to
                           DEFAULT_HOURLY_VARS.
    include_daylight     : Add sunrise/sunset/is_daylight (requires astral+pytz).
    osm_json_path        : Path to a pre-downloaded OSM cycle lane JSON file.
                           Run fetch_osm_cycle_lanes() once to create it.
                           If None, cycle_infra_score is omitted.
    polite_sleep_s       : Pause between Open-Meteo requests (rate limit courtesy).
    station_sample_limit : Restrict to the first N stations (useful for dev/testing).
    """
    if hourly_vars is None:
        hourly_vars = DEFAULT_HOURLY_VARS

    # ── Load bike panel ───────────────────────────────────────────────────────
    if bike_format == "parquet":
        bike = pl.scan_parquet(bike_path)
    elif bike_format == "csv":
        bike = pl.scan_csv(bike_path, try_parse_dates=True)
    else:
        raise ValueError("bike_format must be 'parquet' or 'csv'")

    bike = bike.with_columns([
        pl.col("station_id").cast(pl.Int32),
        pl.col("strike_exposed").cast(pl.Int8),
        pl.col("ts").cast(pl.Int32),
        pl.col("trips_start").cast(pl.Datetime),
        pl.col("lat").cast(pl.Float64),
        pl.col("lon").cast(pl.Float64),
    ])

    # ── Station map (small collect — unique stations only) ────────────────────
    station_map = (
        bike.select(["station_id", "lat", "lon"])
        .unique()
        .collect()
        .to_pandas()
    )
    if station_sample_limit is not None:
        station_map = station_map.head(station_sample_limit)

    # ── Weather grid cells ────────────────────────────────────────────────────
    station_map["weather_cell_id"] = [
        grid_cell_id(lat, lon, grid_km)
        for lat, lon in zip(station_map["lat"], station_map["lon"])
    ]
    centers = [
        grid_cell_center(lat, lon, grid_km)
        for lat, lon in zip(station_map["lat"], station_map["lon"])
    ]
    station_map["cell_lat"] = [c[0] for c in centers]
    station_map["cell_lon"] = [c[1] for c in centers]
    cells = station_map[["weather_cell_id", "cell_lat", "cell_lon"]].drop_duplicates()

    # ── Fetch / cache weather per cell ────────────────────────────────────────
    for _, r in cells.iterrows():
        load_or_fetch_weather_for_cell(
            cell_id=str(r["weather_cell_id"]),
            cell_lat=float(r["cell_lat"]),
            cell_lon=float(r["cell_lon"]),
            start_date=start_date, end_date=end_date,
            timezone=timezone, cache_dir=cache_dir,
            hourly_vars=hourly_vars, polite_sleep_s=polite_sleep_s,
        )

    # ── Tube proximity (station-level, static) ────────────────────────────────
    tube_prox  = _build_tube_proximity_features(station_map)
    station_map = station_map.merge(tube_prox, on="station_id", how="left")

    # ── Cycle infrastructure (station-level, optional) ────────────────────────
    if osm_json_path is not None:
        cycle_infra = _build_cycle_infra_features(station_map, osm_json_path)
        station_map = station_map.merge(cycle_infra, on="station_id", how="left")

    # ── Strike severity + anticipation (date-level) ───────────────────────────
    strike_dates_df = _build_strike_date_features(bike, start_date, end_date)

    # ── Bank holidays (date-level) ────────────────────────────────────────────
    bank_holidays = fetch_uk_bank_holidays()
    bh_set = set(bank_holidays["date"].tolist())

    # ── School holidays (date-level) ──────────────────────────────────────────
    school_holiday_set = _build_school_holiday_set(start_date, end_date)

    # ── Daylight (optional) ───────────────────────────────────────────────────
    daylight_daily = None
    if include_daylight:
        daylight_daily = _build_daylight_daily_by_cell(cells, start_date, end_date, timezone)
        if daylight_daily is None:
            include_daylight = False

    # ── Convert all lookup tables to Polars lazy ──────────────────────────────
    station_level_cols = [
        "station_id", "weather_cell_id",
        "dist_nearest_tube_km", "n_tube_within_500m", "n_tube_within_1km",
        *(["cycle_infra_score"] if osm_json_path else []),
    ]
    station_map_pl = pl.from_pandas(station_map[station_level_cols]).lazy()

    strike_pl = (
        pl.from_pandas(strike_dates_df)
        .lazy()
        .with_columns(pl.col("date").cast(pl.Date))
    )

    weather_lf = (
        pl.scan_parquet(
            os.path.join(cache_dir, f"weather_*_{start_date}_{end_date}.parquet")
        )
        .with_columns(pl.col("trips_start").cast(pl.Datetime))
        .unique(subset=["weather_cell_id", "trips_start"])
    )

    if include_daylight:
        daylight_pl = pl.from_pandas(daylight_daily).lazy().with_columns([
            pl.col("sunrise").cast(pl.Datetime),
            pl.col("sunset").cast(pl.Datetime),
        ])

    # ── Assemble lazy pipeline ────────────────────────────────────────────────
    df = bike.join(station_map_pl, on="station_id", how="left", validate="m:1")
    df = df.join(weather_lf, on=["weather_cell_id", "trips_start"], how="left", validate="m:1")

    # Calendar features
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
        pl.col("ts").log1p().alias("y_log1p"),
    ])

    # Bank holiday flag
    bh_series = pl.Series("bh_dates", sorted(bh_set))
    df = df.with_columns(
        pl.col("date").is_in(bh_series).cast(pl.Int8).alias("is_bank_holiday")
    )

    # School holiday flag
    sh_series = pl.Series("sh_dates", sorted(school_holiday_set))
    df = df.with_columns(
        pl.col("date").is_in(sh_series).cast(pl.Int8).alias("is_school_holiday")
    )

    # Strike severity + anticipation join (date-level)
    df = df.join(strike_pl, on="date", how="left")

    # Daylight join + flag
    if include_daylight:
        df = df.join(daylight_pl, on=["weather_cell_id", "date"], how="left")
        df = df.with_columns(
            ((pl.col("trips_start") >= pl.col("sunrise")) &
             (pl.col("trips_start") <  pl.col("sunset")))
            .cast(pl.Int8).alias("is_daylight")
        )

    # ── Final column selection ────────────────────────────────────────────────
    output_cols = [
        # Identifiers
        "station_id", "trips_start", "date",
        # Outcome
        "ts", "y_log1p",
        # Treatment
        "strike_exposed",
        # Station metadata
        "lat", "lon", "station_name", "weather_cell_id",
        # Calendar
        "hour", "dow", "month", "year", "doy",
        "is_weekend", "is_am_peak", "is_pm_peak",
        "is_bank_holiday", "is_school_holiday",
        # Weather
        *hourly_vars,
        # Tube proximity (NEW)
        "dist_nearest_tube_km", "n_tube_within_500m", "n_tube_within_1km",
        # Strike context (NEW)
        "strike_severity_daily_frac", "days_to_next_strike", "days_since_last_strike",
        # Optional
        *(["cycle_infra_score"] if osm_json_path else []),
        *(["sunrise", "sunset", "is_daylight"] if include_daylight else []),
    ]

    df_out = df.select(output_cols)

    if out_path is not None:
        df_out.sink_parquet(out_path, compression="zstd")
        return None

    # No out_path: return a sample (avoids accidentally collecting 7M rows)
    return df_out.limit(100_000).collect()


# ── Example usage ─────────────────────────────────────────────────────────────
#
# Step 1 — download OSM cycle lane data (run once, takes ~60s):
#   from covariate_join_utils import fetch_osm_cycle_lanes
#   fetch_osm_cycle_lanes(out_path="osm_cycle_lanes.json")
#
# Step 2 — build the panel:
#   from covariate_join_utils import build_bike_panel_with_covariates_from_panel
#   build_bike_panel_with_covariates_from_panel(
#       bike_path     = "bike_hourly.parquet",
#       out_path      = "bike_hourly_with_covariates.parquet",
#       start_date    = "2016-01-01",
#       end_date      = "2020-12-31",
#       grid_km       = 1.0,
#       osm_json_path = "osm_cycle_lanes.json",   # omit to skip cycle_infra_score
#       include_daylight = True,                   # pip install astral pytz
#   )
