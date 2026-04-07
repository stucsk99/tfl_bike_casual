"""
tfl_utils.py

Core utilities for the TfL cycle hire × tube strike causal analysis project.

Pipeline overview:
  1. Aggregate raw TfL journey CSVs into a station-hour panel
  2. Load bike station coordinates from the TfL BikePoint API
  3. Fetch tube station locations and served Underground lines from TfL StopPoint API
  4. Build a spatial mapping: bike station → nearby tube stations → lines served
  5. Expand line-level strike events into daily indicators
  6. Attach binary strike exposure to the station-hour panel
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import requests


# ── Type aliases ──────────────────────────────────────────────────────────────

AggSide = Literal["start", "end", "both"]
Freq    = Literal["h", "d"]


# ── Column name normalisation helpers ─────────────────────────────────────────

def _norm(s: str) -> str:
    """Strip non-alphanumeric characters and lowercase — makes column matching schema-agnostic."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _pick_col(cols: list[str], candidates: list[str]) -> str | None:
    """Return the first column whose normalised name matches a candidate, or None."""
    norm_map = {_norm(c): c for c in cols}
    for candidate in candidates:
        if candidate in norm_map:
            return norm_map[candidate]
    return None


# Column name variants across TfL data releases
_START_ID = ["startstationid", "start station id", "startstationnumber", "startstationlogicalterminal"]
_START_DT = ["startdate", "startdatetime", "starttime"]
_END_ID   = ["endstationid",   "end station id",   "endstationnumber",   "endstationlogicalterminal"]
_END_DT   = ["enddate",   "enddatetime",   "endtime"]

# TfL Underground line IDs (excludes DLR, Overground, Elizabeth line)
UNDERGROUND_LINES = {
    "bakerloo", "central", "circle", "district",
    "hammersmith-city", "jubilee", "metropolitan",
    "northern", "piccadilly", "victoria", "waterloo-city",
}


# ── 1. Journey aggregation ────────────────────────────────────────────────────

def _looks_like_text_csv(path: Path, nbytes: int = 2048) -> bool:
    """Return True if file bytes look like a text CSV rather than binary XLS."""
    with open(path, "rb") as f:
        head = f.read(nbytes)
    if b"\x00" in head:
        return False
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = head.decode("latin-1")
        except UnicodeDecodeError:
            return False
    return ("," in text) and ("\n" in text or "\r" in text)


def _aggregate_dataframe(
    df:             pd.DataFrame,
    side:           AggSide,
    start_id_col:   str | None,
    start_dt_col:   str | None,
    end_id_col:     str | None,
    end_dt_col:     str | None,
    freq:           str,
    dayfirst:       bool,
    max_station_id: int | None,
) -> pd.DataFrame:
    """Aggregate a single DataFrame chunk into station-time trip counts."""
    parts = []

    for id_col, dt_col, trip_col, active in [
        (start_id_col, start_dt_col, "trips_start", side in ("start", "both")),
        (end_id_col,   end_dt_col,   "trips_end",   side in ("end",   "both")),
    ]:
        if not active or id_col is None or dt_col is None:
            continue

        tmp = df[[id_col, dt_col]].dropna().copy()
        tmp["station_id"] = pd.to_numeric(tmp[id_col], errors="coerce")
        tmp["ts"]         = pd.to_datetime(tmp[dt_col], dayfirst=dayfirst, errors="coerce")
        tmp = tmp.dropna(subset=["station_id", "ts"])

        if max_station_id is not None:
            tmp = tmp[tmp["station_id"] <= max_station_id]

        tmp["station_id"] = tmp["station_id"].astype("int32")
        tmp["ts"]         = tmp["ts"].dt.floor(freq)
        parts.append(
            tmp.groupby(["station_id", "ts"], as_index=False)
               .size()
               .rename(columns={"size": trip_col})
        )

    if not parts:
        return pd.DataFrame(columns=["station_id", "ts"])

    result = parts[0]
    for p in parts[1:]:
        result = result.merge(p, on=["station_id", "ts"], how="outer")
    return result


def aggregate_one_file_to_parquet(
    path:           str | Path,
    out_dir:        str | Path,
    *,
    freq:           str       = "h",
    side:           AggSide   = "start",
    chunksize:      int       = 120_000,
    dayfirst:       bool      = True,
    sheet_name:     int | str = 0,
    max_station_id: int | None = 5000,
) -> Path | None:
    """
    Aggregate one journey file (CSV or XLS) into a station-time parquet.

    One parquet file is written per input file into out_dir.
    Returns the output path, or None if the file could not be processed.

    Parameters
    ----------
    path           : Path to a single TfL journey CSV or XLS file.
    out_dir        : Directory to write the output parquet.
    freq           : Time floor — "h" for hourly, "d" for daily.
    side           : Which journey endpoint to count — "start", "end", or "both".
    chunksize      : Rows per chunk when reading CSV (keeps memory bounded).
    max_station_id : Discard station IDs above this threshold (removes legacy logical
                     terminal IDs that crept into some data releases).
    """
    path    = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext           = path.suffix.lower()
    treat_as_csv  = (ext == ".csv") or (ext == ".xls" and _looks_like_text_csv(path))

    header  = (pd.read_csv(path, nrows=0)         if treat_as_csv
               else pd.read_excel(path, sheet_name=sheet_name, nrows=0, engine="xlrd"))
    cols    = list(header.columns)

    start_id = _pick_col(cols, _START_ID)
    start_dt = _pick_col(cols, _START_DT)
    end_id   = _pick_col(cols, _END_ID)
    end_dt   = _pick_col(cols, _END_DT)

    # Build usecols list — only load columns we actually need
    usecols = []
    if side in ("start", "both") and start_id and start_dt:
        usecols += [start_id, start_dt]
    if side in ("end",   "both") and end_id   and end_dt:
        usecols += [end_id, end_dt]
    usecols = list(dict.fromkeys(usecols))
    if not usecols:
        return None

    agg_kwargs = dict(
        side=side, start_id_col=start_id, start_dt_col=start_dt,
        end_id_col=end_id, end_dt_col=end_dt,
        freq=freq, dayfirst=dayfirst, max_station_id=max_station_id,
    )

    parts = []
    if treat_as_csv:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=True):
            agg = _aggregate_dataframe(chunk, **agg_kwargs)
            if len(agg):
                parts.append(agg)
    else:
        df  = pd.read_excel(path, sheet_name=sheet_name, usecols=usecols, engine="xlrd")
        agg = _aggregate_dataframe(df, **agg_kwargs)
        if len(agg):
            parts.append(agg)

    if not parts:
        return None

    result   = pd.concat(parts, ignore_index=True)
    sum_cols = [c for c in ["trips_start", "trips_end"] if c in result.columns]
    result   = result.groupby(["station_id", "ts"], as_index=False)[sum_cols].sum()
    for c in sum_cols:
        result[c] = result[c].fillna(0).astype("int32")

    out_path = out_dir / f"{path.stem}.parquet"
    result.to_parquet(out_path, index=False)
    return out_path


def aggregate_folder_to_parquet(
    folder_path: str | Path,
    out_dir:     str | Path,
    **kwargs,
) -> list[Path]:
    """
    Aggregate every CSV/XLS in a folder into per-file parquets, then merge.

    Returns a list of successfully written parquet paths.
    This is the recommended entry point — it processes files one at a time,
    so peak memory is bounded to a single file rather than the full dataset.
    """
    folder = Path(folder_path)
    files  = sorted(list(folder.rglob("*.csv")) + list(folder.rglob("*.xls")))
    if not files:
        raise ValueError(f"No CSV/XLS files found under: {folder.resolve()}")

    print(f"Found {len(files)} files to process")
    written = []
    for f in files:
        print(f"  Processing {f.name} ...", end=" ")
        out = aggregate_one_file_to_parquet(f, out_dir, **kwargs)
        if out:
            written.append(out)
            print("✓")
        else:
            print("skipped (no usable columns)")
    return written


def combine_parquet_parts(parts_dir: str | Path) -> pd.DataFrame:
    """
    Concatenate per-file parquets from aggregate_folder_to_parquet and sum
    trip counts for any duplicate station-time rows (which can arise when
    multiple files cover overlapping date ranges).
    """
    parts = sorted(Path(parts_dir).glob("*.parquet"))
    if not parts:
        raise ValueError(f"No parquet files found in: {parts_dir}")

    acc = None
    for p in parts:
        chunk = pd.read_parquet(p)
        acc   = chunk if acc is None else pd.concat([acc, chunk], ignore_index=True)
        sum_cols = [c for c in ["trips_start", "trips_end"] if c in acc.columns]
        acc = acc.groupby(["station_id", "ts"], as_index=False)[sum_cols].sum()

    acc = acc.rename(columns={'ts': 'trips_start', 'trips_start': 'ts'})
    return acc.sort_values(["station_id", "trips_start"]).reset_index(drop=True)


# ── 2. Bike station locations ─────────────────────────────────────────────────

def load_bike_station_locations(bikepoints_json_path: str | Path) -> pd.DataFrame:
    """
    Parse a BikePoint JSON dump from the TfL API into a station locations table.

    Returns columns: station_id (int), lat, lon, station_name.

    The JSON can be obtained by calling:
        GET https://api.tfl.gov.uk/BikePoint
    and saving the response body to a file.
    """
    import json

    with open(bikepoints_json_path, encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    return pd.DataFrame({
        "station_id":   df["id"].astype(str).str.split("_").str[-1].astype(int),
        "lat":          pd.to_numeric(df["lat"], errors="coerce"),
        "lon":          pd.to_numeric(df["lon"], errors="coerce"),
        "station_name": df.get("commonName", pd.Series([None] * len(df))),
    }).dropna(subset=["lat", "lon"]).reset_index(drop=True)


# ── 3. Tube stations from FOI CSV ─────────────────────────────────────────────

# Mapping from the display names in the FOI LINES column to the affected_line
# format used in the strikes CSV (e.g. "central_line", "hammersmith-city_line").
# Only Underground lines are included — DLR, Overground, etc. are excluded.
_FOI_LINE_NAME_MAP: dict[str, str] = {
    "bakerloo":          "bakerloo_line",
    "central":           "central_line",
    "circle":            "circle_line",
    "district":          "district_line",
    "hammersmith & city":"hammersmith-city_line",
    "hammersmith and city":"hammersmith-city_line",
    "jubilee":           "jubilee_line",
    "metropolitan":      "metropolitan_line",
    "northern":          "northern_line",
    "piccadilly":        "piccadilly_line",
    "victoria":          "victoria_line",
    "waterloo & city":   "waterloo-city_line",
    "waterloo and city": "waterloo-city_line",
}


def load_tube_stations_foi(csv_path: str | Path) -> pd.DataFrame:
    """
    Load all 271 London Underground stations from the TfL FOI CSV
    (FOI-1451-1819, file Stations_20180921.csv).

    The CSV has columns: NAME, LINES, NETWORK, x (longitude), y (latitude).
    This function filters to London Underground only, parses the comma-separated
    LINES field, and returns one row per station-line pair — ready to pass
    directly to build_station_line_map().

    Returns columns: name, lat, lon, affected_line
    """
    df = pd.read_csv(csv_path)
    lu = df[df["NETWORK"] == "London Underground"].copy()

    rows = []
    for _, row in lu.iterrows():
        lat  = float(row["y"])
        lon  = float(row["x"])
        name = str(row["NAME"]).strip()

        # LINES is a comma-separated display string, e.g. "District, Circle"
        for raw_line in str(row["LINES"]).split(","):
            line_key     = raw_line.strip().lower()
            affected_line = _FOI_LINE_NAME_MAP.get(line_key)
            if affected_line is not None:
                rows.append({
                    "name":          name,
                    "lat":           lat,
                    "lon":           lon,
                    "affected_line": affected_line,
                })

    result = pd.DataFrame(rows).drop_duplicates()
    n_stations = result["name"].nunique()
    n_pairs    = len(result)
    print(f"Loaded {n_stations} Underground stations, {n_pairs} station-line pairs")
    return result


# ── 4. Spatial mapping ────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorised great-circle distance in metres."""
    R = 6_371_000.0
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def build_station_line_map(
    bike_stations:       pd.DataFrame,
    tube_stations_foi:   pd.DataFrame,
    *,
    radius_m:            float = 800.0,
    fallback_to_nearest: bool  = True,
) -> pd.DataFrame:
    """
    Build a bike station → Underground line exposure table using the TfL FOI
    station CSV (loaded via load_tube_stations_foi()).

    For each bike station we find all tube stations within radius_m and record
    which Underground lines they serve. If no tube station falls within the
    radius and fallback_to_nearest=True, the single nearest tube station is
    used regardless of distance — this ensures every bike station gets at least
    one line assignment.

    The 800m radius is a standard walking catchment consistent with TfL's own
    accessibility modelling (~10 minutes on foot).

    Parameters
    ----------
    bike_stations     : DataFrame with columns station_id, lat, lon.
    tube_stations_foi : Output of load_tube_stations_foi() — one row per
                        station-line pair with columns name, lat, lon, affected_line.
    radius_m          : Search radius in metres.
    fallback_to_nearest : If True, assign the nearest tube station when none
                          fall within radius_m.

    Returns
    -------
    DataFrame with columns: station_id, affected_line, tube_station_name, dist_m
        One row per (bike station, Underground line) pair.
        Where a line is served by multiple tube stations within radius_m,
        only the nearest is retained.
    """
    b_lats  = bike_stations["lat"].to_numpy()
    b_lons  = bike_stations["lon"].to_numpy()
    t_lats  = tube_stations_foi["lat"].to_numpy()
    t_lons  = tube_stations_foi["lon"].to_numpy()
    t_names = tube_stations_foi["name"].to_numpy()
    t_lines = tube_stations_foi["affected_line"].to_numpy()

    rows = []
    for bike_id, blat, blon in zip(bike_stations["station_id"], b_lats, b_lons):
        dists = haversine_m(blat, blon, t_lats, t_lons)
        idxs  = np.where(dists <= radius_m)[0]

        if len(idxs) == 0 and fallback_to_nearest:
            idxs = np.array([int(np.argmin(dists))])

        for j in idxs:
            rows.append({
                "station_id":        int(bike_id),
                "tube_station_name": t_names[j],
                "affected_line":     t_lines[j],
                "dist_m":            float(dists[j]),
            })

    if not rows:
        return pd.DataFrame(columns=["station_id", "tube_station_name",
                                     "affected_line", "dist_m"])

    result = (
        pd.DataFrame(rows)
        # Keep nearest tube station per (bike station, line) — eliminates
        # duplicate line entries when multiple stations on the same line
        # are within radius_m
        .sort_values(["station_id", "affected_line", "dist_m"])
        .drop_duplicates(subset=["station_id", "affected_line"], keep="first")
        .reset_index(drop=True)
    )

    n_bike    = result["station_id"].nunique()
    n_pairs   = len(result)
    n_lines   = result["affected_line"].nunique()
    print(f"Station-line map: {n_bike} bike stations × {n_lines} lines = {n_pairs} pairs")
    return result


# ── 5. Strike expansion ───────────────────────────────────────────────────────

def expand_strikes_daily(strike_data: pd.DataFrame) -> pd.DataFrame:
    """
    Expand strike events into daily rows.
    Normalises affected_line to use hyphens consistently
    (e.g. 'hammersmith_city_line' → 'hammersmith-city_line')
    to match the format produced by load_tube_stations_foi().
    """
    s = strike_data.copy()
    s["date_start"]    = pd.to_datetime(s["date_start"], dayfirst=True, errors="coerce")
    s["date_end"]      = pd.to_datetime(s["date_end"],   dayfirst=True, errors="coerce")
    s                  = s.dropna(subset=["date_start", "date_end", "affected_line"])

    # Normalise line names: replace underscores with hyphens, lowercase, strip
    s["affected_line"] = (
        s["affected_line"]
        .str.strip()
        .str.lower()
        .str.replace("_", "-", regex=False)
    )

    rows = []
    for row in s.itertuples(index=False):
        for d in pd.date_range(row.date_start.floor("D"), row.date_end.floor("D"), freq="D"):
            rows.append({"date": d, "affected_line": row.affected_line, "strike": 1})

    result = pd.DataFrame(rows).drop_duplicates()
    print(f"Strike-day-line records: {len(result)}")
    print(f"Lines covered: {sorted(result['affected_line'].unique())}")
    return result


# ── 6. Attach strike exposure ─────────────────────────────────────────────────

def attach_strikes_to_base(
    base:             pd.DataFrame,
    strikes_daily:    pd.DataFrame,
    station_line_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach a binary strike_exposed indicator to the station-hour panel.

    A station-hour is treated (strike_exposed = 1) if any Underground line
    serving that station is on strike on that day.

    base must have columns: station_id, trips_start (datetime), ts (numeric trip count).
    """
    df          = base.copy()
    df["date"]  = pd.to_datetime(df["trips_start"]).dt.floor("D")

    station_day_treat = (
        strikes_daily
        .merge(station_line_map[["station_id", "affected_line"]], on="affected_line", how="inner")
        .drop_duplicates(subset=["station_id", "date"])
        .assign(strike_exposed=1)
        [["station_id", "date", "strike_exposed"]]
    )

    df = df.merge(station_day_treat, on=["station_id", "date"], how="left")
    df["strike_exposed"] = df["strike_exposed"].fillna(0).astype(int)
    return df.drop(columns=["date"])
