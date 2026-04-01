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
    
    NOTE: The internal aggregation creates columns (ts=datetime, trips_start=count)
    but downstream code expects the opposite (ts=count, trips_start=datetime).
    We rename here to match downstream expectations.
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

    result = acc.sort_values(["station_id", "ts"]).reset_index(drop=True)
    
    # Rename to match downstream expectations: ts→trips_start (datetime), trips_start→ts (count)
    result = result.rename(columns={
        "ts": "trips_start",           # ts (datetime) → trips_start
        "trips_start": "ts",           # trips_start (count) → ts
    })
    if "trips_end" in result.columns:
        # If trips_end exists, we don't rename it (it stays as a count column)
        pass
    
    return result


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


# ── 3. Tube stations + lines ──────────────────────────────────────────────────

def fetch_tube_stations_and_lines(*, timeout: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch all Underground stop points from the TfL StopPoint API.

    Returns:
        tube_stations : DataFrame with tube_station_id, tube_station_name, lat, lon
        tube_lines    : DataFrame with tube_station_id, affected_line
                        (e.g. 'central_line') — one row per station-line pair.

    Only Underground lines are retained (see UNDERGROUND_LINES).
    """
    r = requests.get("https://api.tfl.gov.uk/StopPoint/Mode/tube", timeout=timeout)
    r.raise_for_status()
    stop_points = r.json().get("stopPoints", r.json())

    station_rows, line_rows = [], []
    for sp in stop_points:
        sid  = sp.get("naptanId") or sp.get("id")
        lat  = sp.get("lat")
        lon  = sp.get("lon")
        if not (sid and lat and lon):
            continue

        station_rows.append({
            "tube_station_id":   sid,
            "tube_station_name": sp.get("commonName"),
            "lat": lat, "lon": lon,
        })
        for ln in sp.get("lines", []) or []:
            line_id = ln.get("id")
            if line_id in UNDERGROUND_LINES:
                line_rows.append({
                    "tube_station_id": sid,
                    "affected_line":   f"{line_id}_line",
                })

    tube_stations = pd.DataFrame(station_rows).drop_duplicates(subset=["tube_station_id"])
    tube_lines    = pd.DataFrame(line_rows).drop_duplicates()
    return tube_stations, tube_lines


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
    bike_stations:           pd.DataFrame,
    tube_stations:           pd.DataFrame,
    tube_lines:              pd.DataFrame,
    *,
    radius_m:                float = 800.0,
    fallback_to_nearest:     bool  = True,
) -> pd.DataFrame:
    """
    Build a bike station → Underground line exposure table.

    For each bike station, we find all tube stations within radius_m and
    look up which Underground lines they serve.  If no tube station is within
    the radius and fallback_to_nearest=True, we map to the single nearest tube
    station regardless of distance (ensures every bike station has at least one
    line assignment).

    The radius of 800m is a defensible walking catchment: it is roughly
    10 minutes on foot and consistent with TfL's own accessibility modelling.

    Returns columns: station_id, affected_line, tube_station_id, dist_m
    """
    b_lats = bike_stations["lat"].to_numpy()
    b_lons = bike_stations["lon"].to_numpy()
    t_lats = tube_stations["lat"].to_numpy()
    t_lons = tube_stations["lon"].to_numpy()
    t_ids  = tube_stations["tube_station_id"].to_numpy()

    rows = []
    for i, (bike_id, blat, blon) in enumerate(
        zip(bike_stations["station_id"], b_lats, b_lons)
    ):
        dists = haversine_m(blat, blon, t_lats, t_lons)
        idxs  = np.where(dists <= radius_m)[0]
        if len(idxs) == 0 and fallback_to_nearest:
            idxs = np.array([int(np.argmin(dists))])
        for j in idxs:
            rows.append({
                "station_id":      int(bike_id),
                "tube_station_id": t_ids[j],
                "dist_m":          float(dists[j]),
            })

    station_tube = pd.DataFrame(rows).drop_duplicates()
    station_line_map = (
        station_tube
        .merge(tube_lines, on="tube_station_id", how="left")
        .dropna(subset=["affected_line"])
        [["station_id", "affected_line", "tube_station_id", "dist_m"]]
        # Keep the nearest tube station per (station_id, affected_line)
        .sort_values(["station_id", "affected_line", "dist_m"])
        .drop_duplicates(subset=["station_id", "affected_line"], keep="first")
        .reset_index(drop=True)
    )
    return station_line_map


# ── 5. Strike expansion ───────────────────────────────────────────────────────

def expand_strikes_daily(strike_data: pd.DataFrame) -> pd.DataFrame:
    """
    Expand strike events from a date-range table into one row per strike day.

    Expected input columns:
        date_start   (dd/mm/yy or yyyy-mm-dd)
        date_end     (same)
        affected_line  (e.g. 'central_line')

    Returns columns: date (datetime64), affected_line, strike (=1).
    """
    s = strike_data.copy()
    s["date_start"] = pd.to_datetime(s["date_start"], dayfirst=True, errors="coerce")
    s["date_end"]   = pd.to_datetime(s["date_end"],   dayfirst=True, errors="coerce")
    s = s.dropna(subset=["date_start", "date_end", "affected_line"])

    rows = []
    for row in s.itertuples(index=False):
        for d in pd.date_range(row.date_start.floor("D"), row.date_end.floor("D"), freq="D"):
            rows.append({"date": d, "affected_line": row.affected_line, "strike": 1})

    return pd.DataFrame(rows).drop_duplicates()


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

    base must have columns: station_id, trips_start (datetime).
    """
    df                = base.copy()
    df["trips_start"] = pd.to_datetime(df["trips_start"])
    df["date"]        = df["trips_start"].dt.floor("D")

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
