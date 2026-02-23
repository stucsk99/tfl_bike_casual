"""
tfl_utils.py

Utilities for the TfL cycle hire × tube strike causal ML project.

Core idea:
- Build a station-time outcome panel from raw journey extracts.
- Build a station↔line exposure mapping (bike station near tube stations served by lines).
- Expand line-level strike events into daily indicators.
- Join everything into a modelling-ready basefile.

These utilities are intentionally "boring": pure functions + explicit inputs/outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd
import requests

AggSide = Literal["start", "end", "both"]
Freq = Literal["h", "d"]


# -----------------------------
# Journey data aggregation
# -----------------------------

def _norm_colname(s: str) -> str:
    """Lowercase and drop non-alphanumerics to make schema matching robust."""
    return "".join(ch for ch in s.lower() if ch.isalnum())



def aggregate_from_folder_chunked(
    folder_path: str | Path,
    *,
    freq: Freq = "h",
    side: AggSide = "start",
    chunksize: int = 200_000,
    dayfirst: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scan a folder recursively for journey CSVs and aggregate into a station-time panel.

    Output schema:
      - station_id (int)
      - ts (datetime floored to hour/day)
      - trips_start (int) if side includes start
      - trips_end (int) if side includes end

    Notes:
      - Reads each file in chunks to avoid out-of-memory crashes.
      - Handles schema changes by detecting the correct date/station id columns per file.
      - Uses concat + groupby sum instead of repeated outer merges (more memory-stable).
    """
    folder = Path(folder_path)
    files = sorted(folder.rglob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found under: {folder.resolve()}")

    if verbose:
        print(f"Found {len(files)} files under {folder.resolve()}")

    acc = None
    skipped: list[tuple[str, str]] = []

    for f in files:
        if verbose:
            print(f"Processing {f.name} ...")

        # Header-only read to detect schema
        try:
            header = pd.read_csv(f, nrows=0)
        except Exception as e:
            skipped.append((f.name, f"header_read_failed: {e}"))
            continue

        cols = list(header.columns)
        start_id_col = _pick_col(cols, START_ID_NORMS)
        start_date_col = _pick_col(cols, START_DATE_NORMS)
        end_id_col = _pick_col(cols, END_ID_NORMS)
        end_date_col = _pick_col(cols, END_DATE_NORMS)

        needed: list[str] = []
        if side in ("start", "both"):
            if start_id_col is None or start_date_col is None:
                skipped.append((f.name, "missing start columns"))
                continue
            needed += [start_id_col, start_date_col]
        if side in ("end", "both"):
            if end_id_col is None or end_date_col is None:
                skipped.append((f.name, "missing end columns"))
                continue
            needed += [end_id_col, end_date_col]
        needed = list(dict.fromkeys(needed))  # dedupe

        try:
            for chunk in pd.read_csv(
                f,
                usecols=needed,
                chunksize=chunksize,
                low_memory=True,
            ):
                parts: list[pd.DataFrame] = []

                if side in ("start", "both"):
                    tmp = chunk[[start_id_col, start_date_col]].copy().dropna()
                    tmp["station_id"] = pd.to_numeric(tmp[start_id_col], errors="coerce")
                    tmp["ts"] = pd.to_datetime(tmp[start_date_col], dayfirst=dayfirst, errors="coerce")
                    tmp = tmp.dropna(subset=["station_id", "ts"])
                    tmp["station_id"] = tmp["station_id"].astype(int)
                    tmp["ts"] = tmp["ts"].dt.floor(freq)
                    g = tmp.groupby(["station_id", "ts"], as_index=False).size()
                    g = g.rename(columns={"size": "trips_start"})
                    parts.append(g)

                if side in ("end", "both"):
                    tmp = chunk[[end_id_col, end_date_col]].copy().dropna()
                    tmp["station_id"] = pd.to_numeric(tmp[end_id_col], errors="coerce")
                    tmp["ts"] = pd.to_datetime(tmp[end_date_col], dayfirst=dayfirst, errors="coerce")
                    tmp = tmp.dropna(subset=["station_id", "ts"])
                    tmp["station_id"] = tmp["station_id"].astype(int)
                    tmp["ts"] = tmp["ts"].dt.floor(freq)
                    g = tmp.groupby(["station_id", "ts"], as_index=False).size()
                    g = g.rename(columns={"size": "trips_end"})
                    parts.append(g)

                if not parts:
                    continue

                chunk_agg = parts[0]
                for p in parts[1:]:
                    chunk_agg = chunk_agg.merge(p, on=["station_id", "ts"], how="outer")

                if acc is None:
                    acc = chunk_agg
                else:
                    acc = pd.concat([acc, chunk_agg], ignore_index=True)
                    sum_cols = [c for c in ["trips_start", "trips_end"] if c in acc.columns]
                    acc = acc.groupby(["station_id", "ts"], as_index=False)[sum_cols].sum()

        except Exception as e:
            skipped.append((f.name, f"read_or_parse_failed: {e}"))
            continue

    if acc is None:
        raise RuntimeError("No files were successfully processed.")

    # Ensure columns exist even if side excluded them
    if "trips_start" not in acc.columns:
        acc["trips_start"] = 0
    if "trips_end" not in acc.columns:
        acc["trips_end"] = 0

    acc = acc.sort_values(["station_id", "ts"]).reset_index(drop=True)

    if verbose and skipped:
        print("\nSkipped files summary (first 20):")
        for name, reason in skipped[:20]:
            print(f"  - {name}: {reason}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    return acc




def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())

def _pick_col(cols, candidates_norm):
    norm_map = {_norm(c): c for c in cols}
    for cn in candidates_norm:
        if cn in norm_map:
            return norm_map[cn]
    return None

START_ID = ["startstationid", "startstationlogicalterminal", "startstationnumber"]
START_DT = ["startdate", "startdatetime", "starttime"]
END_ID   = ["endstationid", "endstationlogicalterminal", "endstationnumber"]
END_DT   = ["enddate", "enddatetime", "endtime"]

def aggregate_one_csv_to_parquet(
    csv_path: str | Path,
    out_dir: str | Path,
    *,
    freq: str = "h",
    side: str = "start",
    chunksize: int = 120_000,
    dayfirst: bool = True,
) -> Path | None:
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    header = pd.read_csv(csv_path, nrows=0)
    cols = list(header.columns)

    start_id = _pick_col(cols, START_ID)
    start_dt = _pick_col(cols, START_DT)
    end_id   = _pick_col(cols, END_ID)
    end_dt   = _pick_col(cols, END_DT)

    usecols = []
    if side in ("start", "both"):
        if start_id is None or start_dt is None:
            return None
        usecols += [start_id, start_dt]
    if side in ("end", "both"):
        if end_id is None or end_dt is None:
            return None
        usecols += [end_id, end_dt]
    usecols = list(dict.fromkeys(usecols))

    # Aggregate in-memory for THIS ONE FILE only (safe)
    agg_list = []

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=True):
        parts = []

        if side in ("start", "both"):
            tmp = chunk[[start_id, start_dt]].dropna()
            tmp["station_id"] = pd.to_numeric(tmp[start_id], errors="coerce")
            tmp["ts"] = pd.to_datetime(tmp[start_dt], dayfirst=dayfirst, errors="coerce")
            tmp = tmp.dropna(subset=["station_id", "ts"])
            tmp["station_id"] = tmp["station_id"].astype("int32")
            tmp["ts"] = tmp["ts"].dt.floor(freq)
            g = tmp.groupby(["station_id", "ts"], as_index=False).size().rename(columns={"size": "trips_start"})
            parts.append(g)

        if side in ("end", "both"):
            tmp = chunk[[end_id, end_dt]].dropna()
            tmp["station_id"] = pd.to_numeric(tmp[end_id], errors="coerce")
            tmp["ts"] = pd.to_datetime(tmp[end_dt], dayfirst=dayfirst, errors="coerce")
            tmp = tmp.dropna(subset=["station_id", "ts"])
            tmp["station_id"] = tmp["station_id"].astype("int32")
            tmp["ts"] = tmp["ts"].dt.floor(freq)
            g = tmp.groupby(["station_id", "ts"], as_index=False).size().rename(columns={"size": "trips_end"})
            parts.append(g)

        if not parts:
            continue

        chunk_agg = parts[0]
        for p in parts[1:]:
            chunk_agg = chunk_agg.merge(p, on=["station_id", "ts"], how="outer")

        agg_list.append(chunk_agg)

    if not agg_list:
        return None

    file_agg = pd.concat(agg_list, ignore_index=True)
    sum_cols = [c for c in ["trips_start", "trips_end"] if c in file_agg.columns]
    file_agg = file_agg.groupby(["station_id", "ts"], as_index=False)[sum_cols].sum()

    for c in sum_cols:
        file_agg[c] = file_agg[c].fillna(0).astype("int32")

    out_path = out_dir / f"{csv_path.stem}.parquet"
    file_agg.to_parquet(out_path, index=False)
    return out_path


def _looks_like_text_csv(path: Path, nbytes: int = 2048) -> bool:
    """
    Returns True if the file begins with printable text and contains commas/newlines,
    which strongly suggests it's CSV/TSV-like, even if the extension is .xls.
    """
    with open(path, "rb") as f:
        head = f.read(nbytes)

    # If it contains a lot of null bytes, it's probably binary (real .xls)
    if b"\x00" in head:
        return False

    # Try decoding as UTF-8 or latin-1; if that works and we see commas, likely CSV
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = head.decode("latin-1")
        except UnicodeDecodeError:
            return False

    return ("," in text) and ("\n" in text or "\r" in text)



def aggregate_one_file_to_parquet(
    path: str | Path,
    out_dir: str | Path,
    *,
    freq: str = "h",
    side: str = "start",
    chunksize: int = 120_000,
    dayfirst: bool = True,
    sheet_name: int | str = 0,
) -> Path | None:
    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = path.suffix.lower()

    # Decide how to read
    treat_as_csv = (ext == ".csv") or (ext == ".xls" and _looks_like_text_csv(path))

    # --- header + cols ---
    if treat_as_csv:
        header = pd.read_csv(path, nrows=0)
        cols = list(header.columns)
    elif ext == ".xls":
        header = pd.read_excel(path, sheet_name=sheet_name, nrows=0, engine="xlrd")
        cols = list(header.columns)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    start_id = _pick_col(cols, START_ID)
    start_dt = _pick_col(cols, START_DT)
    end_id   = _pick_col(cols, END_ID)
    end_dt   = _pick_col(cols, END_DT)

    usecols = []
    if side in ("start", "both"):
        if start_id is None or start_dt is None:
            return None
        usecols += [start_id, start_dt]
    if side in ("end", "both"):
        if end_id is None or end_dt is None:
            return None
        usecols += [end_id, end_dt]
    usecols = list(dict.fromkeys(usecols))

    def agg_df(df: pd.DataFrame) -> pd.DataFrame:
        parts = []

        if side in ("start", "both"):
            tmp = df[[start_id, start_dt]].dropna()
            tmp["station_id"] = pd.to_numeric(tmp[start_id], errors="coerce")
            tmp["ts"] = pd.to_datetime(tmp[start_dt], dayfirst=dayfirst, errors="coerce")
            tmp = tmp.dropna(subset=["station_id", "ts"])
            tmp["station_id"] = tmp["station_id"].astype("int32")
            tmp["ts"] = tmp["ts"].dt.floor(freq)
            g = tmp.groupby(["station_id", "ts"], as_index=False).size().rename(columns={"size": "trips_start"})
            parts.append(g)

        if side in ("end", "both"):
            tmp = df[[end_id, end_dt]].dropna()
            tmp["station_id"] = pd.to_numeric(tmp[end_id], errors="coerce")
            tmp["ts"] = pd.to_datetime(tmp[end_dt], dayfirst=dayfirst, errors="coerce")
            tmp = tmp.dropna(subset=["station_id", "ts"])
            tmp["station_id"] = tmp["station_id"].astype("int32")
            tmp["ts"] = tmp["ts"].dt.floor(freq)
            g = tmp.groupby(["station_id", "ts"], as_index=False).size().rename(columns={"size": "trips_end"})
            parts.append(g)

        if not parts:
            return pd.DataFrame(columns=["station_id", "ts"])

        out = parts[0]
        for p in parts[1:]:
            out = out.merge(p, on=["station_id", "ts"], how="outer")
        return out

    agg_list = []

    if treat_as_csv:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=True):
            chunk_agg = agg_df(chunk)
            if len(chunk_agg):
                agg_list.append(chunk_agg)
    else:
        df = pd.read_excel(path, sheet_name=sheet_name, usecols=usecols, engine="xlrd")
        file_agg = agg_df(df)
        if len(file_agg):
            agg_list.append(file_agg)

    if not agg_list:
        return None

    file_agg = pd.concat(agg_list, ignore_index=True)
    sum_cols = [c for c in ["trips_start", "trips_end"] if c in file_agg.columns]
    file_agg = file_agg.groupby(["station_id", "ts"], as_index=False)[sum_cols].sum()

    for c in sum_cols:
        file_agg[c] = file_agg[c].fillna(0).astype("int32")

    out_path = out_dir / f"{path.stem}.parquet"
    file_agg.to_parquet(out_path, index=False)
    return out_path


# -----------------------------
# Bike station locations
# -----------------------------

def download_bikepoints_json(out_path: str | Path, *, timeout: int = 60) -> Path:
    """
    Download BikePoint data (all docking stations) from the TfL API to a JSON file.
    Endpoint: https://api.tfl.gov.uk/BikePoint
    """
    out_path = Path(out_path)
    r = requests.get("https://api.tfl.gov.uk/BikePoint", timeout=timeout)
    r.raise_for_status()
    out_path.write_text(pd.io.json.dumps(r.json(), indent=2), encoding="utf-8")
    return out_path


def load_bike_station_locations(bikepoints_json_path: str | Path) -> pd.DataFrame:
    """
    Load bike station locations from a BikePoint JSON dump.

    Returns DataFrame with:
      - station_id (int; extracted from 'BikePoints_###')
      - lat
      - lon
      - station_name
    """
    bikepoints_json_path = Path(bikepoints_json_path)
    data = pd.read_json(bikepoints_json_path)

    # If read_json produced a DataFrame of dicts, handle accordingly
    if isinstance(data, pd.DataFrame) and "id" in data.columns:
        df = data
    else:
        # Fallback: raw JSON list
        import json
        with open(bikepoints_json_path, encoding="utf-8") as f:
            raw = json.load(f)
        df = pd.DataFrame(raw)

    out = pd.DataFrame({
        "station_id": df["id"].astype(str).str.split("_").str[-1].astype(int),
        "lat": pd.to_numeric(df["lat"], errors="coerce"),
        "lon": pd.to_numeric(df["lon"], errors="coerce"),
        "station_name": df.get("commonName", pd.Series([None] * len(df))),
    }).dropna(subset=["lat", "lon"])

    return out


# -----------------------------
# Tube stations + lines
# -----------------------------

UNDERGROUND_LINES = {
    "bakerloo",
    "central",
    "circle",
    "district",
    "hammersmith-city",
    "jubilee",
    "metropolitan",
    "northern",
    "piccadilly",
    "victoria",
    "waterloo-city",
}

def fetch_tube_stations_and_lines(*, timeout: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch tube stop points and served Underground lines from the TfL API.

    Returns:
      tube_stations: tube_station_id, tube_station_name, lat, lon
      tube_lines: tube_station_id, affected_line (e.g. 'central_line')
    """
    url = "https://api.tfl.gov.uk/StopPoint/Mode/tube"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    stop_points = data.get("stopPoints", data)

    st_rows = []
    line_rows = []

    for sp in stop_points:
        tube_station_id = sp.get("naptanId") or sp.get("id")
        name = sp.get("commonName")
        lat = sp.get("lat")
        lon = sp.get("lon")

        if tube_station_id is None or lat is None or lon is None:
            continue

        st_rows.append({
            "tube_station_id": tube_station_id,
            "tube_station_name": name,
            "lat": lat,
            "lon": lon,
        })

        for ln in sp.get("lines", []) or []:
            line_id = ln.get("id")
            if line_id in UNDERGROUND_LINES:
                line_rows.append({
                    "tube_station_id": tube_station_id,
                    "affected_line": f"{line_id}_line",
                })

    tube_stations = pd.DataFrame(st_rows).drop_duplicates(subset=["tube_station_id"])
    tube_lines = pd.DataFrame(line_rows).drop_duplicates()

    return tube_stations, tube_lines


# -----------------------------
# Spatial mapping: bike station → tube station(s) → line(s)
# -----------------------------

def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Great-circle distance in metres.
    lat2/lon2 may be arrays.
    """
    R = 6371000.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def build_station_line_map(
    bike_stations: pd.DataFrame,
    tube_stations: pd.DataFrame,
    tube_lines: pd.DataFrame,
    *,
    radius_m: float = 800.0,
    fallback_to_nearest: bool = True,
    dedupe_line_per_station: bool = True,
) -> pd.DataFrame:
    """
    Build (bike station) → (affected_line) exposure map.

    Logic:
      - For each bike station, find all tube stations within radius_m.
      - If none and fallback_to_nearest=True, map to the single nearest tube station.
      - Expand tube stations to the Underground lines they serve (tube_lines).
      - Optionally dedupe to one row per (station_id, affected_line).

    Returns columns:
      station_id, affected_line, tube_station_id, dist_m
    """
    b = bike_stations[["station_id", "lat", "lon"]].dropna().copy()
    t = tube_stations[["tube_station_id", "lat", "lon"]].dropna().copy()

    b_lat = b["lat"].to_numpy()
    b_lon = b["lon"].to_numpy()
    t_lat = t["lat"].to_numpy()
    t_lon = t["lon"].to_numpy()
    t_id = t["tube_station_id"].to_numpy()

    rows = []
    for i in range(len(b)):
        d = haversine_m(b_lat[i], b_lon[i], t_lat, t_lon)
        idx = np.where(d <= radius_m)[0]
        if len(idx) == 0 and fallback_to_nearest:
            idx = np.array([int(np.argmin(d))])

        for j in idx:
            rows.append({
                "station_id": int(b.iloc[i]["station_id"]),
                "tube_station_id": t_id[j],
                "dist_m": float(d[j]),
            })

    station_tube = pd.DataFrame(rows).drop_duplicates()

    station_line_map = (
        station_tube.merge(tube_lines, on="tube_station_id", how="left")
        .dropna(subset=["affected_line"])
        [["station_id", "affected_line", "tube_station_id", "dist_m"]]
        .drop_duplicates()
    )

    if dedupe_line_per_station:
        # Keep the nearest tube station record per (station_id, affected_line)
        station_line_map = (
            station_line_map.sort_values(["station_id", "affected_line", "dist_m"])
            .drop_duplicates(subset=["station_id", "affected_line"], keep="first")
            .reset_index(drop=True)
        )

    return station_line_map


# -----------------------------
# Strike table utilities
# -----------------------------

def expand_strikes_daily(strike_data: pd.DataFrame) -> pd.DataFrame:
    """
    Expand strike events (date_start/date_end) into daily rows.

    Input columns required:
      - date_start (dd/mm/yy or yyyy-mm-dd)
      - date_end
      - affected_line (e.g. 'central_line')

    Output:
      - date (datetime64[ns], floored to day)
      - affected_line
      - strike (=1)
    """
    s = strike_data.copy()
    s["date_start"] = pd.to_datetime(s["date_start"], dayfirst=True, errors="coerce")
    s["date_end"] = pd.to_datetime(s["date_end"], dayfirst=True, errors="coerce")
    s = s.dropna(subset=["date_start", "date_end", "affected_line"]).copy()

    rows = []
    for r in s.itertuples(index=False):
        for d in pd.date_range(r.date_start.floor("D"), r.date_end.floor("D"), freq="D"):
            rows.append({"date": d, "affected_line": r.affected_line, "strike": 1})

    return pd.DataFrame(rows).drop_duplicates()


def attach_strikes_to_base(
    base: pd.DataFrame,
    strikes_daily: pd.DataFrame,
    station_line_map: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach a binary station-day strike exposure to a station-time panel.

    base must have:
      - station_id
      - ts (datetime)

    Output adds:
      - strike_exposed (0/1)
    """
    df = base.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df["date"] = df["ts"].dt.floor("D")

    station_day_treat = (
        strikes_daily.merge(
            station_line_map[["station_id", "affected_line"]],
            on="affected_line",
            how="inner",
        )
        .drop_duplicates(subset=["station_id", "date"])
        .assign(strike_exposed=1)[["station_id", "date", "strike_exposed"]]
    )

    df = df.merge(station_day_treat, on=["station_id", "date"], how="left")
    df["strike_exposed"] = df["strike_exposed"].fillna(0).astype(int)

    return df.drop(columns=["date"])
