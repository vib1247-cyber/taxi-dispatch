from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def ensure_dirs(paths: Iterable[os.PathLike | str]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def load_trips_csv(
    path: str | os.PathLike,
    *,
    time_col: str = "timestamp",
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """Load a CSV of trip records and standardize column names.

    Returns a DataFrame with columns: timestamp (datetime64[ns]), lat, lon
    """
    df = pd.read_csv(path)
    if time_col not in df or lat_col not in df or lon_col not in df:
        raise ValueError(
            f"CSV missing required columns. Found: {list(df.columns)}; expected: {time_col}, {lat_col}, {lon_col}"
        )
    df = df[[time_col, lat_col, lon_col]].rename(
        columns={time_col: "timestamp", lat_col: "lat", lon_col: "lon"}
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "lat", "lon"]).reset_index(drop=True)
    df = df.astype({"lat": float, "lon": float})
    return df


def generate_synthetic_trips(
    *, days: int = 7, trips_per_day: int = 5000, seed: int = 42
) -> pd.DataFrame:
    """Generate a realistic synthetic trip dataset around a city.

    - Uses hourly demand profile with morning/evening peaks
    - Samples a hotspot per trip with hour-varying probabilities
    - Adds small Gaussian noise around hotspot coordinates
    """
    rng = np.random.default_rng(seed)

    # Manhattan-like hotspots [lat, lon]
    hotspots = np.array(
        [
            [40.7580, -73.9855],  # Midtown
            [40.7060, -74.0086],  # FiDi
            [40.7306, -73.9866],  # East Village
            [40.7794, -73.9632],  # Upper East/West
            [40.6782, -73.9442],  # Brooklyn
        ]
    )
    n_hot = len(hotspots)

    # Hourly demand profile (0..23)
    hour_weights = np.array(
        [
            0.2, 0.15, 0.12, 0.1, 0.1, 0.12,  # 0-5
            0.25, 0.45, 0.65, 0.55, 0.40, 0.35,  # 6-11 (morning)
            0.35, 0.35, 0.40, 0.45, 0.60, 0.75,  # 12-17 (evening starts)
            0.80, 0.70, 0.55, 0.40, 0.35, 0.30,  # 18-23
        ]
    )
    hour_weights = hour_weights / hour_weights.sum()

    # Hotspot preferences by hour (24 x n_hot)
    hotspot_pref = np.ones((24, n_hot))
    # Morning: FiDi, Midtown
    for h in [7, 8, 9, 10]:
        hotspot_pref[h] = np.array([0.35, 0.40, 0.10, 0.10, 0.05])
    # Evening: Midtown, UES/UWS
    for h in [16, 17, 18, 19, 20]:
        hotspot_pref[h] = np.array([0.45, 0.10, 0.15, 0.25, 0.05])
    # Late night: East Village, Midtown
    for h in [21, 22, 23, 0, 1]:
        hotspot_pref[h % 24] = np.array([0.30, 0.10, 0.45, 0.10, 0.05])
    # Normalize rows
    hotspot_pref = hotspot_pref / hotspot_pref.sum(axis=1, keepdims=True)

    rows = []
    start = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)

    for d in range(days):
        date0 = start + pd.Timedelta(days=d)
        # Allocate per-hour counts from Poisson around target trips_per_day
        hourly_means = trips_per_day * hour_weights
        hourly_counts = rng.poisson(lam=hourly_means)
        for hour, n in enumerate(hourly_counts):
            if n <= 0:
                continue
            # Choose hotspot indices for this hour
            hot_idx = rng.choice(n_hot, size=n, p=hotspot_pref[hour])
            # Random minutes/seconds
            mins = rng.integers(0, 60, size=n)
            secs = rng.integers(0, 60, size=n)
            # Gaussian noise around hotspot
            noise = rng.normal(loc=0.0, scale=0.0075, size=(n, 2))  # ~0.8km std
            coords = hotspots[hot_idx] + noise
            for i in range(n):
                ts = date0 + pd.Timedelta(hours=hour, minutes=int(mins[i]), seconds=int(secs[i]))
                rows.append((ts, float(coords[i, 0]), float(coords[i, 1])))

    df = pd.DataFrame(rows, columns=["timestamp", "lat", "lon"]).sort_values("timestamp").reset_index(drop=True)
    return df
