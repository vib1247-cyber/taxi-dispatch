from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def _ensure_ts_hour(df: pd.DataFrame, time_col: str) -> pd.Series:
    ts = pd.to_datetime(df[time_col], errors="coerce")
    return ts.dt.floor("h")


def load_weather_csv(
    path: str | Path,
    *,
    time_col: str = "timestamp",
    feature_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load weather CSV, return DataFrame with 'timestamp_hour' and selected numeric features.

    If feature_cols is None, include all numeric columns except the time column.
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=[time_col])
    df["timestamp_hour"] = _ensure_ts_hour(df, time_col)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != time_col and np.issubdtype(df[c].dtype, np.number)]
    out = df[["timestamp_hour", *feature_cols]].groupby("timestamp_hour", as_index=False).mean()
    return out


def load_events_csv(
    path: str | Path,
    *,
    time_col: str = "timestamp",
    zone_col: Optional[str] = None,
    feature_cols: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load events CSV. If zone_col is provided, return per-zone events; otherwise global per-hour.

    Output includes columns: 'timestamp_hour', optional 'zone', and selected numeric features.
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=[time_col])
    df["timestamp_hour"] = _ensure_ts_hour(df, time_col)
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in {time_col, zone_col} and np.issubdtype(df[c].dtype, np.number)]
    group_keys = ["timestamp_hour"] + ([zone_col] if zone_col else [])
    out = df[group_keys + list(feature_cols)].groupby(group_keys, as_index=False).sum()
    if zone_col and zone_col != "zone":
        out = out.rename(columns={zone_col: "zone"})
    return out


def merge_external_features(
    agg_df: pd.DataFrame,
    *,
    weather_df: Optional[pd.DataFrame] = None,
    events_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge external features into aggregated demand table.

    - weather_df: merged on 'timestamp_hour' and broadcast to all zones
    - events_df: if contains 'zone', merge on ['timestamp_hour', 'zone']; else broadcast
    """
    out = agg_df.copy()
    if weather_df is not None and not weather_df.empty:
        out = out.merge(weather_df, on="timestamp_hour", how="left")
    if events_df is not None and not events_df.empty:
        if "zone" in events_df.columns:
            out = out.merge(events_df, on=["timestamp_hour", "zone"], how="left")
        else:
            out = out.merge(events_df, on="timestamp_hour", how="left")
    # Fill NaNs in added numeric columns with zeros
    added_cols = [c for c in out.columns if c not in agg_df.columns]
    for c in added_cols:
        if np.issubdtype(out[c].dtype, np.number):
            out[c] = out[c].fillna(0.0)
    return out
