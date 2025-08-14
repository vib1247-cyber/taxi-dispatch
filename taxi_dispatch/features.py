from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, *, time_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    out = out.dropna(subset=[time_col]).reset_index(drop=True)
    out["hour"] = out[time_col].dt.hour.astype(int)
    out["dow"] = out[time_col].dt.dayofweek.astype(int)
    return out


def _complete_time_zone_grid(agg: pd.DataFrame) -> pd.DataFrame:
    zones = np.sort(agg["zone"].unique())
    t_min = agg["timestamp_hour"].min()
    t_max = agg["timestamp_hour"].max()
    full_times = pd.date_range(t_min, t_max, freq="h")
    idx = pd.MultiIndex.from_product([full_times, zones], names=["timestamp_hour", "zone"])
    agg = agg.set_index(["timestamp_hour", "zone"]).reindex(idx).reset_index()
    agg["demand"] = agg["demand"].fillna(0.0)
    return agg


def aggregate_demand(
    df: pd.DataFrame,
    *,
    time_col: str = "timestamp",
    zone_col: str = "zone",
) -> pd.DataFrame:
    g = df.copy()
    g["timestamp_hour"] = g[time_col].dt.floor("h")
    agg = g.groupby(["timestamp_hour", zone_col]).size().rename("demand").reset_index()
    agg = agg.rename(columns={zone_col: "zone"})
    agg = _complete_time_zone_grid(agg)
    # Recreate hour/dow after completion
    agg["hour"] = agg["timestamp_hour"].dt.hour.astype(int)
    agg["dow"] = agg["timestamp_hour"].dt.dayofweek.astype(int)
    return agg


def build_features(agg: pd.DataFrame):
    """Create supervised features and target for demand modeling.

    Returns X_all (with 'timestamp_hour' and integer 'zone' column preserved) and y_all.
    The model will be trained on one-hot features and we keep time/zone for later pivoting.
    """
    # Identify additional numeric features beyond core keys
    core_cols = {"timestamp_hour", "zone", "hour", "dow", "demand"}
    extra_num_cols = [c for c in agg.columns if c not in core_cols and np.issubdtype(agg[c].dtype, np.number)]

    base = agg[["timestamp_hour", "zone", "hour", "dow", "demand"] + extra_num_cols].copy()
    X_cats = pd.get_dummies(base[["zone", "hour", "dow"]].astype(int).astype(str), prefix=["zone", "hour", "dow"])  # strings for safe get_dummies
    X = pd.concat([base[["timestamp_hour", "zone"]], X_cats, base[extra_num_cols]], axis=1)
    y = base["demand"].astype(float)
    return X, y


def train_test_split_time(X: pd.DataFrame, y: pd.Series, *, test_ratio: float = 0.2):
    times = np.sort(X["timestamp_hour"].unique())
    split_idx = int((1.0 - test_ratio) * len(times))
    train_times = set(times[:split_idx])
    is_train = X["timestamp_hour"].isin(train_times)

    X_train = X[is_train].drop(columns=["timestamp_hour"]).reset_index(drop=True)
    y_train = y[is_train].reset_index(drop=True)
    X_test = X[~is_train].drop(columns=["timestamp_hour"]).reset_index(drop=True)
    y_test = y[~is_train].reset_index(drop=True)
    return X_train, X_test, y_train, y_test
