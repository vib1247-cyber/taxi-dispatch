from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def fit_kmeans(
    df: pd.DataFrame,
    *,
    n_clusters: int,
    lat_col: str = "lat",
    lon_col: str = "lon",
    seed: int = 42,
):
    X = df[[lat_col, lon_col]].to_numpy()
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(X)
    return km, km.cluster_centers_


def assign_zones(
    df: pd.DataFrame,
    km: KMeans,
    *,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    X = df[[lat_col, lon_col]].to_numpy()
    zones = km.predict(X)
    out = df.copy()
    out["zone"] = zones.astype(int)
    return out
