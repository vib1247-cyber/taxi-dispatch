from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import requests


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    import math

    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def haversine_matrix(centroids: np.ndarray) -> np.ndarray:
    Z = centroids.shape[0]
    D = np.zeros((Z, Z), dtype=float)
    for i in range(Z):
        for j in range(Z):
            if i == j:
                D[i, j] = 0.0
            else:
                D[i, j] = haversine_km(centroids[i, 0], centroids[i, 1], centroids[j, 0], centroids[j, 1])
    return D


def _osrm_table_request(osrm_url: str, centroids: np.ndarray) -> np.ndarray:
    # OSRM table API: /table/v1/driving/{lon,lat;lon,lat}?annotations=distance
    coords = ";".join([f"{lon:.6f},{lat:.6f}" for lat, lon in centroids])
    url = osrm_url.rstrip("/") + f"/table/v1/driving/{coords}?annotations=distance"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    dist_m = np.array(data.get("distances", []), dtype=float)
    if dist_m.size == 0:
        raise ValueError("OSRM response missing distances")
    return dist_m / 1000.0  # meters -> km


def compute_distance_matrix(
    centroids: np.ndarray,
    *,
    osrm_url: Optional[str] = None,
    cache_path: Optional[str | Path] = None,
) -> np.ndarray:
    """Compute pairwise distance matrix (km) using OSRM if url provided, otherwise haversine.

    If cache_path (.npy) is provided and exists, it will be loaded. On success, distances are saved to cache.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception:
                pass

    if osrm_url:
        try:
            D = _osrm_table_request(osrm_url, centroids)
        except Exception:
            # Fallback to haversine if OSRM fails
            D = haversine_matrix(centroids)
    else:
        D = haversine_matrix(centroids)

    if cache_path is not None:
        try:
            np.save(cache_path, D)
        except Exception:
            pass
    return D
