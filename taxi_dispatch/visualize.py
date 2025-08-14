from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cluster_map(
    df: pd.DataFrame,
    centers: np.ndarray,
    *,
    lat_col: str,
    lon_col: str,
    zone_col: str,
    out_path: str | Path,
    zone_avg_demand: dict[int, float] | None = None,
):
    plt.figure(figsize=(7, 7))
    # Scatter a sample if very large
    sample = df
    if len(df) > 20000:
        sample = df.sample(20000, random_state=0)
    sc = plt.scatter(sample[lon_col], sample[lat_col], c=sample[zone_col], s=4, alpha=0.4, cmap="tab20")
    if zone_avg_demand:
        # Scale centroid size by avg demand (normalized)
        vals = np.array([zone_avg_demand.get(i, 0.0) for i in range(len(centers))], dtype=float)
        if vals.max() > 0:
            sizes = 50 + 150 * (vals / (vals.max() + 1e-8))
        else:
            sizes = np.full(len(centers), 80.0)
        plt.scatter(centers[:, 1], centers[:, 0], c="black", s=sizes, marker="x", label="Centroids")
    else:
        plt.scatter(centers[:, 1], centers[:, 0], c="black", s=80, marker="x", label="Centroids")
    # Label zone IDs at centroids
    for i, (la, lo) in enumerate(centers):
        plt.text(lo, la, str(i), fontsize=9, ha="left", va="bottom", color="black")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("City Clusters (KMeans)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_demand_heatmap(agg_df: pd.DataFrame, *, out_path: str | Path):
    # Average demand by hour and zone
    piv = agg_df.groupby(["zone", "hour"])['demand'].mean().unstack(fill_value=0.0)
    plt.figure(figsize=(10, max(4, piv.shape[0] * 0.3)))
    sns.heatmap(piv, cmap="YlOrRd", linewidths=0.3)
    plt.title("Average Demand Heatmap (Hour x Zone)")
    plt.xlabel("Hour of Day")
    plt.ylabel("Zone")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_forecast_vs_actual(
    actual_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    zone: int,
    out_path: str | Path,
):
    a = actual_df[actual_df["zone"] == zone].sort_values("timestamp_hour")
    p = pred_df[pred_df["zone"] == zone].sort_values("timestamp_hour")
    # Align by timestamp
    merged = (
        a[["timestamp_hour", "actual"]]
        .merge(p[["timestamp_hour", "pred"]], on="timestamp_hour", how="inner")
        .tail(200)  # plot recent window for clarity
    )
    plt.figure(figsize=(10, 4))
    plt.plot(merged["timestamp_hour"], merged["actual"], label="Actual", lw=1.5)
    plt.plot(merged["timestamp_hour"], merged["pred"], label="Predicted", lw=1.5)
    plt.title(f"Demand Forecast vs Actual (Zone {zone})")
    plt.xlabel("Time")
    plt.ylabel("Demand")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
