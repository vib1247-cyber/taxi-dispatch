from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taxi_dispatch.data_ingest import generate_synthetic_trips
from taxi_dispatch.clustering import fit_kmeans, assign_zones
from taxi_dispatch.features import add_time_features, aggregate_demand, build_features, train_test_split_time
from taxi_dispatch.demand_model import train_model
from taxi_dispatch.rl_env import TaxiZoneEnv, train_q_learning


@pytest.mark.parametrize("days,trips_per_day", [(1, 300)])
def test_generate_and_cluster(days, trips_per_day):
    df = generate_synthetic_trips(days=days, trips_per_day=trips_per_day, seed=0)
    assert set(["timestamp", "lat", "lon"]).issubset(df.columns)
    km, centers = fit_kmeans(df, n_clusters=5, lat_col="lat", lon_col="lon", seed=0)
    df2 = assign_zones(df, km, lat_col="lat", lon_col="lon")
    assert "zone" in df2.columns
    assert len(centers) == 5


def test_features_and_model():
    df = generate_synthetic_trips(days=1, trips_per_day=300, seed=1)
    # Assign zones before aggregation
    km, centers = fit_kmeans(df, n_clusters=5, lat_col="lat", lon_col="lon", seed=1)
    df = assign_zones(df, km, lat_col="lat", lon_col="lon")
    df = add_time_features(df)
    agg = aggregate_demand(df, time_col="timestamp", zone_col="zone")
    X_all, y_all = build_features(agg)
    X_train, X_test, y_train, y_test = train_test_split_time(X_all, y_all, test_ratio=0.3)
    X_train_model = X_train.drop(columns=[c for c in ["timestamp_hour", "zone"] if c in X_train.columns], errors="ignore")
    # Train RF quickly
    model = train_model(X_train_model, y_train, model_type="rf", tune=False, n_jobs=1, seed=1)
    assert hasattr(model, "predict")


def test_rl_env_and_q_learning():
    # Minimal env
    centers = np.array([[40.0, -73.0], [40.1, -73.1], [40.2, -73.2]])
    T, Z = 12, 3
    rng = np.random.default_rng(0)
    demand = rng.random((T, Z)) * 10.0
    env = TaxiZoneEnv(centroids=centers, demand_matrix=demand, move_cost=0.1)
    Q = train_q_learning(env, episodes=10, alpha=0.2, gamma=0.95, eps_start=0.5, eps_min=0.05, eps_decay=0.99, seed=0)
    assert Q.shape == (T, Z, Z)
    # One step simulate
    t1, z1, r1 = env.step(0, 0, 1)
    assert isinstance(r1, float)
