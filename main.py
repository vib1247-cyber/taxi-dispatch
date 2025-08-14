import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from taxi_dispatch.config import DEFAULTS
from taxi_dispatch.data_ingest import (
    generate_synthetic_trips,
    load_trips_csv,
    ensure_dirs,
)
from taxi_dispatch.clustering import fit_kmeans, assign_zones
from taxi_dispatch.features import (
    add_time_features,
    aggregate_demand,
    build_features,
    train_test_split_time,
)
from taxi_dispatch.demand_model import (
    evaluate_regression,
    train_model,
    DemandModelWrapper,
)
from taxi_dispatch.visualize import (
    plot_cluster_map,
    plot_demand_heatmap,
    plot_forecast_vs_actual,
)
from taxi_dispatch.rl_env import TaxiZoneEnv, train_q_learning
from taxi_dispatch.simulate import (
    simulate_policy,
    animate_route,
    simulate_multi_agents,
    animate_routes_multi,
)
from taxi_dispatch.external_features import (
    load_weather_csv,
    load_events_csv,
    merge_external_features,
)
from taxi_dispatch.routing import compute_distance_matrix
from taxi_dispatch.io_utils import save_joblib, save_npy


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"


def parse_args():
    p = argparse.ArgumentParser(description="Smart Taxi Dispatch System")
    p.add_argument("--data", type=str, default=None, help="Path to CSV trip data")
    p.add_argument("--time-col", type=str, default="timestamp", help="Timestamp column name")
    p.add_argument("--lat-col", type=str, default="lat", help="Latitude column name")
    p.add_argument("--lon-col", type=str, default="lon", help="Longitude column name")
    p.add_argument("--n-zones", type=int, default=DEFAULTS["N_ZONES"], help="KMeans clusters")
    p.add_argument("--generate-synth", action="store_true", help="Generate synthetic dataset")
    p.add_argument("--days", type=int, default=7, help="Synthetic: number of days")
    p.add_argument("--trips-per-day", type=int, default=5000, help="Synthetic: trips per day")
    p.add_argument("--episodes", type=int, default=DEFAULTS["RL"]["EPISODES"], help="Q-learning episodes")
    p.add_argument("--seed", type=int, default=DEFAULTS["RANDOM_SEED"], help="Random seed")
    # Model options
    p.add_argument("--model-type", type=str, default="rf", help="Demand model type: rf or xgb")
    p.add_argument("--tune", action="store_true", help="Enable hyperparameter tuning")
    p.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for model training")
    # External features
    p.add_argument("--weather-csv", type=str, default=None, help="Path to weather CSV")
    p.add_argument("--weather-time-col", type=str, default="timestamp", help="Weather time column")
    p.add_argument("--weather-features", type=str, default=None, help="Comma-separated weather feature cols")
    p.add_argument("--events-csv", type=str, default=None, help="Path to events CSV")
    p.add_argument("--events-time-col", type=str, default="timestamp", help="Events time column")
    p.add_argument("--events-zone-col", type=str, default=None, help="Events zone column (optional)")
    p.add_argument("--events-features", type=str, default=None, help="Comma-separated events feature cols")
    # Routing / RL
    p.add_argument("--osrm-url", type=str, default=None, help="OSRM base URL (e.g., http://localhost:5000)")
    p.add_argument("--distance-cache", type=str, default=None, help="Path to cache distance matrix .npy")
    p.add_argument("--move-cost", type=float, default=DEFAULTS["RL"]["MOVE_COST"], help="Cost per km to move")
    p.add_argument("--n-agents", type=int, default=1, help="Number of taxis to simulate (>=1)")
    return p.parse_args()


def main():
    args = parse_args()

    ensure_dirs([OUTPUTS_DIR, DATA_DIR])

    if args.generate_synth or not args.data:
        print("[i] Generating synthetic trips ...")
        df = generate_synthetic_trips(days=args.days, trips_per_day=args.trips_per_day, seed=args.seed)
        synth_path = DATA_DIR / "synthetic_trips.csv"
        df.to_csv(synth_path, index=False)
        print(f"[i] Saved synthetic trips -> {synth_path}")
    else:
        print(f"[i] Loading trips from {args.data} ...")
        df = load_trips_csv(args.data, time_col=args.time_col, lat_col=args.lat_col, lon_col=args.lon_col)

    # Clustering
    print("[i] Clustering into zones with KMeans ...")
    # After load_trips_csv, columns are standardized to 'lat' and 'lon'
    kmeans, centers = fit_kmeans(df, n_clusters=args.n_zones, lat_col="lat", lon_col="lon", seed=args.seed)
    df = assign_zones(df, kmeans, lat_col="lat", lon_col="lon")

    # Aggregate demand and build features
    # After load_trips_csv, time column is standardized to 'timestamp'
    df = add_time_features(df)
    agg_df = aggregate_demand(df)
    # Optional external features
    def _split_cols(s):
        return [c.strip() for c in s.split(",") if c.strip()] if s else None
    weather_df = (
        load_weather_csv(
            args.weather_csv,
            time_col=args.weather_time_col,
            feature_cols=_split_cols(args.weather_features),
        )
        if args.weather_csv
        else None
    )
    events_df = (
        load_events_csv(
            args.events_csv,
            time_col=args.events_time_col,
            zone_col=args.events_zone_col,
            feature_cols=_split_cols(args.events_features),
        )
        if args.events_csv
        else None
    )
    agg_df_ext = merge_external_features(agg_df, weather_df=weather_df, events_df=events_df)

    # Visualize clusters (with avg demand to scale centroid markers)
    zone_means_viz = agg_df.groupby("zone")["demand"].mean().to_dict()
    plot_cluster_map(
        df,
        centers,
        lat_col="lat",
        lon_col="lon",
        zone_col="zone",
        out_path=OUTPUTS_DIR / "cluster_map.png",
        zone_avg_demand=zone_means_viz,
    )

    # Demand heatmap
    plot_demand_heatmap(agg_df_ext, out_path=OUTPUTS_DIR / "demand_heatmap.png")

    # Build supervised dataset
    X_all, y_all = build_features(agg_df_ext)
    X_train, X_test, y_train, y_test = train_test_split_time(X_all, y_all, test_ratio=DEFAULTS["TEST_RATIO"])
    # Drop non-feature columns for model input
    X_train_model = X_train.drop(columns=[c for c in ["timestamp_hour", "zone"] if c in X_train.columns], errors="ignore")
    X_test_model = X_test.drop(columns=[c for c in ["timestamp_hour", "zone"] if c in X_test.columns], errors="ignore")

    # Train model
    print(f"[i] Training demand model ({args.model_type}{' + tuning' if args.tune else ''}) ...")
    model = train_model(
        X_train_model,
        y_train,
        model_type=args.model_type,
        seed=args.seed,
        tune=args.tune,
        n_jobs=args.n_jobs,
    )
    metrics = evaluate_regression(model, X_test_model, y_test)
    print(f"[i] Demand model metrics: MAE={metrics['mae']:.3f}, RMSE={metrics['rmse']:.3f}")

    # Forecast for visualization (entire horizon)
    features_all_model = X_all.drop(columns=[c for c in ["timestamp_hour", "zone"] if c in X_all.columns], errors="ignore")
    y_pred_all = model.predict(features_all_model)
    # Pivot predictions and actual into time x zone matrices
    pred_df = pd.DataFrame({
        "timestamp_hour": X_all["timestamp_hour"].values,
        "zone": X_all["zone"].values,
        "pred": y_pred_all,
    })
    actual_df = pd.DataFrame({
        "timestamp_hour": X_all["timestamp_hour"].values,
        "zone": X_all["zone"].values,
        "actual": y_all.values,
    })
    # Choose a representative zone (highest average demand)
    zone_means = actual_df.groupby("zone")["actual"].mean().sort_values(ascending=False)
    top_zone = int(zone_means.index[0])
    plot_forecast_vs_actual(
        actual_df, pred_df, zone=top_zone, out_path=OUTPUTS_DIR / f"demand_forecast_zone_{top_zone}.png"
    )

    # Save prediction/actual CSVs for dashboard use
    pred_path = OUTPUTS_DIR / "predictions.csv"
    actual_path = OUTPUTS_DIR / "actuals.csv"
    pred_df.to_csv(pred_path, index=False)
    actual_df.to_csv(actual_path, index=False)

    # Persist model wrapper for API/serving
    wrapper = DemandModelWrapper(
        model=model,
        feature_cols=list(features_all_model.columns),
        zone_levels=sorted(int(z) for z in actual_df["zone"].unique()),
        hour_levels=list(range(24)),
        dow_levels=list(range(7)),
        model_type=args.model_type,
    )
    save_joblib(wrapper, OUTPUTS_DIR / "demand_model.joblib")

    # Prepare demand matrix for RL (time x zone)
    times_sorted = np.sort(X_all["timestamp_hour"].unique())
    zones_sorted = np.sort(X_all["zone"].unique())
    T = len(times_sorted)
    Z = len(zones_sorted)
    demand_mat = (
        pred_df
        .pivot_table(index="timestamp_hour", columns="zone", values="pred", aggfunc="mean")
        .reindex(index=times_sorted, columns=zones_sorted, fill_value=0.0)
        .values
    )

    # Compute routing distances (km)
    dist_cache = args.distance_cache if args.distance_cache else str(OUTPUTS_DIR / "distance_matrix.npy")
    D = compute_distance_matrix(centers, osrm_url=args.osrm_url, cache_path=dist_cache)
    # Save centers for API/dashboard
    save_npy(centers, OUTPUTS_DIR / "centers.npy")

    print("[i] Training Q-learning agent ...")
    env = TaxiZoneEnv(centroids=centers, demand_matrix=demand_mat, move_cost=args.move_cost, distance_matrix=D)
    Q = train_q_learning(
        env,
        episodes=args.episodes,
        alpha=DEFAULTS["RL"]["ALPHA"],
        gamma=DEFAULTS["RL"]["GAMMA"],
        eps_start=DEFAULTS["RL"]["EPSILON_START"],
        eps_min=DEFAULTS["RL"]["EPSILON_MIN"],
        eps_decay=DEFAULTS["RL"]["EPSILON_DECAY"],
        seed=args.seed,
    )

    # Simulate route following learned policy
    steps = min(200, T)
    start_zone = top_zone
    if args.n_agents <= 1:
        path_zones, rewards = simulate_policy(env, Q, start_zone=start_zone, start_t=0, steps=steps)
        animate_route(
            centroids=centers,
            path_zones=path_zones,
            out_path=OUTPUTS_DIR / "agent_route.gif",
            fps=DEFAULTS["RL"]["ANIMATION_FPS"],
        )
    else:
        paths, rewards = simulate_multi_agents(env, Q, n_agents=args.n_agents, start_t=0, steps=steps, seed=args.seed)
        animate_routes_multi(
            centroids=centers,
            paths=paths,
            out_path=OUTPUTS_DIR / "agent_routes_multi.gif",
            fps=DEFAULTS["RL"]["ANIMATION_FPS"],
        )

    print("[i] Outputs saved to:")
    print(f" - {OUTPUTS_DIR / 'cluster_map.png'}")
    print(f" - {OUTPUTS_DIR / 'demand_heatmap.png'}")
    print(f" - {OUTPUTS_DIR / f'demand_forecast_zone_{top_zone}.png'}")
    if args.n_agents <= 1:
        print(f" - {OUTPUTS_DIR / 'agent_route.gif'}")
    else:
        print(f" - {OUTPUTS_DIR / 'agent_routes_multi.gif'}")
    print(f" - {OUTPUTS_DIR / 'demand_model.joblib'}")


if __name__ == "__main__":
    main()
