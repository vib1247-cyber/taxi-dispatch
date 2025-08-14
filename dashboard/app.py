from __future__ import annotations

from pathlib import Path
import json
import numpy as np
from datetime import datetime, time
import pandas as pd
import streamlit as st
import plotly.express as px

from taxi_dispatch.io_utils import load_joblib
from taxi_dispatch.demand_model import DemandModelWrapper, build_api_feature_row

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

st.set_page_config(page_title="Smart Taxi Dispatch Dashboard", layout="wide")

st.title("🚕 Smart Taxi Dispatch Dashboard")

# Helpers
@st.cache_data(show_spinner=False)
def load_centers() -> np.ndarray | None:
    path = OUTPUTS_DIR / "centers.npy"
    if path.exists():
        return np.load(path)
    return None

@st.cache_data(show_spinner=False)
def load_distance_matrix() -> np.ndarray | None:
    path = OUTPUTS_DIR / "distance_matrix.npy"
    if path.exists():
        return np.load(path)
    return None

@st.cache_data(show_spinner=False)
def load_preds_actuals() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    pred_path = OUTPUTS_DIR / "predictions.csv"
    actual_path = OUTPUTS_DIR / "actuals.csv"
    pred_df = pd.read_csv(pred_path) if pred_path.exists() else None
    act_df = pd.read_csv(actual_path) if actual_path.exists() else None
    return pred_df, act_df

@st.cache_resource(show_spinner=False)
def load_wrapper() -> DemandModelWrapper | None:
    path = OUTPUTS_DIR / "demand_model.joblib"
    if path.exists():
        return load_joblib(path)
    return None

centers = load_centers()
D = load_distance_matrix()
pred_df, actual_df = load_preds_actuals()
wrapper = load_wrapper()

# Top row: images
st.subheader("Maps")
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Clusters Map")
    img_path = OUTPUTS_DIR / "cluster_map.png"
    if img_path.exists():
        st.image(str(img_path), use_column_width=True)
    else:
        st.info("Run main.py to generate outputs/cluster_map.png")
with col2:
    st.markdown("### Demand Heatmap")
    heat_path = OUTPUTS_DIR / "demand_heatmap.png"
    if heat_path.exists():
        st.image(str(heat_path), use_column_width=True)
    else:
        st.info("Run main.py to generate outputs/demand_heatmap.png")

st.markdown("---")

# Forecast vs actual explorer
st.subheader("Forecast vs Actual Demand")
if pred_df is not None and actual_df is not None:
    zones = sorted(set(pred_df["zone"].unique()))
    col1, col2 = st.columns([1, 3])
    with col1:
        sel_zone = st.selectbox("Select Zone", zones, index=0)
    
    a = actual_df[actual_df["zone"] == sel_zone].sort_values("timestamp_hour")
    p = pred_df[pred_df["zone"] == sel_zone].sort_values("timestamp_hour")
    
    # Ensure we have matching timestamps
    merged = a[["timestamp_hour", "actual"]].merge(
        p[["timestamp_hour", "pred"]], 
        on="timestamp_hour", 
        how="inner"
    )
    
    # Create a more informative plot
    fig = px.line(
        merged, 
        x="timestamp_hour", 
        y=["actual", "pred"], 
        labels={"value": "Demand", "timestamp_hour": "Time"},
        title=f"Demand Forecast vs Actual for Zone {sel_zone}",
        template="plotly_white"
    )
    
    # Update line styles and legend
    fig.update_traces(
        selector={"name": "actual"}, 
        line=dict(color="blue"),
        name="Actual"
    )
    fig.update_traces(
        selector={"name": "pred"}, 
        line=dict(color="red", dash="dash"),
        name="Forecast"
    )
    
    # Improve layout
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Demand",
        legend_title="",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add some statistics
    if not merged.empty:
        mae = (merged["actual"] - merged["pred"]).abs().mean()
        st.metric("Mean Absolute Error", f"{mae:.2f}")
else:
    st.info("Missing predictions.csv/actuals.csv. Run main.py to generate them.")

st.markdown("---")

# On-demand prediction
st.subheader("On-demand Prediction")
if wrapper is not None and centers is not None:
    zones = list(range(len(centers)))
    cz = st.selectbox("Zone", zones, index=0, key="pred_zone")
    # Use separate date and time inputs for Streamlit 1.19.0 compatibility
    date = st.date_input("Date")
    time = st.time_input("Time")
    ts = datetime.combine(date, time)
    extra = st.text_input("Extra features (JSON, numeric values)", value="{}")
    if st.button("Predict"):
        try:
            extra_dict = {} if not extra.strip() else json.loads(extra)
            if not isinstance(extra_dict, dict):
                raise ValueError("JSON must decode to an object")
        except Exception:
            st.error("Invalid JSON for extra features; expected e.g. {\"temp\": 21.5}")
            extra_dict = {}
        ts_pd = pd.to_datetime(ts)
        X = build_api_feature_row(zone=int(cz), timestamp_hour=ts_pd, extra_numeric=extra_dict, wrapper=wrapper)
        yhat = float(wrapper.predict_vector(X)[0])
        st.success(f"Predicted demand: {yhat:.2f}")
else:
    st.info("Model not found. Train via main.py to create outputs/demand_model.joblib and centers.npy")

st.markdown("---")

# Distance lookup
st.subheader("Zone Distance (km)")
if D is not None and centers is not None:
    zones = list(range(len(centers)))
    c1, c2 = st.columns(2)
    with c1:
        z1 = st.selectbox("From zone", zones, index=0)
    with c2:
        z2 = st.selectbox("To zone", zones, index=min(1, len(zones)-1))
    st.metric("Distance (km)", f"{float(D[int(z1), int(z2)]):.2f}")
else:
    st.info("Distance matrix not found. It is generated when running main.py.")

st.markdown("---")

# Route animations
st.subheader("Route Animations")
col1, col2 = st.columns(2)
with col1:
    gif_multi = OUTPUTS_DIR / "agent_routes_multi.gif"
    if gif_multi.exists():
        st.image(str(gif_multi))
    else:
        st.info("Multi-agent GIF not found.")
with col2:
    gif_single = OUTPUTS_DIR / "agent_route.gif"
    if gif_single.exists():
        st.image(str(gif_single))
    else:
        st.info("Single-agent GIF not found.")
