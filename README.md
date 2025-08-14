# Smart Taxi Dispatch System

A modular Python system that:

- Uses KMeans to cluster a city into high-demand zones
- Trains Random Forest models (regression and optional classification) to forecast ride demand by time and zone
- Applies Q-learning to optimize taxi movement across zones to maximize pickups
- Works with real CSV trip datasets or synthetic data
- Produces visual outputs: cluster maps, demand heatmaps, demand forecast plots, and agent route animations

## Features

- KMeans clustering on latitude/longitude to detect zones
- RandomForestRegressor (and optional RandomForestClassifier) for demand prediction
- Tabular Q-learning with time-aware states and move-cost penalties
- Matplotlib/Seaborn-based visualizations, animated GIF route output
- Modular, documented code to support future dataset updates

## Project Structure

```
smart-taxi-dispatch/
├── README.md
├── requirements.txt
├── main.py
├── taxi_dispatch/
│   ├── __init__.py
│   ├── config.py
│   ├── data_ingest.py
│   ├── clustering.py
│   ├── features.py
│   ├── demand_model.py
│   ├── rl_env.py
│   ├── simulate.py
│   ├── visualize.py
│   ├── routing.py
│   ├── external_features.py
│   └── io_utils.py
├── api/
│   └── serve.py
├── dashboard/
│   └── app.py
├── tests/
│   └── test_pipeline.py
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml
├── data/
│   └── (optional real dataset here)
└── outputs/
    └── (generated visuals and results)
```

## Installation

1) Create a virtual environment (macOS/Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the full demo with synthetic data (default settings):

```bash
python main.py --generate-synth
```

Key arguments:

- `--data PATH`          Path to CSV with trip records
- `--time-col NAME`      Timestamp column name (default: `timestamp`)
- `--lat-col NAME`       Latitude column name (default: `lat`)
- `--lon-col NAME`       Longitude column name (default: `lon`)
- `--n-zones INT`        Number of KMeans clusters (default from config)
- `--days INT`           Days to synthesize (synthetic only)
- `--trips-per-day INT`  Trip count per day (synthetic only)
- `--episodes INT`       Q-learning training episodes
- `--seed INT`           Random seed

Outputs (saved under `outputs/`):

- `cluster_map.png` – KMeans zones over trip points
- `demand_heatmap.png` – Hour x Zone demand intensity
- `demand_forecast_zone_<id>.png` – Actual vs predicted demand for a sample zone
- `agent_route.gif` – Animated taxi route across zone centroids
- `predictions.csv` – Model hourly predictions by zone
- `actuals.csv` – Actual hourly demand by zone
- `centers.npy` – Zone centroids array (Z x 2)
- `distance_matrix.npy` – Pairwise zone distances (km)

## Real Dataset Format

Provide a CSV with at least these columns:

- `timestamp` (parseable datetime) – OR specify with `--time-col`
- `lat` – OR specify with `--lat-col`
- `lon` – OR specify with `--lon-col`

The system will:

1. Cluster trips into zones by KMeans on `(lat, lon)`
2. Aggregate demand as counts per hour per zone
3. Train the demand model and forecast demand
4. Train a Q-learning agent using the forecast as reward proxy
5. Simulate and visualize results

## Updating for New Datasets

- Adjust column names via CLI flags.
- If your dataset includes pre-aggregated counts, you can adapt `taxi_dispatch/data_ingest.py` to skip raw event -> aggregate steps.
- Extend features in `taxi_dispatch/features.py` (e.g., weather, events). Random Forests handle mixed features well.

## Notes

- This is a simplified research/education project, not production dispatch software.
- For geographic accuracy (distance, road network), integrate mapping libraries and graph routing in future work.

## API (FastAPI)

After running `main.py` to generate `outputs/` artifacts (`demand_model.joblib`, `centers.npy`, `distance_matrix.npy`), start the API:

```bash
uvicorn api.serve:app --reload --port 8000
```

Health:

```bash
curl http://localhost:8000/health
```

List zones and centers:

```bash
curl http://localhost:8000/zones
```

Predict demand:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"zone": 0, "timestamp": "2025-01-01T12:00:00", "extra": {}}'
```

Distance between zones:

```bash
curl -X POST http://localhost:8000/distance \
  -H 'Content-Type: application/json' \
  -d '{"from_zone": 0, "to_zone": 1}'
```

## Dashboard (Streamlit)

Start the dashboard to explore cluster maps, demand heatmap, forecast vs actual, on-demand predictions, distances, and route GIFs:

```bash
streamlit run dashboard/app.py --server.port 8501
```

Open http://localhost:8501

## Docker

Build and run API:

```bash
docker build -t taxi-api -f Dockerfile.api .
docker run --rm -p 8000:8000 -v $(pwd)/outputs:/app/outputs taxi-api
```

Build and run Dashboard:

```bash
docker build -t taxi-dashboard -f Dockerfile.dashboard .
docker run --rm -p 8501:8501 -v $(pwd)/outputs:/app/outputs taxi-dashboard
```

Using docker-compose:

```bash
docker compose up --build
```

This mounts `./outputs` so both services read the latest artifacts from training.

## Testing

Run unit tests:

```bash
pytest -q
```
