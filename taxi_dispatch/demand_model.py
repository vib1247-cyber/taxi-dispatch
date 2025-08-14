from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from dataclasses import dataclass
from typing import Dict, Any, Optional

try:
    from xgboost import XGBRegressor  # type: ignore
    _HAS_XGB = True
except Exception:  # pragma: no cover
    XGBRegressor = None
    _HAS_XGB = False


def train_random_forest_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    trees: int = 300,
    seed: int = 42,
) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=trees,
        random_state=seed,
        n_jobs=-1,
        oob_score=False,
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_regression(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    # Compute RMSE without relying on the 'squared' argument for compatibility with older sklearn versions
    mse = mean_squared_error(y_test, pred)
    rmse = float(np.sqrt(mse))
    return {"mae": float(mae), "rmse": float(rmse)}


@dataclass
class DemandModelWrapper:
    """Serializable wrapper for demand model and its feature layout.

    feature_cols: exact model input columns in order
    zone_levels: list of zones used during training (ints)
    hour_levels: list of hours 0..23 used during training
    dow_levels: list of day-of-week 0..6 used during training
    model_type: 'rf' or 'xgb'
    """

    model: Any
    feature_cols: list
    zone_levels: list
    hour_levels: list
    dow_levels: list
    model_type: str = "rf"

    def predict_vector(self, X: pd.DataFrame) -> np.ndarray:
        # Ensure the same column order, add missing with zeros, drop extra
        Xc = X.copy()
        for c in self.feature_cols:
            if c not in Xc.columns:
                Xc[c] = 0.0
        Xc = Xc[self.feature_cols]
        return self.model.predict(Xc)


def build_api_feature_row(
    *,
    zone: int,
    timestamp_hour: pd.Timestamp,
    extra_numeric: Optional[Dict[str, float]],
    wrapper: DemandModelWrapper,
) -> pd.DataFrame:
    hour = int(timestamp_hour.hour)
    dow = int(timestamp_hour.dayofweek)
    # One-hot columns
    cols = {}
    z_col = f"zone_{zone}"
    h_col = f"hour_{hour}"
    d_col = f"dow_{dow}"
    cols[z_col] = 1
    cols[h_col] = 1
    cols[d_col] = 1
    # Extra numeric features (if expected)
    for c in (extra_numeric or {}):
        if c in wrapper.feature_cols:
            cols[c] = float(extra_numeric[c])
    X = pd.DataFrame([cols])
    return X


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    model_type: str = "rf",
    seed: int = 42,
    tune: bool = False,
    n_jobs: int = -1,
) -> Any:
    model_type = model_type.lower()
    if model_type == "rf":
        base = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=n_jobs)
        if not tune:
            base.fit(X_train, y_train)
            return base
        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
        search = GridSearchCV(base, param_grid, cv=3, n_jobs=n_jobs, scoring="neg_mean_absolute_error")
        search.fit(X_train, y_train)
        return search.best_estimator_
    elif model_type == "xgb":
        if not _HAS_XGB:
            raise RuntimeError("xgboost is not installed; cannot use model_type='xgb'")
        base = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=n_jobs,
        )
        if not tune:
            base.fit(X_train, y_train)
            return base
        param_dist = {
            "n_estimators": [300, 500, 800],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
        }
        search = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=10,
            cv=3,
            random_state=seed,
            n_jobs=n_jobs,
            scoring="neg_mean_absolute_error",
        )
        search.fit(X_train, y_train)
        return search.best_estimator_
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
