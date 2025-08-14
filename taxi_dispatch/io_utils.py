from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import yaml
import joblib
import numpy as np


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r") as f:
        return yaml.safe_load(f) or {}


def merge_dicts(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def save_joblib(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, p)


def load_joblib(path: str | Path) -> Any:
    return joblib.load(Path(path))


def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        json.dump(obj, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


def save_npy(arr: np.ndarray, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr)
