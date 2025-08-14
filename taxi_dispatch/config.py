from pathlib import Path

DEFAULTS = {
    "N_ZONES": 8,
    "RANDOM_SEED": 42,
    "RF_TREES": 300,
    "TEST_RATIO": 0.2,
    "RL": {
        "ALPHA": 0.2,
        "GAMMA": 0.95,
        "EPSILON_START": 0.8,
        "EPSILON_MIN": 0.05,
        "EPSILON_DECAY": 0.995,
        "EPISODES": 300,
        "MOVE_COST": 0.05,  # cost per km moved between zones
        "ANIMATION_FPS": 5,
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
