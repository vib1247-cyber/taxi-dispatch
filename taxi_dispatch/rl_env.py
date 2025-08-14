from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _pairwise_distance_km(centroids: np.ndarray) -> np.ndarray:
    Z = centroids.shape[0]
    D = np.zeros((Z, Z), dtype=float)
    for i in range(Z):
        for j in range(Z):
            if i == j:
                D[i, j] = 0.0
            else:
                D[i, j] = _haversine_km(centroids[i, 0], centroids[i, 1], centroids[j, 0], centroids[j, 1])
    return D


@dataclass
class TaxiZoneEnv:
    """Simple time-aware environment for zone movement.

    - State: (t, z) where t in [0..T-1], z in [0..Z-1]
    - Actions: choose next zone z' in [0..Z-1]
    - Reward: demand[t_next, z'] - move_cost * distance_km(z, z')
    - Transition: t_next = (t + 1) % T
    """

    centroids: np.ndarray  # (Z, 2) [lat, lon]
    demand_matrix: np.ndarray  # (T, Z)
    move_cost: float = 0.05
    distance_matrix: np.ndarray | None = None  # optional precomputed distances (km)

    def __post_init__(self):
        self.T, self.Z = self.demand_matrix.shape
        if self.distance_matrix is not None:
            self.D = self.distance_matrix
        else:
            self.D = _pairwise_distance_km(self.centroids)

    def step(self, t: int, z: int, action_next_zone: int):
        t_next = (t + 1) % self.T
        z_next = int(action_next_zone)
        reward = float(self.demand_matrix[t_next, z_next]) - self.move_cost * float(self.D[z, z_next])
        return t_next, z_next, reward


def train_q_learning(
    env: TaxiZoneEnv,
    *,
    episodes: int = 300,
    alpha: float = 0.2,
    gamma: float = 0.95,
    eps_start: float = 0.8,
    eps_min: float = 0.05,
    eps_decay: float = 0.995,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.T, env.Z, env.Z), dtype=float)  # Q[t, z, action->z']

    eps = eps_start
    for ep in range(episodes):
        t = rng.integers(0, env.T)
        z = rng.integers(0, env.Z)
        steps = env.T  # one pass over horizon per episode
        for _ in range(steps):
            if rng.random() < eps:
                a = int(rng.integers(0, env.Z))
            else:
                a = int(np.argmax(Q[t, z]))
            t_next, z_next, r = env.step(t, z, a)
            # Q-learning update
            best_next = np.max(Q[t_next, z_next])
            td_target = r + gamma * best_next
            Q[t, z, a] = (1 - alpha) * Q[t, z, a] + alpha * td_target
            t, z = t_next, z_next
        eps = max(eps_min, eps * eps_decay)
    return Q
