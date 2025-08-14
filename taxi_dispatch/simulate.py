from __future__ import annotations

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import imageio


def simulate_policy(env, Q: np.ndarray, *, start_zone: int, start_t: int, steps: int) -> Tuple[list, list]:
    """Follow greedy policy from Q for given steps.

    Returns (path_zones, rewards)
    """
    t, z = int(start_t), int(start_zone)
    path = [z]
    rewards = []
    for _ in range(steps):
        a = int(np.argmax(Q[t, z]))
        t, z_next, r = env.step(t, z, a)
        path.append(z_next)
        rewards.append(r)
        z = z_next
    return path, rewards


def animate_route(
    *,
    centroids: np.ndarray,
    path_zones: List[int],
    out_path,
    fps: int = 5,
):
    """Create a GIF animation of the agent's movement across zone centroids."""
    lat = centroids[:, 0]
    lon = centroids[:, 1]
    pad_lat = (lat.max() - lat.min()) * 0.1 + 1e-3
    pad_lon = (lon.max() - lon.min()) * 0.1 + 1e-3

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(lon.min() - pad_lon, lon.max() + pad_lon)
    ax.set_ylim(lat.min() - pad_lat, lat.max() + pad_lat)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Agent Route Across Zone Centroids")

    # Pre-plot centroids and labels
    ax.scatter(lon, lat, c="tab:blue", s=60, zorder=2)
    for i, (la, lo) in enumerate(zip(lat, lon)):
        ax.text(lo, la, str(i), fontsize=8, ha="center", va="bottom")

    line, = ax.plot([], [], c="tab:red", lw=2, zorder=3)
    marker, = ax.plot([], [], marker="o", c="tab:red", ms=8, zorder=4)

    frames = []
    with imageio.get_writer(out_path, mode="I", duration=1 / max(1, fps)) as writer:
        # initial frame
        p0 = int(path_zones[0])
        line.set_data([lon[p0]], [lat[p0]])
        marker.set_data([lon[p0]], [lat[p0]])
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(image)

        # path frames
        xs, ys = [lon[p0]], [lat[p0]]
        for z in path_zones[1:]:
            xs.append(lon[int(z)])
            ys.append(lat[int(z)])
            line.set_data(xs, ys)
            marker.set_data([xs[-1]], [ys[-1]])
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(image)

    plt.close(fig)


def simulate_multi_agents(
    env,
    Q: np.ndarray,
    *,
    n_agents: int,
    start_zones: List[int] | None = None,
    start_t: int = 0,
    steps: int = 100,
    seed: int = 42,
) -> Tuple[List[List[int]], List[List[float]]]:
    """Simulate multiple agents following greedy policy.

    Returns (paths_per_agent, rewards_per_agent)
    """
    rng = np.random.default_rng(seed)
    if start_zones is None:
        start_zones = [int(rng.integers(0, env.Z)) for _ in range(n_agents)]
    paths = []
    rewards = []
    for a_idx in range(n_agents):
        p, r = simulate_policy(env, Q, start_zone=int(start_zones[a_idx]), start_t=int(start_t), steps=int(steps))
        paths.append(p)
        rewards.append(r)
    return paths, rewards


def animate_routes_multi(
    *,
    centroids: np.ndarray,
    paths: List[List[int]],
    out_path,
    fps: int = 5,
):
    """Animate multiple taxi routes together."""
    lat = centroids[:, 0]
    lon = centroids[:, 1]
    pad_lat = (lat.max() - lat.min()) * 0.1 + 1e-3
    pad_lon = (lon.max() - lon.min()) * 0.1 + 1e-3

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(lon.min() - pad_lon, lon.max() + pad_lon)
    ax.set_ylim(lat.min() - pad_lat, lat.max() + pad_lat)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Multi-Agent Routes Across Zone Centroids")

    # Centroids and labels
    ax.scatter(lon, lat, c="tab:blue", s=60, zorder=2)
    for i, (la, lo) in enumerate(zip(lat, lon)):
        ax.text(lo, la, str(i), fontsize=8, ha="center", va="bottom")

    # Prepare per-agent lines
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ["tab:red", "tab:green", "tab:orange", "tab:purple"])  # fallback
    lines = []
    markers = []
    for idx, p in enumerate(paths):
        color = colors[idx % len(colors)]
        line, = ax.plot([], [], c=color, lw=2, zorder=3, label=f"Taxi {idx}")
        marker, = ax.plot([], [], marker="o", c=color, ms=6, zorder=4)
        lines.append(line)
        markers.append(marker)
    ax.legend(loc="best")

    max_len = max(len(p) for p in paths)
    with imageio.get_writer(out_path, mode="I", duration=1 / max(1, fps)) as writer:
        # initialize at first points
        xs = [[] for _ in paths]
        ys = [[] for _ in paths]
        for step in range(max_len):
            for i, p in enumerate(paths):
                if step >= len(p):
                    continue
                z = int(p[step])
                xs[i].append(lon[z])
                ys[i].append(lat[z])
                lines[i].set_data(xs[i], ys[i])
                markers[i].set_data([xs[i][-1]], [ys[i][-1]])
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            writer.append_data(image)
    plt.close(fig)
