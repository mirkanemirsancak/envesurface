#!/usr/bin/env python3
"""Generate publication-friendly visuals inside this code-only repository."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CloughTocher2DInterpolator, RBFInterpolator

ROOT = Path(__file__).resolve().parent
VIS_DIR = ROOT / "visuals"
RES_DIR = ROOT / "results"
MULTI_DIR = ROOT / "multi_run"


def save(fig: plt.Figure, name: str) -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(VIS_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(VIS_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_rmse_boxplots(runs: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, regime in zip(axs, ["noise_free", "noisy"]):
        sub = runs[runs["regime"] == regime]
        labels, data = [], []
        for out in ["Output1", "Output2", "Output3"]:
            for method in ["cubic", "rbf"]:
                vals = sub[(sub["output"] == out) & (sub["method"] == method)]["rmse"].dropna().to_numpy()
                if vals.size == 0:
                    continue
                labels.append(f"{out}\n{method}")
                data.append(vals)
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title(f"RMSE distribution ({regime})")
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", rotation=20, labelsize=9)
    save(fig, "rmse_boxplots")


def plot_r2_negative_rate(failures: pd.DataFrame) -> None:
    noisy = failures[failures["regime"] == "noisy"].copy()
    grouped = (
        noisy.groupby(["output", "method"], as_index=False)["negative_r2_rate"]
        .mean()
        .sort_values(["output", "method"])
    )

    outputs = ["Output1", "Output2", "Output3"]
    methods = ["cubic", "rbf"]
    x = np.arange(len(outputs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    vals_c = [float(grouped[(grouped["output"] == o) & (grouped["method"] == "cubic")]["negative_r2_rate"].iloc[0]) for o in outputs]
    vals_r = [float(grouped[(grouped["output"] == o) & (grouped["method"] == "rbf")]["negative_r2_rate"].iloc[0]) for o in outputs]
    ax.bar(x - w / 2, vals_c, width=w, label="cubic")
    ax.bar(x + w / 2, vals_r, width=w, label="rbf")
    ax.set_xticks(x)
    ax.set_xticklabels(outputs)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Negative R2 rate")
    ax.set_title("Noisy regime failure rate across outputs")
    ax.legend()
    save(fig, "negative_r2_rate_noisy")


def plot_rmse_seed_scatter(failures: pd.DataFrame) -> None:
    noisy = failures[failures["regime"] == "noisy"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, out in zip(axs, ["Output1", "Output2", "Output3"]):
        sub = noisy[noisy["output"] == out]
        for method, marker in [("cubic", "o"), ("rbf", "x")]:
            s = sub[sub["method"] == method]
            ax.scatter(s["seed"], s["rmse_mean"], label=method, marker=marker)
        ax.set_title(out)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Mean RMSE")
        ax.grid(alpha=0.25)
    axs[0].legend()
    save(fig, "rmse_by_seed_noisy")


def f1(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    return x1**2 + x2 + np.sin(x3)


def plot_3d_slice_examples() -> None:
    rng = np.random.default_rng(42)

    x1_levels = np.linspace(1.0, 2.0, 4)
    x2_levels = np.linspace(0.5, 1.5, 4)
    fixed_x3 = 3.0

    X1, X2 = np.meshgrid(x1_levels, x2_levels, indexing="ij")
    x1 = X1.ravel()
    x2 = X2.ravel()
    x3 = np.full_like(x1, fixed_x3, dtype=float)

    z_true_nodes = f1(x1, x2, x3)

    perm = rng.permutation(x1.size)
    n_train = 11
    tr = perm[:n_train]

    xy_nodes = np.column_stack([x1, x2])

    grid_u = np.linspace(x1.min(), x1.max(), 60)
    grid_v = np.linspace(x2.min(), x2.max(), 60)
    GU, GV = np.meshgrid(grid_u, grid_v)
    GXY = np.column_stack([GU.ravel(), GV.ravel()])
    GZ_true = (GU**2 + GV + np.sin(fixed_x3))

    z_obs_clean = z_true_nodes.copy()
    z_obs_noisy = z_true_nodes + rng.normal(0, 0.1, size=z_true_nodes.shape)

    cubic_clean = CloughTocher2DInterpolator(xy_nodes[tr], z_obs_clean[tr])
    rbf_clean = RBFInterpolator(xy_nodes[tr], z_obs_clean[tr], kernel="multiquadric", epsilon=1.0, smoothing=0.0)

    cubic_noisy = CloughTocher2DInterpolator(xy_nodes[tr], z_obs_noisy[tr])
    rbf_noisy = RBFInterpolator(xy_nodes[tr], z_obs_noisy[tr], kernel="multiquadric", epsilon=1.0, smoothing=0.0)

    GZ_cubic_clean = cubic_clean(GXY).reshape(GU.shape)
    GZ_rbf_clean = rbf_clean(GXY).reshape(GU.shape)
    GZ_cubic_noisy = cubic_noisy(GXY).reshape(GU.shape)
    GZ_rbf_noisy = rbf_noisy(GXY).reshape(GU.shape)

    panels = [
        ("Ground truth", GZ_true),
        ("Cubic (noise-free)", GZ_cubic_clean),
        ("RBF (noise-free)", GZ_rbf_clean),
        ("Cubic (noisy)", GZ_cubic_noisy),
        ("RBF (noisy)", GZ_rbf_noisy),
    ]

    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    for i, (title, z_grid) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 3, i, projection="3d")
        ax.plot_surface(GU, GV, z_grid, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
        ax.scatter(x1[tr], x2[tr], z_obs_noisy[tr] if "noisy" in title else z_obs_clean[tr], c="k", s=18)
        ax.set_title(title)
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.set_zlabel("Output1")
        ax.view_init(elev=28, azim=-132)

    save(fig, "slice_3d_surfaces")


def main() -> None:
    runs = pd.read_csv(RES_DIR / "interpolation_runs.csv")
    failures = pd.read_csv(MULTI_DIR / "failure_patterns.csv")

    plot_rmse_boxplots(runs)
    plot_r2_negative_rate(failures)
    plot_rmse_seed_scatter(failures)
    plot_3d_slice_examples()

    print(f"Generated visuals in: {VIS_DIR}")


if __name__ == "__main__":
    main()
