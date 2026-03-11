#!/usr/bin/env python3
"""Generate publication-friendly visuals inside this code-only repository."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def main() -> None:
    runs = pd.read_csv(RES_DIR / "interpolation_runs.csv")
    failures = pd.read_csv(MULTI_DIR / "failure_patterns.csv")

    plot_rmse_boxplots(runs)
    plot_r2_negative_rate(failures)
    plot_rmse_seed_scatter(failures)

    print(f"Generated visuals in: {VIS_DIR}")


if __name__ == "__main__":
    main()
