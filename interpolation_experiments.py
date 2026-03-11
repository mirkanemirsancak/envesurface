#!/usr/bin/env python3
"""Reproducible cubic-vs-RBF interpolation benchmark for the paper workflow.

Protocol implemented from the manuscript:
- synthetic 4x4x3 factorial design over (X1, X2, X3)
- outputs: f1, f2, f3 with noise sigmas (0.1, 1.0, 2.0)
- regimes: noise-free and noisy
- slice-wise train/test splits: 70/30, repeated 40 times
- methods: CloughTocher2DInterpolator and multiquadric RBF (smoothing=0)
- metrics: RMSE, MAE, R2 with bootstrap confidence intervals (B=1000)

Artifacts:
- other/code/surfacegraph_arxiv/results/interpolation_runs.csv
- other/code/surfacegraph_arxiv/results/interpolation_summary.csv
- other/code/surfacegraph_arxiv/results/interpolation_meta.json
- tables/new_results.csv, tables/new_results.tex
- tables/experiment_settings.csv, tables/experiment_settings.tex
"""

from __future__ import annotations

import json
import math
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import psutil
from scipy.interpolate import CloughTocher2DInterpolator

try:
    from scipy.interpolate import RBFInterpolator

    HAS_RBF_INTERPOLATOR = True
except Exception:
    from scipy.interpolate import Rbf

    HAS_RBF_INTERPOLATOR = False


SEED = 42
N_REPEATS = 40
TRAIN_FRAC = 0.7
BOOTSTRAP_B = 1000
EPSILON = 1.0


@dataclass(frozen=True)
class OutputSpec:
    name: str
    true_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    noise_sigma: float


def find_project_root(start: Path) -> Path:
    """Locate project root, preferring paper layout, then local script dir."""
    for parent in [start, *start.parents]:
        if (parent / "tables").is_dir() and (parent / "figures").is_dir():
            return parent
    for parent in [start, *start.parents]:
        if (parent / ".git").is_dir():
            return parent
    return start


def f1(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    return x1**2 + x2 + np.sin(x3)


def f2(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    return x1 * x2 + x3**2


def f3(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> np.ndarray:
    return np.cos(x1) + x2 * x3


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def bootstrap_ci(values: np.ndarray, b: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    samples = np.empty(b, dtype=float)
    n = values.size
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(np.mean(values[idx]))
    return (float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975)))


def generate_base_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1 = np.linspace(1.0, 2.0, 4)
    x2 = np.linspace(0.5, 1.5, 4)
    x3 = np.linspace(2.0, 4.0, 3)
    X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing="ij")
    return X1, X2, X3


def make_slice_points(
    X1: np.ndarray,
    X2: np.ndarray,
    X3: np.ndarray,
    fixed_var: str,
    fixed_value: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if fixed_var == "X1":
        mask = np.isclose(X1, fixed_value)
        u = X2[mask]
        v = X3[mask]
        return u, v, X1[mask], X2[mask], X3[mask]

    if fixed_var == "X2":
        mask = np.isclose(X2, fixed_value)
        u = X1[mask]
        v = X3[mask]
        return u, v, X1[mask], X2[mask], X3[mask]

    mask = np.isclose(X3, fixed_value)
    u = X1[mask]
    v = X2[mask]
    return u, v, X1[mask], X2[mask], X3[mask]


def fit_predict_cubic(train_xy: np.ndarray, train_z: np.ndarray, test_xy: np.ndarray) -> np.ndarray:
    interp = CloughTocher2DInterpolator(train_xy, train_z)
    return np.asarray(interp(test_xy), dtype=float)


def fit_predict_rbf(train_xy: np.ndarray, train_z: np.ndarray, test_xy: np.ndarray) -> np.ndarray:
    if HAS_RBF_INTERPOLATOR:
        interp = RBFInterpolator(
            train_xy,
            train_z,
            kernel="multiquadric",
            epsilon=EPSILON,
            smoothing=0.0,
        )
        return np.asarray(interp(test_xy), dtype=float)

    rbf = Rbf(train_xy[:, 0], train_xy[:, 1], train_z, function="multiquadric", epsilon=EPSILON, smooth=0.0)
    return np.asarray(rbf(test_xy[:, 0], test_xy[:, 1]), dtype=float)


def run_experiment_regime(regime_name: str, outputs: list[OutputSpec], noisy: bool, rng_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    X1, X2, X3 = generate_base_grid()

    rows: list[dict] = []

    for out in outputs:
        fixed_specs = [
            ("X1", np.unique(X1)),
            ("X2", np.unique(X2)),
            ("X3", np.unique(X3)),
        ]

        for fixed_var, values in fixed_specs:
            for fv in values:
                u, v, sx1, sx2, sx3 = make_slice_points(X1, X2, X3, fixed_var, float(fv))

                z_true = out.true_func(sx1, sx2, sx3)
                z_obs = z_true.copy()
                if noisy:
                    z_obs = z_obs + rng.normal(0.0, out.noise_sigma, size=z_obs.shape)

                n = u.size
                if n < 8:
                    continue

                n_train = int(math.ceil(TRAIN_FRAC * n))
                n_train = min(max(n_train, 6), n - 2)

                xy = np.column_stack([u, v])

                for rep in range(N_REPEATS):
                    perm = rng.permutation(n)
                    train_idx = perm[:n_train]
                    test_idx = perm[n_train:]

                    train_xy = xy[train_idx]
                    test_xy = xy[test_idx]
                    train_z = z_obs[train_idx]
                    test_true = z_true[test_idx]

                    for method in ("cubic", "rbf"):
                        try:
                            pred = (
                                fit_predict_cubic(train_xy, train_z, test_xy)
                                if method == "cubic"
                                else fit_predict_rbf(train_xy, train_z, test_xy)
                            )
                        except Exception:
                            continue

                        if np.any(~np.isfinite(pred)):
                            continue

                        rows.append(
                            {
                                "regime": regime_name,
                                "output": out.name,
                                "fixed_var": fixed_var,
                                "fixed_value": float(fv),
                                "repeat": rep,
                                "method": method,
                                "n_train": int(train_idx.size),
                                "n_test": int(test_idx.size),
                                "rmse": rmse(test_true, pred),
                                "mae": mae(test_true, pred),
                                "r2": r2(test_true, pred),
                            }
                        )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["regime", "output", "method"]
    summary_rows: list[dict] = []

    for keys, g in df.groupby(group_cols, sort=True):
        regime, output, method = keys
        rmse_vals = g["rmse"].to_numpy(dtype=float)
        mae_vals = g["mae"].to_numpy(dtype=float)
        r2_vals = g["r2"].to_numpy(dtype=float)

        rmse_ci = bootstrap_ci(rmse_vals, BOOTSTRAP_B, SEED + 101)
        mae_ci = bootstrap_ci(mae_vals, BOOTSTRAP_B, SEED + 202)
        r2_ci = bootstrap_ci(r2_vals[np.isfinite(r2_vals)], BOOTSTRAP_B, SEED + 303)

        summary_rows.append(
            {
                "regime": regime,
                "output": output,
                "method": method,
                "n_runs": int(len(g)),
                "rmse_mean": float(np.mean(rmse_vals)),
                "rmse_ci_low": rmse_ci[0],
                "rmse_ci_high": rmse_ci[1],
                "mae_mean": float(np.mean(mae_vals)),
                "mae_ci_low": mae_ci[0],
                "mae_ci_high": mae_ci[1],
                "r2_mean": float(np.nanmean(r2_vals)),
                "r2_ci_low": r2_ci[0],
                "r2_ci_high": r2_ci[1],
            }
        )

    return pd.DataFrame(summary_rows).sort_values(group_cols).reset_index(drop=True)


def write_latex_table(df: pd.DataFrame, path: Path, alignment: str, headers: list[str], formats: list[Callable[[object], str]]) -> None:
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule"]
    lines.append(" & ".join(headers) + r" \\")
    lines.append("\\midrule")

    for _, row in df.iterrows():
        fields = [fmt(row[col]) for col, fmt in zip(df.columns, formats)]
        lines.append(" & ".join(fields) + r" \\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_paper_tables(summary: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    new_results = summary[["regime", "output", "method", "n_runs", "rmse_mean", "r2_mean"]].copy()
    new_results.columns = ["Regime", "Output", "Method", "Runs", "RMSE", "R2"]
    new_results.to_csv(tables_dir / "new_results.csv", index=False)

    write_latex_table(
        df=new_results,
        path=tables_dir / "new_results.tex",
        alignment="llcrrr",
        headers=["Regime", "Output", "Method", "Runs", "RMSE", r"$R^2$"],
        formats=[
            str,
            str,
            str,
            lambda x: str(int(x)),
            lambda x: f"{float(x):.3f}",
            lambda x: f"{float(x):.3f}",
        ],
    )

    exp_settings = pd.DataFrame(
        [
            ("random_seed", SEED),
            ("repeats_per_slice", N_REPEATS),
            ("train_fraction", TRAIN_FRAC),
            ("bootstrap_resamples", BOOTSTRAP_B),
            ("rbf_kernel", "multiquadric"),
            ("rbf_smoothing", 0.0),
            ("cubic_interpolator", "CloughTocher2DInterpolator"),
            ("noise_sigma_output1", 0.1),
            ("noise_sigma_output2", 1.0),
            ("noise_sigma_output3", 2.0),
        ],
        columns=["Setting", "Value"],
    )
    exp_settings.to_csv(tables_dir / "experiment_settings.csv", index=False)

    write_latex_table(
        df=exp_settings,
        path=tables_dir / "experiment_settings.tex",
        alignment="ll",
        headers=["Setting", "Value"],
        formats=[str, lambda x: str(x)],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interpolation benchmark.")
    parser.add_argument("--seed", type=int, default=SEED, help="Base random seed.")
    parser.add_argument("--repeats", type=int, default=N_REPEATS, help="Repeated splits per slice.")
    parser.add_argument("--results-dir", type=str, default="results", help="Output directory for run artifacts.")
    parser.add_argument("--tables-dir", type=str, default="tables", help="Output directory for table artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    project_root = find_project_root(script_path.parent)

    results_dir = (script_path.parent / args.results_dir).resolve()
    tables_dir = (project_root / args.tables_dir).resolve()

    process = psutil.Process()
    t0 = time.time()

    outputs = [
        OutputSpec("Output1", f1, 0.1),
        OutputSpec("Output2", f2, 1.0),
        OutputSpec("Output3", f3, 2.0),
    ]

    results_dir.mkdir(parents=True, exist_ok=True)

    global N_REPEATS
    N_REPEATS = int(args.repeats)

    df_clean = run_experiment_regime("noise_free", outputs, noisy=False, rng_seed=args.seed)
    df_noisy = run_experiment_regime("noisy", outputs, noisy=True, rng_seed=args.seed + 1)

    all_runs = pd.concat([df_clean, df_noisy], ignore_index=True)
    summary = summarize(all_runs)

    runs_path = results_dir / "interpolation_runs.csv"
    summary_path = results_dir / "interpolation_summary.csv"
    meta_path = results_dir / "interpolation_meta.json"

    all_runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)

    export_paper_tables(summary, tables_dir)

    elapsed = time.time() - t0
    rss_mb = process.memory_info().rss / 1024**2

    meta = {
        "seed": int(args.seed),
        "n_repeats": N_REPEATS,
        "train_fraction": TRAIN_FRAC,
        "bootstrap_B": BOOTSTRAP_B,
        "epsilon": EPSILON,
        "rbf_backend": "RBFInterpolator" if HAS_RBF_INTERPOLATOR else "Rbf",
        "runtime_seconds": elapsed,
        "memory_rss_mb": rss_mb,
        "rows_runs": int(len(all_runs)),
        "rows_summary": int(len(summary)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Runs CSV: {runs_path}")
    print(f"Summary CSV: {summary_path}")
    print(f"Meta JSON: {meta_path}")
    print(f"Tables Dir: {tables_dir}")
    print(f"Runtime: {elapsed:.2f} s")
    print(f"RSS memory: {rss_mb:.2f} MB")
    print("\nTopline summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
