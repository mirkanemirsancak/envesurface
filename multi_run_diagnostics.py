#!/usr/bin/env python3
"""Run manuscript protocol with multiple seeds and summarize failure patterns."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import interpolation_experiments as ie

SEEDS = [42, 52, 62, 72, 82]
REPEATS = 40
OUT_DIR = Path("multi_run")


def summarize_failures(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["seed", "regime", "output", "method"], sort=True)
    for (seed, regime, output, method), g in grouped:
        rows.append(
            {
                "seed": seed,
                "regime": regime,
                "output": output,
                "method": method,
                "runs": int(len(g)),
                "rmse_mean": float(g["rmse"].mean()),
                "rmse_p95": float(g["rmse"].quantile(0.95)),
                "r2_mean": float(g["r2"].mean()),
                "negative_r2_rate": float((g["r2"] < 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ie.N_REPEATS = REPEATS

    outputs = [
        ie.OutputSpec("Output1", ie.f1, 0.1),
        ie.OutputSpec("Output2", ie.f2, 1.0),
        ie.OutputSpec("Output3", ie.f3, 2.0),
    ]

    all_runs = []
    all_summary = []

    for seed in SEEDS:
        clean = ie.run_experiment_regime("noise_free", outputs, noisy=False, rng_seed=seed)
        noisy = ie.run_experiment_regime("noisy", outputs, noisy=True, rng_seed=seed + 1)

        runs = pd.concat([clean, noisy], ignore_index=True)
        runs.insert(0, "seed", seed)
        summary = ie.summarize(runs.drop(columns=["seed"]))
        summary.insert(0, "seed", seed)

        all_runs.append(runs)
        all_summary.append(summary)

    runs_df = pd.concat(all_runs, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)
    failure_df = summarize_failures(runs_df)

    runs_path = OUT_DIR / "interpolation_runs_multi.csv"
    summary_path = OUT_DIR / "interpolation_summary_multi.csv"
    failure_path = OUT_DIR / "failure_patterns.csv"
    report_path = OUT_DIR / "run_report.md"

    runs_df.to_csv(runs_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    failure_df.to_csv(failure_path, index=False)

    noisy_worst = (
        failure_df[failure_df["regime"] == "noisy"]
        .sort_values(["negative_r2_rate", "rmse_p95"], ascending=[False, False])
        .head(8)
    )

    lines = []
    lines.append("# Multi-run Diagnostics (Manuscript Protocol)")
    lines.append("")
    lines.append(f"- Seeds: {SEEDS}")
    lines.append(f"- Repeats per slice: {REPEATS}")
    lines.append(f"- Total run rows: {len(runs_df)}")
    lines.append("")
    lines.append("## Key Failure Signals (noisy regime)")
    lines.append("")
    for _, r in noisy_worst.iterrows():
        lines.append(
            f"- seed={int(r['seed'])}, {r['output']} {r['method']}: "
            f"neg_R2_rate={r['negative_r2_rate']:.2%}, rmse_mean={r['rmse_mean']:.3f}, rmse_p95={r['rmse_p95']:.3f}, r2_mean={r['r2_mean']:.3f}"
        )

    lines.append("")
    lines.append("## Typical Stable Signals (noise-free regime)")
    lines.append("")

    stable = (
        failure_df[failure_df["regime"] == "noise_free"]
        .sort_values(["r2_mean", "rmse_mean"], ascending=[False, True])
        .head(8)
    )
    for _, r in stable.iterrows():
        lines.append(
            f"- seed={int(r['seed'])}, {r['output']} {r['method']}: "
            f"r2_mean={r['r2_mean']:.3f}, rmse_mean={r['rmse_mean']:.3f}, neg_R2_rate={r['negative_r2_rate']:.2%}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    meta = {
        "seeds": SEEDS,
        "repeats": REPEATS,
        "rows_runs": int(len(runs_df)),
        "rows_summary": int(len(summary_df)),
        "rows_failures": int(len(failure_df)),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {runs_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {failure_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
