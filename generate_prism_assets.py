#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator, RBFInterpolator

ROOT = Path('/Users/mirkanemirsancak/Desktop/surfacegraph_arxiv')
RESULTS = ROOT / 'other' / 'code' / 'surfacegraph_arxiv' / 'results'
FIG_DIR = ROOT / 'figures'
TAB_DIR = ROOT / 'tables'
CSV_DIR = ROOT / 'csv'

SEED = 42
np.random.seed(SEED)


def save_fig(fig, basename: str):
    fig.savefig(FIG_DIR / f'{basename}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{basename}.pdf', bbox_inches='tight')
    plt.close(fig)


def output_functions(x1, x2, x3):
    o1 = x1**2 + x2 + np.sin(x3)
    o2 = x1 * x2 + x3**2
    o3 = np.cos(x1) + x2 * x3
    return o1, o2, o3


def base_design_points() -> pd.DataFrame:
    x1 = np.linspace(1.0, 2.0, 4)
    x2 = np.linspace(0.5, 1.5, 4)
    x3 = np.linspace(2.0, 4.0, 3)
    X1, X2, X3 = np.meshgrid(x1, x2, x3, indexing='ij')
    o1, o2, o3 = output_functions(X1, X2, X3)
    return pd.DataFrame({
        'X1': X1.ravel(),
        'X2': X2.ravel(),
        'X3': X3.ravel(),
        'Output1_true': o1.ravel(),
        'Output2_true': o2.ravel(),
        'Output3_true': o3.ravel(),
    })


def make_workflow():
    steps = pd.DataFrame([
        (1, 'Design points (X1,X2,X3)', 'input'),
        (2, 'Generate outputs + optional noise', 'simulation'),
        (3, 'Slice-wise train/test splits', 'protocol'),
        (4, 'Fit cubic and RBF interpolants', 'modeling'),
        (5, 'Compute RMSE/MAE/R2', 'metrics'),
        (6, 'Bootstrap confidence intervals', 'uncertainty'),
        (7, 'Publish figures/tables for process decisions', 'reporting'),
    ], columns=['step_id', 'label', 'type'])
    edges = pd.DataFrame([(i, i+1) for i in range(1, 7)], columns=['from_step', 'to_step'])
    steps.to_csv(CSV_DIR / 'workflow_steps.csv', index=False)
    edges.to_csv(CSV_DIR / 'workflow_edges.csv', index=False)

    fig, ax = plt.subplots(figsize=(16, 4.4))
    ax.axis('off')
    x_positions = np.linspace(0.05, 0.95, len(steps))
    for i, (_, row) in enumerate(steps.iterrows()):
        x = x_positions[i]
        ax.text(x, 0.68, f"{int(row['step_id'])}", ha='center', va='center', fontsize=11,
                bbox=dict(boxstyle='circle,pad=0.3', facecolor='#d9edf7', edgecolor='#31708f'))
        wrapped = "\n".join(textwrap.wrap(row['label'], width=22))
        ax.text(x, 0.34, wrapped, ha='center', va='top', fontsize=9)
        if i < len(steps)-1:
            ax.annotate('', xy=(x_positions[i+1]-0.03, 0.68), xytext=(x+0.03, 0.68),
                        arrowprops=dict(arrowstyle='->', lw=1.4, color='black'))
    ax.set_title('Reproducible Interpolation Workflow', fontsize=14, pad=10)
    save_fig(fig, 'workflow')


def make_design_points(df: pd.DataFrame):
    df.to_csv(CSV_DIR / 'design_points.csv', index=False)

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df['X1'], df['X2'], df['X3'], c=df['Output1_true'], cmap='viridis', s=35)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    fig.colorbar(sc, ax=ax, shrink=0.7, label='Output1 true')
    ax.set_title('Design Points (48-point factorial grid)')
    save_fig(fig, 'design_points')


def make_slice_examples(df: pd.DataFrame):
    # Fixed X3 slice for Output1
    fixed_x3 = 3.0
    sdf = df[np.isclose(df['X3'], fixed_x3)].copy()
    u = sdf['X1'].to_numpy()
    v = sdf['X2'].to_numpy()
    z_true = sdf['Output1_true'].to_numpy()

    rng = np.random.default_rng(SEED)
    noise = rng.normal(0, 0.1, size=z_true.shape)
    z_obs = z_true + noise

    idx = rng.permutation(len(sdf))
    n_train = 11
    tr, te = idx[:n_train], idx[n_train:]

    xy = np.column_stack([u, v])
    cubic = CloughTocher2DInterpolator(xy[tr], z_obs[tr])
    rbf = RBFInterpolator(xy[tr], z_obs[tr], kernel='multiquadric', epsilon=1.0, smoothing=0.0)

    grid_u = np.linspace(u.min(), u.max(), 60)
    grid_v = np.linspace(v.min(), v.max(), 60)
    GU, GV = np.meshgrid(grid_u, grid_v)
    GXY = np.column_stack([GU.ravel(), GV.ravel()])
    z_c = cubic(GXY).reshape(GU.shape)
    z_r = rbf(GXY).reshape(GU.shape)

    grid_true = (GU**2 + GV + np.sin(fixed_x3))
    slice_grid = pd.DataFrame({
        'X1': GU.ravel(),
        'X2': GV.ravel(),
        'X3_fixed': fixed_x3,
        'true': grid_true.ravel(),
        'cubic_pred': z_c.ravel(),
        'rbf_pred': z_r.ravel(),
    })
    slice_grid.to_csv(CSV_DIR / 'slice_examples_grid.csv', index=False)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, zz, ttl in [
        (axs[0], grid_true, 'Ground truth'),
        (axs[1], z_c, 'Cubic (train points shown)'),
        (axs[2], z_r, 'RBF (train points shown)')]:
        im = ax.contourf(GU, GV, zz, levels=18, cmap='viridis')
        ax.scatter(u[tr], v[tr], c='white', s=18, edgecolor='black', linewidth=0.4)
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_title(ttl)
    fig.colorbar(im, ax=axs, shrink=0.9, label='Output1')
    fig.suptitle('Slice examples at fixed X3 = 3.0', fontsize=12)
    save_fig(fig, 'slice_examples')


def make_rmse_boxplots(runs: pd.DataFrame):
    runs.to_csv(CSV_DIR / 'rmse_boxplots.csv', index=False)

    fig, axs = plt.subplots(1, 2, figsize=(14.5, 5.2), constrained_layout=True)
    for ax, regime in zip(axs, ['noise_free', 'noisy']):
        sub = runs[runs['regime'] == regime]
        labels, data = [], []
        for out in ['Output1', 'Output2', 'Output3']:
            for method in ['cubic', 'rbf']:
                vals = sub[(sub['output'] == out) & (sub['method'] == method)]['rmse'].dropna().to_numpy()
                if vals.size == 0:
                    continue
                labels.append(f'{out}\n{method}')
                data.append(vals)
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title(f'RMSE distributions ({regime})')
        ax.set_ylabel('RMSE')
        ax.tick_params(axis='x', labelsize=9, rotation=20)
    save_fig(fig, 'rmse_boxplots')


def make_pred_vs_true(df: pd.DataFrame):
    # Build prediction pairs for noisy regime across outputs/methods
    rng = np.random.default_rng(SEED + 7)
    recs = []

    for output_name, noise_sigma, func_idx in [
        ('Output1', 0.1, 1),
        ('Output2', 1.0, 2),
        ('Output3', 2.0, 3),
    ]:
        for fixed_var, values in [('X1', sorted(df['X1'].unique())), ('X2', sorted(df['X2'].unique())), ('X3', sorted(df['X3'].unique()))]:
            for fv in values:
                sdf = df[np.isclose(df[fixed_var], fv)].copy()
                if len(sdf) < 8:
                    continue
                if fixed_var == 'X1':
                    u, v = sdf['X2'].to_numpy(), sdf['X3'].to_numpy()
                elif fixed_var == 'X2':
                    u, v = sdf['X1'].to_numpy(), sdf['X3'].to_numpy()
                else:
                    u, v = sdf['X1'].to_numpy(), sdf['X2'].to_numpy()

                x1, x2, x3 = sdf['X1'].to_numpy(), sdf['X2'].to_numpy(), sdf['X3'].to_numpy()
                o1, o2, o3 = output_functions(x1, x2, x3)
                z_true = o1 if func_idx == 1 else (o2 if func_idx == 2 else o3)
                z_obs = z_true + rng.normal(0, noise_sigma, size=z_true.shape)

                idx = rng.permutation(len(sdf))
                n_train = min(max(int(np.ceil(0.7 * len(sdf))), 6), len(sdf)-2)
                tr, te = idx[:n_train], idx[n_train:]
                xy = np.column_stack([u, v])

                try:
                    cubic = CloughTocher2DInterpolator(xy[tr], z_obs[tr])
                    pred_c = np.asarray(cubic(xy[te]), dtype=float)
                except Exception:
                    pred_c = np.full(te.shape[0], np.nan)

                try:
                    rbf = RBFInterpolator(xy[tr], z_obs[tr], kernel='multiquadric', epsilon=1.0, smoothing=0.0)
                    pred_r = np.asarray(rbf(xy[te]), dtype=float)
                except Exception:
                    pred_r = np.full(te.shape[0], np.nan)

                for t, p in zip(z_true[te], pred_c):
                    if np.isfinite(p):
                        recs.append({'regime': 'noisy', 'output': output_name, 'method': 'cubic', 'y_true': float(t), 'y_pred': float(p)})
                for t, p in zip(z_true[te], pred_r):
                    if np.isfinite(p):
                        recs.append({'regime': 'noisy', 'output': output_name, 'method': 'rbf', 'y_true': float(t), 'y_pred': float(p)})

    pred_df = pd.DataFrame(recs)
    pred_df.to_csv(CSV_DIR / 'pred_vs_true.csv', index=False)

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for ax, method in zip(axs, ['cubic', 'rbf']):
        sub = pred_df[pred_df['method'] == method]
        ax.scatter(sub['y_true'], sub['y_pred'], s=10, alpha=0.5)
        lo = min(sub['y_true'].min(), sub['y_pred'].min())
        hi = max(sub['y_true'].max(), sub['y_pred'].max())
        ax.plot([lo, hi], [lo, hi], 'r--', lw=1)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(f'Pred vs True ({method}, noisy)')
    save_fig(fig, 'pred_vs_true')


def make_tables(summary: pd.DataFrame):
    # new_results table and csv
    table_df = summary[['regime', 'output', 'method', 'n_runs', 'rmse_mean', 'r2_mean']].copy()
    table_df.columns = ['Regime', 'Output', 'Method', 'Runs', 'RMSE', 'R2']
    table_df['RMSE'] = table_df['RMSE'].map(lambda x: f'{x:.3f}')
    table_df['R2'] = table_df['R2'].map(lambda x: f'{x:.3f}')
    table_df.to_csv(TAB_DIR / 'new_results.csv', index=False)

    latex_new = "\\begin{tabular}{llcrrr}\n\\toprule\nRegime & Output & Method & Runs & RMSE & $R^2$ \\\\n\\midrule\n"
    for _, r in table_df.iterrows():
        latex_new += f"{r['Regime']} & {r['Output']} & {r['Method']} & {int(r['Runs'])} & {r['RMSE']} & {r['R2']} \\\\n"
    latex_new += "\\bottomrule\n\\end{tabular}\n"
    (TAB_DIR / 'new_results.tex').write_text(latex_new, encoding='utf-8')

    exp = pd.DataFrame([
        ('random_seed', 42),
        ('repeats_per_slice', 40),
        ('train_fraction', 0.7),
        ('bootstrap_resamples', 1000),
        ('rbf_kernel', 'multiquadric'),
        ('rbf_smoothing', 0.0),
        ('cubic_interpolator', 'CloughTocher2DInterpolator'),
        ('noise_sigma_output1', 0.1),
        ('noise_sigma_output2', 1.0),
        ('noise_sigma_output3', 2.0),
    ], columns=['Setting', 'Value'])
    exp.to_csv(TAB_DIR / 'experiment_settings.csv', index=False)

    latex_exp = "\\begin{tabular}{ll}\n\\toprule\nSetting & Value \\\\n\\midrule\n"
    for _, r in exp.iterrows():
        latex_exp += f"{r['Setting']} & {r['Value']} \\\\n"
    latex_exp += "\\bottomrule\n\\end{tabular}\n"
    (TAB_DIR / 'experiment_settings.tex').write_text(latex_exp, encoding='utf-8')


def main():
    runs = pd.read_csv(RESULTS / 'interpolation_runs.csv')
    summary = pd.read_csv(RESULTS / 'interpolation_summary.csv')

    df = base_design_points()
    make_workflow()
    make_design_points(df)
    make_slice_examples(df)
    make_rmse_boxplots(runs)
    make_pred_vs_true(df)
    make_tables(summary)

    print('Generated assets in:')
    print(FIG_DIR)
    print(TAB_DIR)
    print(CSV_DIR)


if __name__ == '__main__':
    main()
