# surfacegraph_arxiv (code-only)

This repository contains only code and experiment outputs.

## Files
- `interpolation_experiments.py`: Main benchmark protocol
- `generate_prism_assets.py`: Helper script for figure/table generation
- `results/`: Sample benchmark outputs

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python3 interpolation_experiments.py
```

## Multi-run and failure diagnostics
```bash
python3 multi_run_diagnostics.py
```

## Generate visuals
```bash
python3 generate_repo_visuals.py
```

Generated visuals:
- `visuals/rmse_boxplots.png`
- `visuals/negative_r2_rate_noisy.png`
- `visuals/rmse_by_seed_noisy.png`
- `visuals/slice_3d_surfaces.png` (3D surface comparisons)
