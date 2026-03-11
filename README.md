# surfacegraph_arxiv (code-only)

Bu repo yalnızca kod ve deney sonuçlarını içerir.

## Dosyalar
- `interpolation_experiments.py`: Ana deney protokolü
- `generate_prism_assets.py`: Şekil/tablolar için yardımcı script
- `results/`: Örnek benchmark çıktıları

## Kurulum
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Çalıştırma
```bash
python3 interpolation_experiments.py
```

## Çoklu run ve sorun analizi
```bash
python3 multi_run_diagnostics.py
```

## Görselleri üretme
```bash
python3 generate_repo_visuals.py
```

Üretilen görseller:
- `visuals/rmse_boxplots.png`
- `visuals/negative_r2_rate_noisy.png`
- `visuals/rmse_by_seed_noisy.png`
- `visuals/slice_3d_surfaces.png` (3D yüzey karşılaştırmaları)
