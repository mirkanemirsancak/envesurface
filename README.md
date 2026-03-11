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
