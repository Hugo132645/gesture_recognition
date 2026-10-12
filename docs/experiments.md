# Smoothing experiments

The repository includes three recorded trial pairs: `off_trial1`, `ema_trial1`, and `oneeuro_trial1`. Each has a raw-angle CSV and a quantized servo-output CSV in `data/experiments/runs/`. Their names describe the historical recording conditions; the current controller does not offer a smoothing-mode flag.

Install the additional analysis dependencies from the repository root:

```bash
python -m pip install -r analysis/requirements.txt
```

Both analysis scripts search recursively from the working directory for matching CSV pairs. Run them from the experiment folder to avoid mixing other logs:

```bash
cd data/experiments
python ../../analysis/analyse_smoothing.py
# Alternative analysis implementation:
python ../../analysis/analyze_smoothing.py
```

The scripts display Matplotlib figures; they do not automatically regenerate the saved PNG collection. Existing plots are retained in `results/smoothing/`, alongside the originally uploaded ZIP. The two analysis implementations are distinct and have been retained with their original names.

These recordings and plots are reference material. Comparisons between separate trials can reflect different hand motions as well as filtering, so they should not be presented as controlled evidence that one version is best.
