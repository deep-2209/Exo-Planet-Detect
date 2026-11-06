# Exo-Detect — Exoplanet Transit Detection

Deep learning classification of NASA Kepler transit candidates: given a star's
phase-folded light curve, predict whether it hosts a real planet or is a false
positive. A 1D CNN is benchmarked against a classical detection baseline on
real telescope data with leakage-controlled, multi-seed evaluation.

## Results

| Model                                  | ROC-AUC           | PR-AUC            | n_test |
|----------------------------------------|-------------------|-------------------|--------|
| Classical box-fit (BLS-style) baseline | 0.42              | 0.42              | 426    |
| 1D CNN — global view **(final)**       | **0.687 ± 0.005** | **0.634 ± 0.014** | 426    |
| 1D CNN — global + local view           | 0.66 ± 0.03       | 0.60 ± 0.05       | 426    |

*Mean ± std over 5 random seeds. The final model was selected on **validation**
ROC-AUC (global 0.671 > dual-view 0.657), not on the test set.*

The CNN clearly separates planets from false positives, while the depth-only
classical baseline performs at chance — showing the model learns transit
*shape*, not just dip depth.

## Problem

Kepler's pipeline flagged threshold-crossing events, and human/automated vetting
labeled each as a **confirmed planet**, **candidate**, or **false positive**
(mostly eclipsing binaries that mimic a transit). This project frames the task
as **binary classification** — confirmed planet vs. false positive — using NASA's
Kepler Objects of Interest (KOI) `koi_disposition` as ground-truth labels.

## Approach

- **Data:** Kepler PDCSAP light curves via `lightkurve`, phase-folded on each
  KOI's catalog period and epoch into a fixed-length 201-point "global view."
- **Labels:** KOI cumulative table (CONFIRMED → 1, FALSE POSITIVE → 0;
  CANDIDATE dropped for clean labels). Filtered to `period < 30 d` (≥3 transits
  per quarter) and deduplicated to **one KOI per star** to prevent leakage.
- **Baseline:** a classical box-fit (BLS-style) signal-to-noise detector on the
  folded curve, thresholded into a planet/not-planet call.
- **Model:** a 1D CNN (3 conv blocks → global average pool → sigmoid), trained
  with early stopping, dropout, weight decay, and light augmentation
  (time-flip, phase-shift, noise).
- **Evaluation:** star-level stratified 70/15/15 split, 5 seeds, metrics
  reported as mean ± std. Threshold-free AUCs are the headline metrics.

## Key findings

- The CNN beats a chance-level classical baseline by a wide, stable margin on
  ROC-AUC and PR-AUC — real signal from learned transit shape.
- **A dual-view variant (adding a zoomed "local" transit view) did not help.**
  It scored *lower* on validation (0.657 vs 0.671) and with ~6× higher variance.
  Likely cause: single-quarter folds lack the signal-to-noise for a
  high-resolution local view to add information rather than noise.
- Performance plateaus around ROC-AUC 0.69 with train ≈ validation loss, and a
  principled architectural change didn't move it — indicating the ceiling is
  **data/SNR-limited, not model-capacity-limited.**

## Tech stack

Python 3.12 · [uv](https://docs.astral.sh/uv/) (env + dependencies) ·
[lightkurve](https://lightkurve.github.io/lightkurve/) (Kepler data) ·
PyTorch (CNN) · scikit-learn (metrics, splits) · ruff (lint + format).

## Repository layout

```
build_dataset.py       # fetch Kepler light curves + KOI labels -> phase-folded dataset
baseline_bls.py        # classical box-fit baseline
train_cnn.py           # 1D CNN: 5-seed training + evaluation
koi_table.csv          # cached NASA KOI label snapshot (reproducibility)
cnn_results.json       # final model metrics
baseline_results.json  # baseline metrics
CLAUDE.md              # project conventions + log of major decisions
pyproject.toml         # dependencies, pinned via uv.lock
```

*Generated data (`dataset.npz`, download cache) is git-ignored and rebuilt by
`build_dataset.py`.*

## Reproduce

```bash
uv sync
uv run python build_dataset.py   # builds dataset.npz (downloads from MAST — slow on first run)
uv run python baseline_bls.py    # -> baseline_results.json
uv run python train_cnn.py       # -> cnn_results.json
```

## Roadmap

- [x] Data pipeline — Kepler light curves + KOI labels → phase-folded dataset
- [x] Classical box-fit (BLS-style) baseline
- [x] 1D CNN classifier with leakage-controlled, multi-seed evaluation
- [ ] **AWS + Terraform training pipeline** — cloud-native, reproducible training
      provisioned as infrastructure-as-code
- [ ] **Interactive dashboard** — explore light curves and model predictions
- [ ] All-quarters stitching per star for higher signal-to-noise

## Limitations

- **Single quarter per star** and a **short-period bias** (`period < 30 d`):
  the main lever for higher accuracy is stitching all available quarters
  (more transits folded → higher SNR) — tracked in the roadmap above.
- Labels are NASA's current KOI dispositions (~95% reliable), not absolute truth.
- Performance is data/SNR-limited rather than model-limited, so architectural
  changes alone won't lift it much past the current ceiling.

## Data & acknowledgments

This project uses public data from the NASA Exoplanet Archive (KOI cumulative
table) and the Mikulski Archive for Space Telescopes (MAST), accessed via
`lightkurve`.

## License

MIT