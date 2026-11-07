# Exo-Detect — Exoplanet Transit Detection

Deep learning classification of NASA Kepler transit candidates: given a star's
phase-folded light curve, predict whether it hosts a real planet or is a false
positive. A 1D CNN is benchmarked against a classical detection baseline on
real telescope data, with leakage-controlled, multi-seed evaluation and a
reproducible, infrastructure-as-code training pipeline on AWS. An interactive
**Streamlit dashboard** lets you explore individual stars, watch the model and
baseline make their calls live, and look inside the network.

<!-- TODO: replace with the Streamlit Community Cloud URL once deployed:
[![Live demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://<your-app>.streamlit.app) -->
**Live demo:** *coming soon* · **Run locally:** `uv run streamlit run dashboard/app.py`

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

### Pilot: stitching all Kepler quarters

To test whether the single-quarter ceiling is a signal-to-noise problem, the
pipeline was re-run on a **small pilot set of 200 stars (100 planets / 100 false
positives) using every available long-cadence quarter (~4 years, up to 17
quarters) stitched per star** before phase-folding (train / val / test =
139 / 31 / 30).

| Model (all-quarters pilot)               | ROC-AUC       | PR-AUC        | n_test |
|------------------------------------------|---------------|---------------|--------|
| Classical box-fit baseline               | 0.30          | 0.40          | 30     |
| 1D CNN — global + local view             | 0.763 ± 0.038 | 0.784 ± 0.040 | 30     |

*Mean ± std over 5 seeds; validation ROC-AUC 0.723 ± 0.056. Precision 0.59,
recall 0.96, F1 0.73 at the validation-tuned threshold.*

**Read this as a promising signal, not a new headline number.** The point
estimate is higher than the single-quarter dual-view model (0.66), even though
it was trained on ~14× fewer stars. But the test set is only **30 stars**: the
seed std (±0.04) captures training randomness only, while the sampling
uncertainty of an AUC on 15 + 15 stars is about **±0.17 (95 %)**. The pilot's
test set also differs from the 426-star one, and no global-view-only all-quarters
run exists yet, so the gain cannot be attributed to the local view versus the
extra quarters. The headline result and the dashboard's `model.pth` remain the
single-quarter global-view model, which was evaluated on 426 stars.

**A diagnostic worth knowing about.** Stitching all quarters gives each phase bin
hundreds of data points instead of a handful, so the folded curve *should* get
much quieter. It doesn't: across the 192 pilot stars that also exist in the
single-quarter dataset, the median out-of-transit noise of the 201-point view is
essentially unchanged (ratio 1.03; only 40 % of stars improved). The cause is
the folding step, which resamples with `np.interp` and so reads roughly one noisy
point per bin instead of averaging them. A one-star check on Kepler-8 confirmed
it: points per bin rose from ~2 to ~325 with no drop in the interpolated view's
noise (≈1.6e-4 vs 1.4e-4 single-quarter), whereas a per-bin average of the same
points gave ≈1.3e-5, about 11× lower. So the pilot's score has **not yet
benefited from the extra signal-to-noise**; bin-averaging in the folding step is
the first thing to try next. (The dashboard's *All-quarters pilot* page shows this
comparison live.)

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
- **All-quarters option:** `dataset/build_dataset_all_quarters.py` downloads every
  quarter per star, stitches them, then flattens and folds — more transits per
  fold, but ~1–2 min per star, so the pilot is limited to 200 stars. (Its
  resampling step interpolates rather than bin-averages; see the diagnostic above.)
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
- **All-quarters pilot: encouraging but inconclusive.** With all quarters
  stitched, a dual-view CNN trained on just 139 stars reaches ROC-AUC
  0.763 ± 0.038 on a 30-star test set (vs 0.66 single-quarter), but that test
  set is too small to conclude anything (≈ ±0.17 at 95 %). Moreover the pilot's
  folded curves are *not* measurably less noisy than single-quarter ones, because
  the resampling step interpolates instead of averaging. The SNR hypothesis is
  therefore still untested: the next step is bin-averaging plus a larger test set.

## Interactive dashboard

A Streamlit + Plotly app (`dashboard/`) presents the project as a product
rather than a notebook. Every chart is interactive (hover, zoom, pan), and the
CNN runs **live**: the app loads `model.pth` once and does a real forward pass
for each star you pick.

| Page | What it provides |
|------|------------------|
| 🌌 **Overview** | Plain-language intro to transits, the planet-vs-false-positive task and phase folding, headline stats (dataset size, class balance, CNN vs baseline ROC-AUC), a schematic transit plot, and a pipeline diagram (Kepler data → labels → preprocessing → split → CNN / baseline → evaluation). |
| 🔭 **Explore a light curve** | Pick any of the 426 held-out test stars (filter by class, or 🎲 pick a random one). Shows the global view (one full orbit) with the baseline's best box shaded, the zoomed local view, the true NASA disposition, the CNN's probability on a confidence gauge with a clear planet / false-positive verdict, and the baseline's box-fit SNR. Three tabs add motion and context: an **animated phase-folding** demo, an **animated box sweep** of the baseline over box widths 5–41 (with SNR per width), and histograms showing where this star's scores sit in the whole test set. Links to the star's NASA Exoplanet Archive page. |
| 📊 **Model vs baseline** | Comparison table (ROC-AUC, PR-AUC, precision, recall, F1) for the 5-seed CNN, the deployed `model.pth`, the baseline, and random guessing. Includes bar charts with ±std error bars and chance lines, plus ROC and precision-recall curves recomputed live from `model.pth` on the test set. |
| 🛰️ **All-quarters pilot** | The 200-star experiment that stitches every Kepler quarter. KPI tiles and a results table for the pilot CNN and baseline, plus a **forest plot** of every model's ROC-AUC with its 95 % sampling interval so the 30-star test set's uncertainty is visible. Pick any pilot test star to see its all-quarters global and local views overlaid on its single-quarter curve, with the baseline SNR computed live. A final panel measures the out-of-transit noise of the all-quarters vs single-quarter curves for the ~190 stars in both datasets. |
| 🧠 **CNN architecture** | A layer-by-layer diagram of `TransitCNN` from the 1×201 input to P(planet), with output shapes measured from the loaded network and parameter counts (13,057 total). Also a chart of length shrinking as channels grow, plain-language cards on convolutions, pooling and the classifier, and **live feature-map heatmaps** showing what each conv block responds to for the selected star. |
| 🧪 **Try any star** *(experimental)* | Enter any KIC id. Period and epoch auto-fill from the KOI table, or you can type them. The app downloads the light curve live from MAST, runs the same preprocessing as training (remove NaNs → normalise → flatten → phase-fold → resample to 201) and classifies it. It warns you if the star is already in the training data. If MAST is slow or unavailable you get a friendly error. |

**How the dashboard makes its calls.** Inputs use the exact training
normalisation, `(x − 1) / std(X_train − 1)`. The planet/false-positive verdict
uses the decision threshold that maximises F1 on the **validation** set (≈ 0.415
for `model.pth`), the same rule training used. It isn't 0.5 because this model's
probabilities are compressed (test-set max ≈ 0.60), so 0.5 would call almost
everything a false positive. The baseline uses the box-fit SNR from
`baseline/baseline_bls.py`. The dashboard and "Try any star" use the
single-quarter preprocessing that `model.pth` was trained on; an all-quarters
model is not deployed yet.

**Deployed model vs headline numbers.** `model.pth` is the single SageMaker run
(test ROC-AUC 0.629, PR-AUC 0.566). The Results table above reports the 5-seed
mean of the same global-view architecture trained locally (0.687 ± 0.005). The
dashboard shows both side by side so the numbers are never mixed up.

```bash
uv sync
uv run streamlit run dashboard/app.py     # needs dataset.npz + model.pth in the repo root
```

## Cloud training pipeline (AWS)

Training is operationalized as a reproducible, **infrastructure-as-code**
pipeline on **AWS SageMaker** — not because a 2-minute CPU model needs cloud
scale, but to demonstrate production-style, reproducible training infrastructure
instead of ad-hoc local runs.

**What it does.** Terraform provisions the durable infrastructure — an S3 bucket
for the dataset and model/metric artifacts, and a least-privilege IAM role that
SageMaker assumes. A launch script (SageMaker Python SDK) uploads the dataset to
S3 and starts a managed-PyTorch **training job in script mode**: SageMaker spins
up ephemeral compute, runs the training container, writes the trained model
(`model.tar.gz`) and metrics back to S3, and tears the compute down
automatically. CloudWatch captures the run logs.

**Why SageMaker (over EC2 / Fargate).** SageMaker is the ML-native service:
training jobs are ephemeral by design — compute exists only for the duration of
the run, making it the lowest-cost and lowest-footgun option (nothing is left
running to bill). Script mode runs the existing training code inside a managed,
dependency-pinned container with no custom Docker image to build or maintain.
EC2 would mean hand-managing instance lifecycle; Fargate would work but adds
nothing ML-specific.

**Design choices.**
- *Infrastructure as code* — the entire environment is defined in Terraform,
  reproducible with `terraform apply` and removed cleanly with `terraform destroy`.
- *Least-privilege IAM* — the execution role is scoped to just this bucket,
  SageMaker log groups, and read-only image pulls, not broad admin access.
- *Separation of concerns* — Terraform owns durable infrastructure; the training
  job, an ephemeral action, is launched via the SDK. This mirrors how ML
  training is orchestrated in practice.

**What it demonstrates.** The pipeline turns "it trained on my laptop" into a
reproducible, versioned, auditable workflow: data and artifacts live in S3, the
environment is pinned and recreatable, the run is logged to CloudWatch, and the
whole stack is one command to stand up and one to tear down. It bridges ML and
cloud infrastructure, and the same pattern scales directly to larger models and
datasets without re-architecting.

**Run it:**
```bash
cd infra && terraform apply        # provision S3 bucket + SageMaker IAM role
cd ..
uv run python launch_sagemaker.py --bucket <bucket_name> --role_arn <role_arn>
cd infra && terraform destroy      # tear everything down when finished
```

## Tech stack

Python 3.12 · [uv](https://docs.astral.sh/uv/) (env + dependencies) ·
[lightkurve](https://lightkurve.github.io/lightkurve/) (Kepler data) ·
PyTorch (CNN) · scikit-learn (metrics, splits) · ruff (lint + format) ·
**AWS SageMaker** (managed training) · **Terraform** (infrastructure as code) ·
SageMaker Python SDK / boto3 (job orchestration) · **Streamlit** + **Plotly**
(interactive dashboard).

## Repository layout

```
dataset/
  build_dataset.py                # single-quarter Kepler light curves + KOI labels -> dataset.npz
  build_dataset_all_quarters.py   # all quarters stitched (200-star pilot) -> dataset_all_quarters.npz
baseline/
  baseline_bls.py                 # classical box-fit baseline
  baseline_bls_all_quarters.py    # same baseline on the all-quarters pilot set
CNN/
  train_cnn_without_local_view.py # final model: global-view CNN, 5 seeds
  train_cnn.py                    # dual-view (global + local) CNN, 5 seeds
  train_cnn_all_quarters.py                    # dual-view CNN on the all-quarters pilot
  train_cnn_all_quarters_without_local_view.py # global-view CNN on the all-quarters pilot
launch_sagemaker.py        # SDK launcher: upload data to S3 + run the SageMaker job
sagemaker/
  train.py                 # SageMaker script-mode training entrypoint
  requirements.txt         # extra deps installed into the training container
infra/
  main.tf                  # Terraform: S3 bucket + SageMaker execution role
  variables.tf
  outputs.tf
dashboard/
  app.py                   # Streamlit app: pages, charts, animations
  model.py                 # TransitCNN, normalisation, box-SNR baseline, light-curve folding
.streamlit/config.toml     # dashboard theme
koi_table.csv              # cached NASA KOI label snapshot (reproducibility)
model.pth                  # trained global-view CNN weights (from the SageMaker run)
cnn_results.json           # metrics of the SageMaker run that produced model.pth
results/
  cnn_results_without_local_view.json  # final global-view CNN, 5 seeds (headline numbers)
  cnn_results.json                     # dual-view (global + local) experiment, 5 seeds
  baseline_results.json                # classical baseline metrics
  cnn_results_all_quarters.json        # all-quarters pilot: dual-view CNN + baseline, 5 seeds
  baseline_results_all_quarters.json   # all-quarters pilot baseline metrics
CLAUDE.md                  # project conventions + log of major decisions
pyproject.toml             # dependencies, pinned via uv.lock
```

*Generated data (`dataset*.npz`, download cache) and Terraform state are
git-ignored.*

## Reproduce (local)

The scripts in `dataset/`, `baseline/` and `CNN/` use `../` paths, so run each
one from inside its own folder.

```bash
uv sync
# Single-quarter pipeline (headline results)
(cd dataset  && uv run python build_dataset.py)                # -> ../dataset.npz (downloads from MAST — slow on first run)
(cd baseline && uv run python baseline_bls.py)                 # -> ../results/baseline_results.json
(cd CNN      && uv run python train_cnn_without_local_view.py) # -> ../results/cnn_results_without_local_view.json

# All-quarters pilot (200 stars, ~1–2 min per star to download)
(cd dataset  && uv run python build_dataset_all_quarters.py)   # -> ../dataset_all_quarters.npz
(cd baseline && uv run python baseline_bls_all_quarters.py)    # -> ../results/baseline_results_all_quarters.json
(cd CNN      && uv run python train_cnn_all_quarters.py)       # -> ../results/cnn_results_all_quarters.json

uv run streamlit run dashboard/app.py   # open the interactive dashboard
```
For the cloud training pipeline, see **Cloud training pipeline (AWS)** above.

## Roadmap

- [x] Data pipeline — Kepler light curves + KOI labels → phase-folded dataset
- [x] Classical box-fit (BLS-style) baseline
- [x] 1D CNN classifier with leakage-controlled, multi-seed evaluation
- [x] **AWS + Terraform training pipeline** — reproducible, IaC-provisioned
      SageMaker training with artifacts in S3
- [x] **Interactive dashboard**: explore light curves, live CNN predictions,
      CNN vs baseline, architecture, and live MAST lookups (Streamlit + Plotly)
- [ ] Public deployment of the dashboard (Streamlit Community Cloud)
- [x] All-quarters stitching: **200-star pilot** (ROC-AUC 0.763 ± 0.038, but only
      a 30-star test set)
- [ ] Scale all-quarters to the full ~3,000-star set (or a larger test set), run
      the global-view-only ablation on it, then retrain and deploy that model

## Limitations

- **Headline model uses a single quarter per star** and has a **short-period
  bias** (`period < 30 d`). Stitching all quarters looks like the main lever for
  higher accuracy (see the pilot above), but it has only been tried at pilot scale.
- **The all-quarters pilot is statistically thin.** Its test set has 30 stars, so
  an AUC carries roughly ±0.17 (95 %) sampling uncertainty that the seed std does
  not capture, and the pilot has no global-view-only counterpart to isolate the
  effect of the local view.
- Labels are NASA's current KOI dispositions (~95% reliable), not absolute truth.
- With single-quarter inputs, performance is data/SNR-limited rather than
  model-limited, so architectural changes alone won't lift it much past the
  current ceiling.

## Data & acknowledgments

This project uses public data from the NASA Exoplanet Archive (KOI cumulative
table) and the Mikulski Archive for Space Telescopes (MAST), accessed via
`lightkurve`.

## License

MIT