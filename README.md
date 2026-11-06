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
| 🧠 **CNN architecture** | A layer-by-layer diagram of `TransitCNN` from the 1×201 input to P(planet), with output shapes measured from the loaded network and parameter counts (13,057 total). Also a chart of length shrinking as channels grow, plain-language cards on convolutions, pooling and the classifier, and **live feature-map heatmaps** showing what each conv block responds to for the selected star. |
| 🧪 **Try any star** *(experimental)* | Enter any KIC id. Period and epoch auto-fill from the KOI table, or you can type them. The app downloads the light curve live from MAST, runs the same preprocessing as training (remove NaNs → normalise → flatten → phase-fold → resample to 201) and classifies it. It warns you if the star is already in the training data. If MAST is slow or unavailable you get a friendly error. |

**How the dashboard makes its calls.** Inputs use the exact training
normalisation, `(x − 1) / std(X_train − 1)`. The planet/false-positive verdict
uses the decision threshold that maximises F1 on the **validation** set (≈ 0.415
for `model.pth`), the same rule training used. It isn't 0.5 because this model's
probabilities are compressed (test-set max ≈ 0.60), so 0.5 would call almost
everything a false positive. The baseline uses the box-fit SNR from
`baseline_bls.py`.

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
build_dataset.py           # fetch Kepler light curves + KOI labels -> phase-folded dataset
baseline_bls.py            # classical box-fit baseline
train_cnn.py               # 1D CNN: 5-seed training + evaluation (local)
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
CLAUDE.md                  # project conventions + log of major decisions
pyproject.toml             # dependencies, pinned via uv.lock
```

*Generated data (`dataset.npz`, download cache) and Terraform state are
git-ignored.*

## Reproduce (local)

```bash
uv sync
uv run python build_dataset.py   # builds dataset.npz (downloads from MAST — slow on first run)
uv run python baseline_bls.py    # -> baseline_results.json
uv run python train_cnn.py       # -> cnn_results.json
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