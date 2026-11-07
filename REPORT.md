# Exo-Detect: Deep Learning Classification of Kepler Exoplanet Transit Candidates

## Abstract

This project builds an end-to-end system that classifies NASA Kepler transit
candidates as genuine planets or false positives from their phase-folded light
curves. A 1D convolutional neural network is trained on ~2,800 real Kepler
targets, labeled using the NASA Exoplanet Archive's Kepler Objects of Interest
(KOI) dispositions, and benchmarked against a classical box-fitting baseline.
On a held-out test set of 426 stars, the CNN reaches a ROC-AUC of 0.687 ± 0.005
(mean ± std over 5 seeds), while the depth-based classical baseline performs at
chance (ROC-AUC ≈ 0.42). A dual-view variant that adds a high-resolution
"local" view of the transit does **not** improve performance, which — together
with a train-vs-validation loss plateau — indicates the system is limited by the
signal-to-noise of single-quarter light curves rather than by model capacity.
A follow-up pilot that stitches every available Kepler quarter (~4 years) for
200 stars gives a dual-view CNN ROC-AUC of 0.763 ± 0.038 while training on only
139 stars, but the pilot's 30-star test set is too small (≈ ±0.17 at 95 %
confidence) for that to be established, and a diagnostic shows the pilot's folded
curves are not measurably less noisy because the resampling step interpolates
instead of averaging. The signal-to-noise hypothesis therefore remains untested.
Training is operationalized as a reproducible, infrastructure-as-code pipeline on
AWS SageMaker, and the results are explored through an interactive Streamlit
dashboard.

## 1. Motivation

When a planet passes in front of its host star, it blocks a small fraction of
the star's light, producing a periodic dip in the star's brightness over time —
a *transit*. NASA's Kepler mission monitored ~200,000 stars for years to detect
these dips. The core difficulty is that transits are shallow (often well under
1% in depth) and are easily mimicked by other phenomena, most commonly
*eclipsing binaries* — two stars orbiting and eclipsing each other, which produce
transit-like dips without any planet.

Distinguishing real planets from these false positives is therefore a
classification problem on time-series data, and a natural fit for deep learning:
a convolutional network can learn the *shape* of a genuine transit (a smooth,
flat-bottomed dip with characteristic ingress/egress) rather than relying on a
single hand-engineered statistic such as dip depth. This project reproduces, at
small scale, the central idea of Shallue & Vanderburg (2018), who first applied
CNNs to Kepler transit classification, and pairs it with a classical baseline so
that the model's performance is interpretable rather than a number in isolation.

## 2. Data

**Source.** Light curves are obtained from the Mikulski Archive for Space
Telescopes (MAST) via the `lightkurve` library, using the PDCSAP flux product —
the Kepler pipeline's version of the brightness series with known instrumental
systematics removed. Labels come from the NASA Exoplanet Archive's KOI
cumulative table, specifically the `koi_disposition` field.

**Task framing.** The problem is cast as **binary classification**:
`CONFIRMED` → planet (positive), `FALSE POSITIVE` → not a planet (negative).
`CANDIDATE` objects (neither confirmed nor rejected) are dropped so the labels
are unambiguous.

**Preprocessing.** For each star, one quarter of PDCSAP flux is cleaned
(NaN removal), normalized to a median of ~1.0, and flattened to remove slow
stellar/instrumental trends while preserving the sharp transit. It is then
**phase-folded** on the KOI's catalog orbital period and epoch — overlaying every
orbit so the repeating dip reinforces while random noise averages down — and
resampled onto a fixed-length 201-point "global view" vector suitable for a CNN.

**Filtering and leakage control.** Two deliberate choices:
- Only KOIs with orbital period `< 30 days` are used, so that a single ~90-day
  quarter contains at least ~3 transits to fold — ensuring a usable signal.
- The data is **deduplicated to one KOI per star**. Many confirmed planets live
  in multi-planet systems; without this step the same star could appear in both
  the training and test sets, leaking information and inflating results.

**Dataset.** After filtering and balancing the two classes, ~2,837 stars were
successfully processed (the confirmed-planet pool, ~1,611 stars, is the limiting
class). The set is split **at the star level** into stratified 70/15/15
train/validation/test partitions (~1,985 / 426 / 426), preserving class balance
in each.

**All-quarters pilot dataset.** To test the signal-to-noise hypothesis directly,
a second, much smaller dataset was built with the same labels, filters, views
and split procedure, but with a different light-curve step: instead of one
quarter per star, *all* available long-cadence quarters (~4 years, up to 17
quarters) are downloaded, stitched, flattened and then folded, so each fold
contains many more transits. Downloading is slow (~1–2 minutes per star), so the
pilot uses 100 planets and 100 false positives (200 stars, none lost to
processing), split 139 / 31 / 30 for train / validation / test. As before, the
normalisation scale is computed from the training split only.

## 3. Methods

### 3.1 Classical baseline

A box-fitting (BLS-style) detector operates on the folded global view: it slides
a box-shaped dip of several candidate widths across the curve and computes the
best signal-to-noise ratio of an in-transit depression against the
out-of-transit baseline (using a robust median/MAD noise estimate). The score is
thresholded into a planet/not-planet decision, with the threshold chosen on the
validation set. This captures "how strong is the central dip" — the classical,
depth-driven intuition — and serves as the bar the CNN must clear.

### 3.2 Convolutional neural network

The model is a compact 1D CNN:

- Three convolutional blocks (16 → 32 → 64 channels, kernel size 5), each with
  ReLU activations and max-pooling after the first two.
- Global average pooling to a 64-dimensional summary of the curve.
- Dropout (0.3) and a single linear unit producing a logit, with a sigmoid for
  the planet probability.

Training uses binary cross-entropy (`BCEWithLogitsLoss`), the Adam optimizer
(lr = 1e-3, weight decay = 1e-3), and **early stopping on validation loss**
(patience 20) with restoration of the best checkpoint. Light, physically
motivated **augmentation** is applied during training: time-reversal (a transit
is symmetric), small phase shifts of ±3 bins (the catalog epoch is imprecise),
and low-amplitude noise. The model is deliberately small because the training
set is modest — a larger network would memorize rather than generalize.

### 3.3 Dual-view variant

As a planned extension (following Shallue & Vanderburg's two-view design), a
second input was added: a 61-point **local view** that zooms in on the transit
itself at higher resolution. This feeds a two-branch CNN whose per-view features
are concatenated before the classifier head.

### 3.4 Evaluation protocol

All models are trained with 5 random seeds and reported as mean ± std. Because
the classes are roughly balanced but the operating-point metrics are sensitive to
threshold choice, the **threshold-free ROC-AUC and PR-AUC are treated as the
headline metrics**. Any decision threshold is fixed using the validation set
only; the test set is used once, for final reporting. Model selection between the
global-only and dual-view variants is likewise made on **validation** ROC-AUC,
never on test.

## 4. Results

| Model                                  | ROC-AUC           | PR-AUC            | n_test |
|----------------------------------------|-------------------|-------------------|--------|
| Classical box-fit (BLS-style) baseline | 0.42              | 0.42              | 426    |
| 1D CNN — global view **(final)**       | **0.687 ± 0.005** | **0.634 ± 0.014** | 426    |
| 1D CNN — global + local view           | 0.66 ± 0.03       | 0.60 ± 0.05       | 426    |

*Mean ± std over 5 seeds. Validation ROC-AUC: global-only 0.671, dual-view 0.657.*

The classical baseline sits at chance (ROC-AUC ≈ 0.42; a balanced-class random
classifier scores 0.5). The global-view CNN clears it by a wide and extremely
stable margin. The dual-view model, despite being the more sophisticated
architecture, scored *lower* on validation and with ~6× higher seed variance, so
the global-only model was selected as final.

### 4.1 All-quarters pilot

| Model (all-quarters pilot)   | ROC-AUC       | PR-AUC        | n_test |
|------------------------------|---------------|---------------|--------|
| Classical box-fit baseline   | 0.30          | 0.40          | 30     |
| 1D CNN — global + local view | 0.763 ± 0.038 | 0.784 ± 0.040 | 30     |

*Mean ± std over 5 seeds. Validation ROC-AUC 0.723 ± 0.056 (n = 31). At the
validation-tuned threshold the CNN has precision 0.586 ± 0.017, recall
0.960 ± 0.033 and F1 0.727 ± 0.017; the baseline labels every star a planet
(precision 0.50, recall 1.0, F1 0.667).*

With every quarter folded, the dual-view CNN's point estimate rises from 0.66 to
0.76 ROC-AUC, even though its training set shrinks from ~1,985 stars to 139.
The baseline again lands below chance (0.30), as it did on the larger set (0.42).

**Statistical caveat.** These numbers are far less certain than the main table.
With 15 planets and 15 false positives in the test set, the sampling standard
error of an AUC near 0.76 is roughly 0.09 (≈ ±0.17 at 95 % confidence by the
Hanley–McNeil approximation), so the 0.763 and the earlier 0.66 (n = 426)
cannot be separated statistically. The reported ± 0.038 is the spread across
training seeds on one fixed test set; it does not include this sampling
uncertainty. The baseline's 0.30 carries a similar uncertainty. The pilot is
therefore best read as an encouraging but inconclusive result.

### 4.2 Diagnostic: stitching did not reduce the folded-curve noise

If the pilot's higher score came from a higher signal-to-noise ratio, the
stitched views should be visibly quieter than the single-quarter ones. They are
not. For the 192 pilot stars that also exist in the single-quarter dataset (same
catalog period and epoch), the robust out-of-transit noise of the 201-point
global view (1.4826 × MAD outside the widest baseline box) has a median
all-quarters ÷ single-quarter ratio of **1.03**, and only 40 % of stars improved.

The explanation is in the folding step. Both dataset builders resample the folded
points onto the 201-point grid with `np.interp`, which linearly interpolates
between neighbouring points and therefore reads roughly one noisy sample per
bin; extra quarters add more points to choose from but do not average them. A
one-star check on Kepler-8 (KIC 6922244) is consistent with this: stitching raised
the number of points per phase bin from about 2 to about 325, yet the
interpolated view's noise stayed at ≈1.6 × 10⁻⁴ (single quarter ≈1.4 × 10⁻⁴),
while averaging the points within each bin gave ≈1.3 × 10⁻⁵, roughly 11× lower.
This was a single star, not a systematic test, but it identifies a concrete fix.
Whatever produced the pilot's higher AUC, it was therefore not a cleaner
201-point curve, and the full signal-to-noise benefit of stitching is still
unrealised.

## 5. Analysis

**The CNN learns shape, not just depth.** Both classes were folded on their
catalog ephemerides, so planets *and* false positives present a dip at phase 0.
The baseline, which measures only the depth of that dip, cannot separate them —
indeed it performs slightly below chance, consistent with eclipsing-binary false
positives often having *deeper* dips than genuine planets. The CNN's clear
advantage therefore comes from learning the *morphology* of the dip (its profile,
symmetry, and surroundings), which is exactly the information a depth statistic
discards.

**Why the local view did not help.** The local view's premise is a clean,
high-resolution look at transit shape. But with only a single quarter folded, the
transit is under-sampled and noisy, so the high-resolution view largely magnifies
noise rather than adding signal. Adding a second branch also adds parameters
without adding information — producing the observed outcome of slightly lower
accuracy and markedly higher variance. This is a genuine negative result: the
technique that works in the literature depends on the high signal-to-noise of
all-quarters folds, which this single-quarter dataset lacks.

The all-quarters pilot (Sections 4.1 and 4.2) was meant as the first test of this
explanation, and it does not yet settle it. A dual-view model on stitched light
curves scores higher than the single-quarter dual-view model, but the pilot used
a different and tiny test set (about ±0.17 AUC sampling uncertainty), no
global-view-only model was run on the all-quarters data, and, most importantly,
the stitched views are not measurably less noisy than the single-quarter ones
(Section 4.2). Because the folding step interpolates instead of averaging, the
extra quarters have not actually been converted into a higher signal-to-noise
ratio, so the experiment has not yet tested the hypothesis it was designed for.

**The ceiling is data-limited, not model-limited.** Training and validation loss
plateau together around 0.59 (versus 0.69 for random guessing), and the most
principled architectural change available — the two-view model — failed to move
the ceiling. Both are strong evidence that performance (~ROC-AUC 0.69) is bounded
by the information content of the input representation, not by network capacity.
Further architecture or hyperparameter tuning would not break past this; only
richer input (notably stitching all available quarters for higher SNR) is
expected to, and the pilot points in that direction.

**Why AUC, not F1, is the honest metric.** At this separability, the
F1-maximizing decision threshold collapses toward predicting "planet" for almost
everything (recall ≈ 0.98), which inflates F1 for both the CNN and the
all-positive baseline and makes them look deceptively close. The threshold-free
AUCs tell the true story (0.69 vs 0.42). F1 is, in effect, ceiling-bound by
separability: it cannot rise substantially without a higher AUC, so AUC is the
metric to optimize and report.

## 6. Engineering and Reproducibility

The project is built for reproducibility end to end. The Python environment and
all dependencies are managed with `uv` and pinned via a lockfile; code style is
enforced with `ruff`; the NASA label table is cached locally as a fixed snapshot
so results are not affected by later catalog updates.

Model training is operationalized as an **infrastructure-as-code pipeline on AWS
SageMaker**. Terraform provisions the durable infrastructure — an S3 bucket for
the dataset and artifacts, and a least-privilege IAM role scoped to that bucket,
the SageMaker log groups, and read-only container image pulls. A launch script
(SageMaker Python SDK) uploads the dataset to S3 and starts a managed-PyTorch
**training job in script mode**: SageMaker spins up ephemeral compute, runs the
training container, writes the trained model (`model.tar.gz`) and metrics back to
S3, logs the run to CloudWatch, and tears the compute down automatically.
SageMaker was chosen over EC2/Fargate as the ML-native option: training jobs are
ephemeral by design (lowest cost, nothing left running), and script mode reuses
the existing training code in a managed, dependency-pinned container with no
custom image to maintain. This is not about needing cloud scale for a small
model — it demonstrates a reproducible, versioned, auditable training workflow
that scales directly to larger models and datasets without re-architecting.

The results are also exposed through an **interactive Streamlit + Plotly
dashboard** (`dashboard/`). It loads the trained `model.pth` once and runs a live
forward pass for any selected test star, alongside the classical baseline's
box-fit score, with an animated phase-folding demo, a CNN-versus-baseline
comparison page, a layer-by-layer architecture view with feature-map heatmaps,
and an experimental page that fetches any Kepler star from MAST. The dashboard
serves the single-quarter global-view model (the headline model); the
all-quarters pilot is not deployed. The code is organised into `dataset/`,
`baseline/` and `CNN/` folders with one script per experiment, so each
reported number can be regenerated independently.

## 7. Limitations

- **Single quarter per star in the headline model.** Only one Kepler quarter is
  used per target, which caps the signal-to-noise of the folded light curve and
  is the primary driver of the performance ceiling.
- **All-quarters evidence is pilot-scale.** The stitched-quarter experiment uses
  200 stars and a 30-star test set, so its AUC has about ±0.17 sampling
  uncertainty (not reflected in the seed std), and it has no global-view-only
  ablation, so the effect of the local view is not isolated.
- **Short-period bias.** The `period < 30 d` filter biases the sample toward
  short-period systems and does not represent the full planet population.
- **Label reliability.** KOI dispositions are NASA's current best judgments
  (~95% reliable) and occasionally change; they are treated as strong but not
  absolute ground truth.
- **Representation-bounded performance.** Because the system is data/SNR-limited,
  model-side changes alone cannot materially raise accuracy.

## 8. Future Work

- **Bin-average the fold, then re-test all-quarters stitching.** Replace the
  `np.interp` resampling with a per-bin mean (or median) of the folded points, and
  confirm the noise ratio in Section 4.2 drops well below 1. Then build the
  all-quarters dataset for the full ~3,000-star pool (or at least enough stars for
  a test set of several hundred, which narrows the AUC uncertainty to a few
  points), run the global-view-only and dual-view models on the *same* split to
  isolate the local view's contribution, and deploy the winner in the dashboard.
  Bin-averaging could also help the single-quarter pipeline.
- **Better uncertainty reporting.** Add bootstrap confidence intervals over test
  stars alongside the seed std, so small-test-set results are not over-read.
- **Richer labels and architectures.** Incorporating `CANDIDATE` objects under a
  more nuanced scheme, and exploring sequence models (e.g. transformers) once the
  input SNR justifies added capacity.

## 9. Conclusion

A compact 1D CNN reliably distinguishes Kepler planets from false positives on
real, leakage-controlled data (ROC-AUC 0.687 ± 0.005), clearly outperforming a
classical depth-based baseline that performs at chance. A careful, honest
evaluation — validation-based model selection, multi-seed reporting,
threshold-free metrics — shows that the learned advantage is real and that the
remaining ceiling is set by data signal-to-noise rather than model capacity, a
conclusion reinforced by the dual-view negative result. A pilot that stitches all
Kepler quarters (ROC-AUC 0.763 ± 0.038 on a 30-star test set) is encouraging but
inconclusive: the test set is tiny and the folding step does not yet average the
extra data, so the hypothesis still needs a bin-averaged, larger-scale re-run.
The work is wrapped in a reproducible, infrastructure-as-code training
pipeline on AWS and an interactive dashboard, bridging the machine-learning,
cloud-engineering and product sides of the problem.

## 10. References

1. Shallue, C. J., & Vanderburg, A. (2018). *Identifying Exoplanets with Deep
   Learning.* The Astronomical Journal, 155(2), 94.
2. Kovács, G., Zucker, S., & Mazeh, T. (2002). *A box-fitting algorithm in the
   search for periodic transits.* Astronomy & Astrophysics, 391, 369.
3. Thompson, S. E., et al. (2018). *Planetary Candidates Observed by Kepler.
   VIII (DR25).* The Astrophysical Journal Supplement Series, 235, 38.
4. NASA Exoplanet Archive — Kepler Objects of Interest (KOI) cumulative table.
5. Lightkurve Collaboration (2018). *Lightkurve: Kepler and TESS time series
   analysis in Python.*