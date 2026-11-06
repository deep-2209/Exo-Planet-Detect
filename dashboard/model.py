"""Exo-Detect model code used by the dashboard.

This module holds everything that *computes* something, so ``app.py`` can stay
focused on layout and charts:

1. ``TransitCNN``      - the 1D CNN, layer-for-layer identical to the one trained in
                         ``train_cnn_without_local_view.py`` / ``sagemaker/train.py``.
                         It must match exactly, or ``model.pth`` will not load.
2. Normalisation      - ``normalize()`` applies the same (x - 1) / scale transform
                         used during training.
3. ``box_snr()``       - the classical baseline from ``baseline_bls.py``: a box-shaped
                         dip detector scored by signal-to-noise.
4. ``make_global_view()`` - the same light-curve preprocessing as ``build_dataset.py``,
                         used by the experimental "Try any star" page.

Nothing here imports Streamlit, so it can also be used from a notebook or a test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Length of the phase-folded "global view" (one full orbit resampled to 201 points).
L_GLOBAL = 201
# The transit is centred in the view, so the dip sits at index 100.
CENTER = L_GLOBAL // 2
# Box widths (in phase bins) the baseline tries. Odd, so each box is symmetric.
BOX_WIDTHS = [5, 9, 15, 25, 41]


# ---------------------------------------------------------------------------
# 1. The CNN
# ---------------------------------------------------------------------------
class TransitCNN(nn.Module):
    """1D convolutional classifier: phase-folded light curve -> planet logit.

    Input shape  : (batch, 1, 201)  - one channel (normalised flux), 201 phase bins.
    Output shape : (batch,)         - one raw logit per curve; sigmoid -> P(planet).

    The attribute names ``features`` and ``head`` matter: they are the prefixes of
    the keys stored in ``model.pth`` (e.g. ``features.0.weight``).
    """

    def __init__(self) -> None:
        super().__init__()
        # Feature extractor: three conv "pattern detector" blocks.
        self.features = nn.Sequential(
            # Block 1: 16 filters, each 5 bins wide, look for tiny local shapes.
            nn.Conv1d(1, 16, 5, padding=2),   # (1, 201)  -> (16, 201)
            nn.ReLU(),
            nn.MaxPool1d(2),                  # (16, 201) -> (16, 100)
            # Block 2: combine block-1 shapes into larger patterns.
            nn.Conv1d(16, 32, 5, padding=2),  # (16, 100) -> (32, 100)
            nn.ReLU(),
            nn.MaxPool1d(2),                  # (32, 100) -> (32, 50)
            # Block 3: the richest features (e.g. "U-shaped dip").
            nn.Conv1d(32, 64, 5, padding=2),  # (32, 50)  -> (64, 50)
            nn.ReLU(),
            # Average each feature over the whole orbit -> one number per feature.
            nn.AdaptiveAvgPool1d(1),          # (64, 50)  -> (64, 1)
        )
        # Classifier head: 64 features -> 1 logit.
        self.head = nn.Sequential(
            nn.Flatten(),                     # (64, 1) -> (64,)
            nn.Dropout(0.3),                  # only active in training mode
            nn.Linear(64, 1),                 # (64,) -> (1,)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)


def load_model(weights_path: str | Path) -> TransitCNN:
    """Build a TransitCNN, load trained weights, and switch to eval mode."""
    model = TransitCNN()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()  # disables dropout so predictions are deterministic
    return model


def layer_shapes(model: TransitCNN) -> list[tuple[str, tuple[int, ...]]]:
    """Run a dummy (1, 1, 201) input through the model and record every layer's
    output shape. Used by the architecture page so the diagram is *measured*
    from the real network instead of hand-typed."""
    shapes: list[tuple[str, tuple[int, ...]]] = [("Input", (1, L_GLOBAL))]
    x = torch.zeros(1, 1, L_GLOBAL)
    with torch.no_grad():
        for layer in list(model.features) + list(model.head):
            x = layer(x)
            shapes.append((layer.__class__.__name__, tuple(x.shape[1:])))
    return shapes


# ---------------------------------------------------------------------------
# 2. Normalisation + inference
# ---------------------------------------------------------------------------
def compute_scale(X_train: np.ndarray) -> float:
    """The training-set scale: std of (flux - 1). Must be computed from X_train
    only, exactly as in training, or the model sees differently-scaled inputs."""
    return float((X_train - 1.0).std())


def normalize(X: np.ndarray, scale: float) -> torch.Tensor:
    """(N, 201) or (201,) flux array -> (N, 1, 201) float tensor ready for the CNN."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    return torch.tensor((X - 1.0) / scale, dtype=torch.float32).unsqueeze(1)


def predict_proba(model: TransitCNN, X: np.ndarray, scale: float) -> np.ndarray:
    """Return P(planet) for each row of X (shape (N,))."""
    with torch.no_grad():
        return torch.sigmoid(model(normalize(X, scale))).numpy()


def conv_activations(model: TransitCNN, x: np.ndarray, scale: float) -> list[np.ndarray]:
    """Return the post-ReLU feature maps of the three conv blocks for one curve.
    Each item has shape (channels, length) - handy for a heatmap."""
    out = []
    h = normalize(x, scale)
    with torch.no_grad():
        for layer in model.features:
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                out.append(h[0].numpy())
    return out


# ---------------------------------------------------------------------------
# 3. Classical baseline: box-fit SNR (same maths as baseline_bls.py)
# ---------------------------------------------------------------------------
def box_snr_per_width(row: np.ndarray) -> dict[int, dict[str, float]]:
    """For each box width, measure how deep the centred dip is relative to noise.

    depth = median(outside) - mean(inside)       (positive = flux drops)
    sigma = 1.4826 * MAD(outside)                 (robust per-point noise)
    snr   = depth / (sigma / sqrt(width))         (averaging w points cuts noise)
    """
    out: dict[int, dict[str, float]] = {}
    for w in BOX_WIDTHS:
        lo, hi = CENTER - w // 2, CENTER + w // 2 + 1
        inside = row[lo:hi]
        outside = np.concatenate([row[:lo], row[hi:]])
        base = float(np.median(outside))
        sigma = 1.4826 * float(np.median(np.abs(outside - base)))
        depth = base - float(inside.mean())
        snr = depth / (sigma / np.sqrt(w)) if sigma > 0 else 0.0
        out[w] = {"lo": lo, "hi": hi, "base": base, "depth": depth, "snr": snr}
    return out


def box_snr(row: np.ndarray) -> float:
    """Baseline score: the best SNR over all widths, floored at 0 (as in training)."""
    return max(0.0, max(v["snr"] for v in box_snr_per_width(row).values()))


# ---------------------------------------------------------------------------
# 4. Light-curve preprocessing (same steps as build_dataset.make_views)
# ---------------------------------------------------------------------------
def fold_to_global_view(time: np.ndarray, flux: np.ndarray, period: float,
                        epoch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase-fold a flattened light curve and resample it to 201 points.

    Returns (global_view, phase, flux) - the last two are the raw folded points,
    kept so the dashboard can plot them behind the resampled view.
    """
    phase = ((time - epoch) / period) % 1.0
    phase[phase > 0.5] -= 1.0            # centre transit at phase 0, range [-0.5, 0.5]
    order = np.argsort(phase)
    phase, flux = phase[order], flux[order]
    keep = np.isfinite(flux)
    phase, flux = phase[keep], flux[keep]
    if len(flux) < L_GLOBAL:
        raise ValueError(f"Only {len(flux)} usable points - too few to build a view.")
    glob = np.interp(np.linspace(-0.5, 0.5, L_GLOBAL), phase, flux)
    return glob, phase, flux


def make_global_view(kepid: int, period: float, epoch: float):
    """Download a Kepler light curve from MAST and turn it into a global view.

    Pipeline: first long-cadence file -> remove_nans -> normalize ->
    flatten(window_length=401) -> phase-fold -> resample to 201 points.
    Network-bound and can be slow; callers should wrap it in try/except.
    """
    import lightkurve as lk  # imported lazily: it is slow to import

    sr = lk.search_lightcurve(f"KIC {int(kepid)}", author="Kepler", exptime="long")
    if len(sr) == 0:
        raise LookupError(f"No Kepler long-cadence light curve found for KIC {kepid}.")
    flat = sr[0].download().remove_nans().normalize().flatten(window_length=401)
    t = np.asarray(flat.time.value, dtype=float)
    f = np.asarray(flat.flux.value, dtype=float)
    glob, phase, flux = fold_to_global_view(t, f, period, epoch)
    return glob, phase, flux, t, f
