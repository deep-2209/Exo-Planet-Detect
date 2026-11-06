"""Exo-Detect - interactive Streamlit dashboard.

Run from the repo root:
    uv run streamlit run dashboard/app.py

How this file is organised (top to bottom):
    0. Imports, paths, page config
    1. Design tokens (colours) + global CSS + a shared Plotly styling helper
    2. Cached loaders: dataset, model, results JSON, whole-test-set scores
    3. Reusable chart builders (light curve, gauge, animations, ...)
    4. One function per page: overview, explore, compare, architecture, try-any-star
    5. Navigation (sidebar) - wires the page functions together

All the maths (CNN, normalisation, box-SNR baseline, light-curve folding) lives in
dashboard/model.py; this file only loads data, calls those functions and draws.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from model import (
    BOX_WIDTHS,
    L_GLOBAL,
    box_snr,
    box_snr_per_width,
    compute_scale,
    conv_activations,
    layer_shapes,
    load_model,
    make_global_view,
    predict_proba,
)
from plotly.subplots import make_subplots
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# 0. Paths + page config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent      # repo root (dashboard/..)
DATASET = ROOT / "dataset.npz"
WEIGHTS = ROOT / "model.pth"
KOI_TABLE = ROOT / "koi_table.csv"

st.set_page_config(
    page_title="Exo-Detect · Kepler transit classifier",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 1. Design tokens, CSS, Plotly style
# ---------------------------------------------------------------------------
# One colour per *meaning*, used everywhere so the reader learns it once.
C = {
    "bg": "#0b0f1a",
    "card": "#141a2a",
    "grid": "#232b3d",
    "text": "#e8eaf0",
    "text2": "#a9b0c2",
    "muted": "#6b7489",
    "flux": "#3987e5",      # the light-curve data itself
    "cnn": "#3987e5",       # the CNN (it "is" the model reading the flux)
    "baseline": "#9085e9",  # classical box-SNR baseline
    "planet": "#199e70",    # class 1: confirmed planet
    "fp": "#d95926",        # class 0: false positive
    "chance": "#6b7489",    # random-guess reference lines
}
CLASS_NAME = {1: "Confirmed planet", 0: "False positive"}
CLASS_COLOR = {1: C["planet"], 0: C["fp"]}

CSS = f"""
<style>
  .block-container {{ padding-top: 2rem; max-width: 1300px; }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
  .hero {{
    padding: 1.6rem 1.8rem; border-radius: 16px; margin-bottom: 1.2rem;
    background: radial-gradient(1200px 300px at 0% 0%, #1d3a6b 0%, transparent 60%),
                radial-gradient(800px 260px at 100% 100%, #2b2160 0%, transparent 60%),
                {C['card']};
    border: 1px solid {C['grid']};
  }}
  .hero h1 {{ margin: 0; font-size: 2.1rem; }}
  .hero p  {{ margin: .35rem 0 0; color: {C['text2']}; font-size: 1.02rem; }}
  .eyebrow {{ color: {C['flux']}; font-weight: 600; font-size: .8rem;
             letter-spacing: .12em; text-transform: uppercase; }}
  .card {{ background: {C['card']}; border: 1px solid {C['grid']}; border-radius: 12px;
          padding: 1rem 1.1rem; height: 100%; }}
  .card h4 {{ margin: 0 0 .35rem; font-size: 1rem; }}
  .card p, .card li {{ color: {C['text2']}; font-size: .93rem; margin: 0; }}
  .kpi {{ background: {C['card']}; border: 1px solid {C['grid']}; border-radius: 12px;
         padding: .9rem 1rem; }}
  .kpi .lbl {{ color: {C['text2']}; font-size: .8rem; text-transform: uppercase;
              letter-spacing: .06em; }}
  .kpi .val {{ color: {C['text']}; font-size: 1.7rem; font-weight: 650; line-height: 1.2; }}
  .kpi .sub {{ color: {C['muted']}; font-size: .8rem; }}
  .pill {{ display: inline-block; padding: .15rem .6rem; border-radius: 999px;
          font-size: .8rem; font-weight: 600; color: {C['text']}; }}
  /* pipeline + architecture diagrams: a row of boxes joined by arrows */
  .flow {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: .4rem; }}
  .node {{ flex: 1 1 130px; background: {C['card']}; border: 1px solid {C['grid']};
          border-top: 3px solid var(--accent); border-radius: 10px; padding: .7rem .75rem; }}
  .node .t {{ font-weight: 650; font-size: .92rem; color: {C['text']}; }}
  .node .d {{ color: {C['text2']}; font-size: .8rem; margin-top: .2rem; }}
  .node .s {{ font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .8rem;
             color: var(--accent); margin-top: .35rem; }}
  .arrow {{ align-self: center; color: {C['muted']}; font-size: 1.2rem; }}
  .caption {{ color: {C['muted']}; font-size: .82rem; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 340, **layout) -> go.Figure:
    """Apply the shared dark look to any Plotly figure (recessive grid, our fonts)."""
    base = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C["text"], size=13),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
        hoverlabel=dict(bgcolor=C["card"], bordercolor=C["grid"], font_color=C["text"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_layout(**(base | layout))   # caller's keys override the defaults
    fig.update_xaxes(gridcolor=C["grid"], zerolinecolor=C["grid"], linecolor=C["grid"])
    fig.update_yaxes(gridcolor=C["grid"], zerolinecolor=C["grid"], linecolor=C["grid"])
    return fig


def show(fig: go.Figure, key: str | None = None) -> None:
    """Render a figure; theme=None keeps our styling instead of Streamlit's."""
    st.plotly_chart(fig, theme=None, key=key, config={"displaylogo": False})


def kpi(label: str, value: str, sub: str = "", accent: str | None = None) -> str:
    """HTML for one stat tile (label / big number / small caption)."""
    border = f"border-left: 4px solid {accent};" if accent else ""
    return (f"<div class='kpi' style='{border}'><div class='lbl'>{label}</div>"
            f"<div class='val'>{value}</div><div class='sub'>{sub}</div></div>")


def kpi_row(tiles: list[str]) -> None:
    for col, html in zip(st.columns(len(tiles)), tiles, strict=True):
        col.markdown(html, unsafe_allow_html=True)


def hero(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(f"<div class='hero'><div class='eyebrow'>{eyebrow}</div>"
                f"<h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. Cached loaders
# ---------------------------------------------------------------------------
# st.cache_data   -> for data (copied per call, safe to mutate)
# st.cache_resource -> for a single shared object (the PyTorch model)
@st.cache_data(show_spinner="Loading dataset...")
def load_dataset() -> dict[str, np.ndarray]:
    """All arrays from dataset.npz, plus the training normalisation scale."""
    with np.load(DATASET) as d:
        data = {k: d[k] for k in d.files}
    data["scale"] = np.float64(compute_scale(data["X_train"]))
    return data


@st.cache_resource(show_spinner="Loading CNN weights...")
def get_model():
    return load_model(WEIGHTS)


def _read_json(*candidates: Path) -> dict | None:
    """Return the first JSON file that exists among `candidates`."""
    for p in candidates:
        if p.exists():
            return json.loads(p.read_text())
    return None


@st.cache_data
def load_results() -> dict:
    """Metrics files used by the comparison page:
    - cnn_results.json (root): the single SageMaker run that produced model.pth
    - results/cnn_results_without_local_view.json: 5-seed local run of the SAME
      global-view architecture (gives mean +/- std; the README's headline numbers)
    Note: results/cnn_results.json is the dual-view (global + local) experiment -
    a different model - so it is deliberately NOT used here.
    """
    deployed = _read_json(ROOT / "cnn_results.json")
    multi = _read_json(ROOT / "results" / "cnn_results_without_local_view.json")
    baseline = _read_json(ROOT / "baseline_results.json",
                          ROOT / "results" / "baseline_results.json")
    return {
        "deployed": deployed["cnn"] if deployed else None,
        "deployed_seeds": deployed.get("seeds") if deployed else None,
        "multi": multi["cnn"] if multi else None,
        "multi_seeds": multi.get("seeds") if multi else None,
        "baseline": baseline,
    }


def _best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Pick the score cutoff that maximises F1 on the *validation* set - the same
    rule training used, so dashboard verdicts match the reported precision/recall."""
    return float(max(np.unique(scores),
                     key=lambda t: f1_score(y, (scores >= t).astype(int), zero_division=0)))


@st.cache_data(show_spinner="Scoring the test set...")
def score_all(_model) -> dict[str, np.ndarray | float]:
    """Run the CNN and the baseline over val + test once, and derive thresholds.
    (`_model` starts with an underscore so Streamlit doesn't try to hash it.)"""
    d = load_dataset()
    s = float(d["scale"])
    out = {
        "cnn_val": predict_proba(_model, d["X_val"], s),
        "cnn_test": predict_proba(_model, d["X_test"], s),
        "base_val": np.array([box_snr(r) for r in d["X_val"]]),
        "base_test": np.array([box_snr(r) for r in d["X_test"]]),
    }
    out["cnn_thr"] = _best_f1_threshold(d["y_val"], out["cnn_val"])
    out["base_thr"] = _best_f1_threshold(d["y_val"], out["base_val"])
    return out


@st.cache_data
def load_koi() -> pd.DataFrame | None:
    return pd.read_csv(KOI_TABLE) if KOI_TABLE.exists() else None


# ---------------------------------------------------------------------------
# 3. Chart builders
# ---------------------------------------------------------------------------
PHASE = np.linspace(-0.5, 0.5, L_GLOBAL)   # x-axis of every global view


def lightcurve_fig(y: np.ndarray, x: np.ndarray, title: str, xlab: str,
                   box: tuple[float, float] | None = None, height: int = 320) -> go.Figure:
    """A single flux line. Optional `box` = (x0, x1) shades the baseline's best box."""
    fig = go.Figure()
    if box:
        fig.add_vrect(x0=box[0], x1=box[1], fillcolor=C["baseline"], opacity=0.15,
                      line_width=0, annotation_text="baseline box",
                      annotation_font_color=C["text2"], annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", name="flux",
        line=dict(color=C["flux"], width=2), marker=dict(size=4),
        hovertemplate="phase %{x:.3f}<br>flux %{y:.5f}<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color=C["muted"], dash="dot", width=1))
    style(fig, height, title=dict(text=title, font=dict(size=15)), showlegend=False,
          hovermode="x unified")
    fig.update_xaxes(title=xlab)
    fig.update_yaxes(title="normalised flux", tickformat=".4f")
    return fig


def gauge_fig(prob: float, thr: float, verdict_color: str) -> go.Figure:
    """Confidence gauge: needle = P(planet); marker line = decision threshold."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number=dict(suffix="%", font=dict(size=44, color=C["text"]), valueformat=".1f"),
        title=dict(text="CNN P(planet)", font=dict(size=14, color=C["text2"])),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=C["muted"], tickfont=dict(color=C["text2"])),
            bar=dict(color=verdict_color, thickness=0.28),
            bgcolor=C["card"], borderwidth=0,
            steps=[dict(range=[0, thr * 100], color="rgba(217,89,38,0.18)"),
                   dict(range=[thr * 100, 100], color="rgba(25,158,112,0.18)")],
            threshold=dict(line=dict(color=C["text"], width=3), thickness=0.85,
                           value=thr * 100),
        ),
    ))
    return style(fig, 260, margin=dict(l=25, r=25, t=50, b=0))


def fold_animation_fig(glob: np.ndarray, period: float, n_cycles: int = 5,
                       n_frames: int = 24, seed: int = 0) -> go.Figure:
    """Animated *illustration* of phase folding.

    We don't keep raw time series in dataset.npz, so we rebuild a plausible one:
    repeat this star's folded curve over `n_cycles` orbits (with a little noise so
    the copies are distinguishable), lay them end-to-end in time, then animate
    every orbit sliding onto the same phase axis. Frame 0 = time series; last
    frame = the folded view the CNN sees.
    """
    rng = np.random.default_rng(seed)
    noise = 1.4826 * np.median(np.abs(glob - np.median(glob))) * 0.6
    offsets = np.arange(n_cycles) - (n_cycles - 1) / 2       # e.g. -2..2 orbits
    # sequential blue steps (ordinal: orbit 1 -> orbit N)
    shades = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#1c5cab", "#184f95"]
    flux = [glob + rng.normal(0, noise, glob.size) for _ in range(n_cycles)]

    def traces(a: float) -> list[go.Scatter]:
        # a=0 -> unfolded (x = phase + k orbits); a=1 -> folded (x = phase)
        return [go.Scatter(x=PHASE + (1 - a) * k, y=f, mode="markers",
                           marker=dict(size=5, color=shades[i % len(shades)]),
                           name=f"orbit {i + 1}",
                           hovertemplate=f"orbit {i + 1}<br>flux %{{y:.5f}}<extra></extra>")
                for i, (k, f) in enumerate(zip(offsets, flux, strict=True))]

    steps = np.linspace(0, 1, n_frames) ** 2   # ease-in so the fold "snaps" shut
    frames = [go.Frame(data=traces(a), name=f"{i}") for i, a in enumerate(steps)]
    fig = go.Figure(data=traces(0.0), frames=frames)
    span = n_cycles / 2 + 0.05
    style(fig, 380, title=dict(text=f"Folding {n_cycles} orbits (P = {period:.3f} d)",
                               font=dict(size=15)))
    fig.update_xaxes(range=[-span, span], title="time, in orbital periods  →  orbital phase")
    fig.update_yaxes(title="normalised flux", tickformat=".4f")
    fig.update_layout(
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.18, xanchor="left",
                          buttons=[
                              dict(label="▶ Fold", method="animate",
                                   args=[None, dict(frame=dict(duration=70, redraw=False),
                                                    transition=dict(duration=0),
                                                    fromcurrent=False)]),
                              dict(label="↺ Reset", method="animate",
                                   args=[["0"], dict(frame=dict(duration=0, redraw=False),
                                                     mode="immediate")]),
                          ], bgcolor=C["card"], font=dict(color=C["text"]))],
        sliders=[dict(active=0, x=0.18, len=0.82, y=-0.12, pad=dict(t=20),
                      currentvalue=dict(prefix="fold progress: ", font=dict(color=C["text2"])),
                      steps=[dict(label=f"{int(a * 100)}%", method="animate",
                                  args=[[f"{i}"], dict(frame=dict(duration=0, redraw=False),
                                                       mode="immediate")])
                             for i, a in enumerate(steps)])],
        margin=dict(l=10, r=10, t=60, b=90),
    )
    return fig


def box_sweep_fig(glob: np.ndarray) -> go.Figure:
    """Animated baseline: step through each box width and show depth + SNR.

    Each frame shades the box, draws the out-of-box median (the 'continuum') and
    the in-box mean (the dip level). The vertical gap between them is the depth;
    SNR = depth / (noise / sqrt(width)).
    """
    per_w = box_snr_per_width(glob)
    best_w = max(per_w, key=lambda w: per_w[w]["snr"])

    def layout_for(w: int) -> dict:
        r = per_w[w]
        x0, x1 = PHASE[r["lo"]], PHASE[r["hi"] - 1]
        inside_mean = r["base"] - r["depth"]
        tag = "  ← best" if w == best_w else ""
        return dict(
            shapes=[
                dict(type="rect", x0=x0, x1=x1, y0=0, y1=1, yref="paper",
                     fillcolor=C["baseline"], opacity=0.18, line_width=0),
                dict(type="line", x0=-0.5, x1=0.5, y0=r["base"], y1=r["base"],
                     line=dict(color=C["text2"], dash="dash", width=1.5)),
                dict(type="line", x0=x0, x1=x1, y0=inside_mean, y1=inside_mean,
                     line=dict(color=C["baseline"], width=3)),
            ],
            annotations=[dict(x=0.01, y=0.04, xref="paper", yref="paper", showarrow=False,
                              align="left", font=dict(size=14, color=C["text"]),
                              bgcolor=C["card"], bordercolor=C["grid"],
                              text=f"width = {w} bins · depth = {r['depth'] * 1e6:,.0f} ppm"
                                   f" · <b>SNR = {r['snr']:.2f}</b>{tag}")],
        )

    curve = go.Scatter(x=PHASE, y=glob, mode="lines", line=dict(color=C["flux"], width=2),
                       name="flux", hovertemplate="phase %{x:.3f}<br>%{y:.5f}<extra></extra>")
    frames = [go.Frame(data=[curve], name=str(w), layout=layout_for(w)) for w in BOX_WIDTHS]
    fig = go.Figure(data=[curve], frames=frames, layout=layout_for(BOX_WIDTHS[0]))
    style(fig, 380, title=dict(text="Baseline box sweep (centred on the transit)",
                               font=dict(size=15)), showlegend=False)
    fig.update_xaxes(title="orbital phase")
    fig.update_yaxes(title="normalised flux", tickformat=".4f")
    fig.update_layout(
        updatemenus=[dict(type="buttons", x=0, y=-0.18, xanchor="left",
                          buttons=[dict(label="▶ Sweep widths", method="animate",
                                        args=[None, dict(frame=dict(duration=900, redraw=True),
                                                         transition=dict(duration=300),
                                                         fromcurrent=False)])],
                          bgcolor=C["card"], font=dict(color=C["text"]))],
        sliders=[dict(active=0, x=0.22, len=0.78, y=-0.12, pad=dict(t=20),
                      currentvalue=dict(prefix="box width: ", font=dict(color=C["text2"])),
                      steps=[dict(label=str(w), method="animate",
                                  args=[[str(w)], dict(frame=dict(duration=0, redraw=True),
                                                       mode="immediate")])
                             for w in BOX_WIDTHS])],
        margin=dict(l=10, r=10, t=60, b=90),
    )
    return fig


def score_hist_fig(scores: np.ndarray, y: np.ndarray, mark: float, thr: float,
                   title: str, xlab: str) -> go.Figure:
    """Overlaid histograms of a score by true class, with this star + threshold marked."""
    fig = go.Figure()
    for cls in (0, 1):
        fig.add_trace(go.Histogram(x=scores[y == cls], name=CLASS_NAME[cls], nbinsx=30,
                                   marker=dict(color=CLASS_COLOR[cls],
                                               line=dict(color=C["bg"], width=1)),
                                   opacity=0.7))
    fig.add_vline(x=thr, line=dict(color=C["text2"], dash="dash", width=1.5),
                  annotation_text="threshold", annotation_font_color=C["text2"])
    fig.add_vline(x=mark, line=dict(color=C["text"], width=3),
                  annotation_text="this star", annotation_position="top right",
                  annotation_font_color=C["text"])
    style(fig, 300, barmode="overlay", title=dict(text=title, font=dict(size=15)))
    fig.update_xaxes(title=xlab)
    fig.update_yaxes(title="test stars")
    return fig


def verdict(prob: float, thr: float) -> tuple[int, str, str]:
    cls = int(prob >= thr)
    return cls, CLASS_NAME[cls], CLASS_COLOR[cls]


# ---------------------------------------------------------------------------
# 4. Pages
# ---------------------------------------------------------------------------
def page_overview() -> None:
    d, res = load_dataset(), load_results()
    sc = score_all(get_model())
    hero("Exo-Detect",
         "Finding planets in starlight dips",
         "A 1D convolutional network that reads NASA Kepler light curves and decides whether "
         "a transit candidate is a <b>confirmed planet</b> or a <b>false positive</b>, "
         "benchmarked against a classical box-fit detector.")

    n_tot = sum(len(d[f"y_{s}"]) for s in ("train", "val", "test"))
    cnn_auc = roc_auc_score(d["y_test"], sc["cnn_test"])
    base_auc = res["baseline"]["roc_auc"] if res["baseline"] else float("nan")
    kpi_row([
        kpi("Kepler stars", f"{n_tot:,}", f"{len(d['y_test'])} held out for testing", C["flux"]),
        kpi("Class balance", f"{d['y_test'].mean():.0%}", "of test stars are planets",
            C["planet"]),
        kpi("CNN ROC-AUC", f"{cnn_auc:.3f}", "deployed model.pth, test set", C["cnn"]),
        kpi("Baseline ROC-AUC", f"{base_auc:.3f}", "box-fit SNR, test set", C["baseline"]),
    ])
    st.write("")

    # --- the science, in three cards + a schematic transit -----------------
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("### What is a transit?")
        st.markdown(
            "When a planet passes in front of its star, it blocks a tiny fraction of the "
            "light - typically **0.01 % to 1 %**. Kepler stared at ~150,000 stars for four "
            "years and recorded their brightness every 30 minutes. A planet shows up as a "
            "small, flat-bottomed **dip that repeats every orbit**.")
        st.markdown("### Planet or impostor?")
        st.markdown(
            "Many dips are *not* planets. **Eclipsing binary stars**, light blended in from a "
            "background star, and spacecraft systematics all produce look-alike signals. "
            "NASA's Kepler Objects of Interest (KOI) table labels each candidate "
            "**CONFIRMED** or **FALSE POSITIVE** - our two classes.")
        st.markdown("### Phase folding")
        st.markdown(
            "A single dip is buried in noise. If we know the orbital period, we can cut the "
            "light curve into one-orbit chunks and **stack them on top of each other** "
            "(\"fold\" them). Every transit lands at the same place - phase 0 - so the dip "
            "adds up while random noise averages out. Each star becomes a fixed **201-point "
            "vector**: one full orbit, transit in the middle. See it animated on the "
            "*Explore* page.")
    with right:
        # Schematic only: a trapezoid dip + noise, clearly labelled as such.
        rng = np.random.default_rng(3)
        ph = np.linspace(-0.5, 0.5, 400)
        depth, half, ingress = 0.006, 0.035, 0.012
        model_flux = 1 - depth * np.clip((half - np.abs(ph)) / ingress, 0, 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ph, y=model_flux + rng.normal(0, 0.0012, ph.size),
                                 mode="markers", marker=dict(size=4, color=C["muted"]),
                                 name="observed (noisy)", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=ph, y=model_flux, mode="lines", name="transit shape",
                                 line=dict(color=C["flux"], width=3), hoverinfo="skip"))
        fig.add_annotation(x=0, y=1 - depth, ax=110, ay=30, text="planet blocks light",
                           font=dict(color=C["text"]), arrowcolor=C["text2"])
        style(fig, 380, title=dict(text="Schematic: a folded transit", font=dict(size=15)))
        fig.update_xaxes(title="orbital phase")
        fig.update_yaxes(title="relative brightness", tickformat=".3f")
        show(fig, key="schematic")
        st.markdown("<div class='caption'>Synthetic illustration, not real data.</div>",
                    unsafe_allow_html=True)

    # --- pipeline diagram (styled HTML boxes) --------------------------------
    st.markdown("### The pipeline")
    stages = [
        (C["flux"], "Kepler light curves", "Long-cadence photometry downloaded from MAST "
                                           "via lightkurve", "~4 yrs / star"),
        (C["planet"], "NASA KOI labels", "CONFIRMED → 1, FALSE POSITIVE → 0; "
                                         "candidates dropped", "balanced classes"),
        (C["flux"], "Preprocess", "remove NaNs → normalise → flatten (401) → fold on "
                                  "period/epoch → resample", "global 201 · local 61"),
        (C["muted"], "Split", "stratified 70 / 15 / 15 train / val / test", "no leakage"),
        (C["cnn"], "1D CNN", "3 conv blocks + linear head on the global view",
         "13k params"),
        (C["baseline"], "Box-SNR baseline", "best box-shaped dip vs. robust noise",
         "no training"),
        (C["text2"], "Evaluate", "ROC-AUC, PR-AUC, precision / recall / F1 on the held-out "
                                 "test set", "threshold tuned on val"),
    ]
    html = "<div class='flow'>"
    for i, (acc, t, desc, s) in enumerate(stages):
        if i:
            html += "<div class='arrow'>→</div>"
        html += (f"<div class='node' style='--accent:{acc}'><div class='t'>{t}</div>"
                 f"<div class='d'>{desc}</div><div class='s'>{s}</div></div>")
    st.markdown(html + "</div>", unsafe_allow_html=True)
    st.write("")
    st.info("Use the sidebar to explore individual stars, compare the two models, look inside "
            "the CNN, or try a star straight from the Kepler archive.", icon="🧭")


def page_explore() -> None:
    d, model = load_dataset(), get_model()
    sc = score_all(model)
    hero("Explore", "Inspect a light curve",
         "Pick a held-out test star, see exactly what the CNN sees, and compare its verdict "
         "with the classical baseline.")

    y, meta = d["y_test"], d["meta_test"]

    # --- star picker: class filter + selectbox + random button ---------------
    c1, c2, c3 = st.columns([1.2, 2.6, 0.8], vertical_alignment="bottom")
    with c1:
        flt = st.segmented_control("Show", ["All", "Planets", "False positives"],
                                   default="All", key="explore_filter")
    pool = np.arange(len(y))
    if flt == "Planets":
        pool = pool[y == 1]
    elif flt == "False positives":
        pool = pool[y == 0]

    def label(i: int) -> str:
        return (f"KIC {int(meta[i, 0])} · {CLASS_NAME[int(y[i])]} · "
                f"P = {meta[i, 1]:.2f} d")

    # Keep the current choice valid when the filter changes.
    if st.session_state.get("star_idx") not in set(pool.tolist()):
        st.session_state["star_idx"] = int(pool[0])

    def pick_random() -> None:
        st.session_state["star_idx"] = int(np.random.default_rng().choice(pool))

    with c2:
        idx = st.selectbox("Test star", pool.tolist(), format_func=label, key="star_idx")
    # Widget state is dropped when a page doesn't render the widget, so copy the
    # choice to a plain key that the Architecture page can read.
    st.session_state["chosen_star"] = idx
    with c3:
        st.button("🎲 Random star", on_click=pick_random, width="stretch")

    glob, loc = d["X_test"][idx], d["Xl_test"][idx]
    kepid, period, epoch = int(meta[idx, 0]), float(meta[idx, 1]), float(meta[idx, 2])
    true_cls = int(y[idx])

    # --- live forward pass (one star) + baseline -----------------------------
    prob = float(predict_proba(model, glob, float(d["scale"]))[0])
    pred_cls, pred_name, pred_color = verdict(prob, sc["cnn_thr"])
    per_w = box_snr_per_width(glob)
    snr = box_snr(glob)
    best_w = max(per_w, key=lambda w: per_w[w]["snr"])
    base_cls = int(snr >= sc["base_thr"])

    correct = "✔ correct" if pred_cls == true_cls else "✘ incorrect"
    kpi_row([
        kpi("True disposition", CLASS_NAME[true_cls], f"KIC {kepid} · NASA KOI",
            CLASS_COLOR[true_cls]),
        kpi("CNN verdict", pred_name, f"{correct} · threshold {sc['cnn_thr']:.3f}",
            pred_color),
        kpi("CNN P(planet)", f"{prob:.1%}", "live forward pass on the global view", C["cnn"]),
        kpi("Baseline SNR", f"{snr:.2f}", f"best box = {best_w} bins → "
            f"{CLASS_NAME[base_cls].lower()}", C["baseline"]),
    ])
    st.write("")

    # --- views + gauge --------------------------------------------------------
    left, right = st.columns([2.2, 1])
    with left:
        r = per_w[best_w]
        show(lightcurve_fig(glob, PHASE, f"Global view - one full orbit (P = {period:.3f} d)",
                            "orbital phase (transit at 0)",
                            box=(PHASE[r["lo"]], PHASE[r["hi"] - 1])), key="glob")
    with right:
        show(gauge_fig(prob, sc["cnn_thr"], pred_color), key="gauge")
        st.markdown(
            f"<div class='caption'>Green zone = called a planet. The white tick is the "
            f"decision threshold ({sc['cnn_thr']:.3f}), chosen to maximise F1 on the "
            f"validation set. This model's raw scores are compressed (test max "
            f"{sc['cnn_test'].max():.2f}), so read the needle relative to the tick, not to "
            f"50 %.</div>", unsafe_allow_html=True)

    left, right = st.columns([2.2, 1])
    with left:
        # Local view spans +/- 3 transit durations; the duration isn't stored, so the
        # x-axis is shown in normalised units (-1 .. +1 of the window).
        show(lightcurve_fig(loc, np.linspace(-1, 1, loc.size),
                            "Local view - zoomed on the transit (61 points)",
                            "position in zoom window (±3 transit durations)",
                            height=280), key="loc")
        st.markdown("<div class='caption'>The deployed model uses only the global view; the "
                    "local view is shown for context (it was used in the dual-view "
                    "experiment).</div>", unsafe_allow_html=True)
    with right:
        st.markdown("#### Star details")
        st.markdown(
            f"- **KIC id:** {kepid}\n"
            f"- **Orbital period:** {period:.4f} days\n"
            f"- **Epoch (BKJD):** {epoch:.4f}\n"
            f"- **Box depth:** {per_w[best_w]['depth'] * 1e6:,.0f} ppm\n"
            f"- **Baseline threshold:** SNR ≥ {sc['base_thr']:.2f}")
        st.link_button("View in NASA Exoplanet Archive",
                       f"https://exoplanetarchive.ipac.caltech.edu/overview/KIC%20{kepid}",
                       width="stretch")

    # --- animations + context -------------------------------------------------
    st.markdown("### See it in motion")
    t1, t2, t3 = st.tabs(["🌀 Phase folding", "📦 Baseline box sweep",
                          "📊 Where this star sits"])
    with t1:
        show(fold_animation_fig(glob, period), key="fold")
        st.markdown("<div class='caption'>Illustration: dataset.npz stores only the folded "
                    "curve, so the 'raw' orbits here are copies of it with added noise. Press "
                    "▶ Fold to watch them collapse onto one phase axis - the dip reinforces "
                    "while the scatter averages out.</div>", unsafe_allow_html=True)
    with t2:
        a, b = st.columns([2.2, 1])
        with a:
            show(box_sweep_fig(glob), key="sweep")
        with b:
            bar = go.Figure(go.Bar(
                x=[str(w) for w in BOX_WIDTHS], y=[per_w[w]["snr"] for w in BOX_WIDTHS],
                marker=dict(color=[C["baseline"] if w == best_w else C["grid"]
                                   for w in BOX_WIDTHS]),
                hovertemplate="width %{x}<br>SNR %{y:.2f}<extra></extra>"))
            style(bar, 380, title=dict(text="SNR per box width", font=dict(size=15)),
                  barcornerradius=4, showlegend=False)
            bar.update_xaxes(title="box width (bins)", type="category")
            bar.update_yaxes(title="SNR")
            show(bar, key="snrbar")
        st.markdown("<div class='caption'>The baseline asks one question: <i>how deep is the "
                    "dip at phase 0, relative to the noise?</i> It tries five box widths and "
                    "keeps the best SNR. Dashed line = out-of-box median; purple bar = in-box "
                    "mean.</div>", unsafe_allow_html=True)
    with t3:
        a, b = st.columns(2)
        with a:
            show(score_hist_fig(sc["cnn_test"], y, prob, sc["cnn_thr"],
                                "CNN probability across the test set", "P(planet)"),
                 key="hcnn")
        with b:
            clip = np.clip(sc["base_test"], 0, np.percentile(sc["base_test"], 98))
            show(score_hist_fig(clip, y, min(snr, clip.max()), sc["base_thr"],
                                "Baseline SNR across the test set (98th pct clip)", "SNR"),
                 key="hbase")


def page_compare() -> None:
    d, res = load_dataset(), load_results()
    sc = score_all(get_model())
    hero("Benchmark", "CNN vs classical baseline",
         "Held-out test-set performance. Threshold-free metrics (ROC-AUC, PR-AUC) are the "
         "fairest comparison; precision / recall / F1 depend on the chosen cutoff.")

    y = d["y_test"]
    prevalence = float(y.mean())
    multi, dep, base = res["multi"], res["deployed"], res["baseline"]
    if base is None or (multi is None and dep is None):
        st.error("Results files not found (cnn_results.json / baseline_results.json).")
        return
    cnn = multi or dep                      # prefer the multi-seed run for error bars
    n_seeds = len(res["multi_seeds"]) if multi and isinstance(res["multi_seeds"], list) else 1

    # --- table ------------------------------------------------------------------
    metrics = [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("Precision", "precision"),
               ("Recall", "recall"), ("F1", "f1")]
    chance = {"roc_auc": 0.5, "pr_auc": prevalence, "precision": prevalence,
              "recall": None, "f1": None}
    rows = []
    for name, k in metrics:
        row = {"Metric": name,
               f"CNN ({n_seeds} seeds, mean ± std)": f"{cnn[k]['mean']:.3f} ± {cnn[k]['std']:.3f}"}
        if dep:
            row["CNN · deployed model.pth"] = f"{dep[k]['mean']:.3f}"
        row["Box-SNR baseline"] = f"{base[k]:.3f}"
        row["Random guess"] = "-" if chance[k] is None else f"{chance[k]:.3f}"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # --- bar charts with error bars + chance line --------------------------------
    def bar_fig(k: str, title: str, chance_val: float) -> go.Figure:
        names = [f"CNN ({n_seeds} seeds)"] + (["CNN deployed"] if dep else []) + ["Baseline"]
        vals = [cnn[k]["mean"]] + ([dep[k]["mean"]] if dep else []) + [base[k]]
        errs = [cnn[k]["std"]] + ([0] if dep else []) + [0]
        cols = [C["cnn"]] + (["#256abf"] if dep else []) + [C["baseline"]]
        fig = go.Figure(go.Bar(
            x=names, y=vals, marker=dict(color=cols),
            error_y=dict(type="data", array=errs, color=C["text"], thickness=1.5, width=6),
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            textfont=dict(color=C["text"]),
            hovertemplate="%{x}<br>%{y:.3f}<extra></extra>"))
        fig.add_hline(y=chance_val, line=dict(color=C["chance"], dash="dash", width=2),
                      annotation_text=f"random guess = {chance_val:.2f}",
                      annotation_font_color=C["text2"], annotation_position="top right")
        style(fig, 360, title=dict(text=title, font=dict(size=15)), showlegend=False,
              barcornerradius=4, bargap=0.45)
        fig.update_yaxes(range=[0, 1], title=title.split(" ")[0])
        return fig

    a, b = st.columns(2)
    with a:
        show(bar_fig("roc_auc", "ROC-AUC (higher is better)", 0.5), key="bar_roc")
    with b:
        show(bar_fig("pr_auc", "PR-AUC (higher is better)", prevalence), key="bar_pr")

    st.success(
        f"**Reading it:** the CNN ranks planets above false positives clearly better than "
        f"chance (ROC-AUC {cnn['roc_auc']['mean']:.2f} vs 0.50), while the box-SNR baseline "
        f"sits at or below chance ({base['roc_auc']:.2f}). A deep, clean dip is *not* "
        f"enough to call a planet - many false positives (e.g. eclipsing binaries) have dips "
        f"as deep or deeper - so the CNN's ability to learn dip **shape** is what adds signal.",
        icon="💡")
    st.caption(
        f"Precision {cnn['precision']['mean']:.2f} / recall {cnn['recall']['mean']:.2f}: at the "
        f"validation-tuned threshold the CNN flags almost every planet but also many false "
        f"positives - useful as a first-pass filter, not a final arbiter. The baseline's F1-best "
        f"threshold was {base.get('threshold', 0):.2f}, i.e. it labels nearly everything a "
        f"planet, which is why its recall is 1.0.")

    # --- live curves from the deployed model ------------------------------------
    st.markdown("### Curves from the deployed model (computed live)")
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for name, s, col in [("CNN", sc["cnn_test"], C["cnn"]),
                             ("Baseline", sc["base_test"], C["baseline"])]:
            fpr, tpr, _ = roc_curve(y, s)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=col, width=2.5),
                                     name=f"{name} (AUC {roc_auc_score(y, s):.3f})",
                                     hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="random",
                                 line=dict(color=C["chance"], dash="dash"), hoverinfo="skip"))
        style(fig, 380, title=dict(text="ROC curve", font=dict(size=15)))
        fig.update_xaxes(title="false-positive rate", range=[0, 1])
        fig.update_yaxes(title="true-positive rate (recall)", range=[0, 1.02])
        show(fig, key="roc")
    with b:
        fig = go.Figure()
        for name, s, col in [("CNN", sc["cnn_test"], C["cnn"]),
                             ("Baseline", sc["base_test"], C["baseline"])]:
            p, r, _ = precision_recall_curve(y, s)
            fig.add_trace(go.Scatter(x=r, y=p, mode="lines", line=dict(color=col, width=2.5),
                                     name=f"{name} (AP {average_precision_score(y, s):.3f})",
                                     hovertemplate="recall %{x:.2f}<br>precision %{y:.2f}"
                                                   "<extra></extra>"))
        fig.add_hline(y=prevalence, line=dict(color=C["chance"], dash="dash"),
                      annotation_text="random", annotation_font_color=C["text2"])
        style(fig, 380, title=dict(text="Precision-recall curve", font=dict(size=15)))
        fig.update_xaxes(title="recall", range=[0, 1])
        fig.update_yaxes(title="precision", range=[0, 1.02])
        show(fig, key="pr")
    st.caption("These curves are recomputed from model.pth on the 426 test stars, so they match "
               "the 'deployed' column above exactly.")


def page_architecture() -> None:
    d, model = load_dataset(), get_model()
    hero("Under the hood", "TransitCNN, layer by layer",
         "A small 1D convolutional network - about 13 thousand parameters - that turns a "
         "201-point folded light curve into a single planet probability.")

    # Shapes are measured from the real model (see model.layer_shapes).
    shp = dict(enumerate(s for _, s in layer_shapes(model)))
    fmt = lambda s: " × ".join(map(str, s))  # noqa: E731
    n_params = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    f = model.features
    blocks = [
        (C["muted"], "Input", "phase-folded flux, (x−1)/scale", fmt(shp[0])),
        (C["cnn"], "Conv1d 1→16 + ReLU", f"kernel 5 · {n_params(f[0]):,} params", fmt(shp[2])),
        (C["planet"], "MaxPool ×2", "keep the strongest response", fmt(shp[3])),
        (C["cnn"], "Conv1d 16→32 + ReLU", f"kernel 5 · {n_params(f[3]):,} params", fmt(shp[5])),
        (C["planet"], "MaxPool ×2", "halve length again", fmt(shp[6])),
        (C["cnn"], "Conv1d 32→64 + ReLU", f"kernel 5 · {n_params(f[6]):,} params", fmt(shp[8])),
        (C["planet"], "Global avg pool", "average over the orbit", fmt(shp[9])),
        (C["baseline"], "Flatten + Dropout 0.3", "regularise (train only)", fmt(shp[11])),
        (C["baseline"], "Linear 64→1", f"{n_params(model.head[2]):,} params → logit",
         fmt(shp[12])),
        (C["fp"], "Sigmoid", "logit → probability", "P(planet)"),
    ]
    html = "<div class='flow'>"
    for i, (acc, t, desc, s) in enumerate(blocks):
        if i:
            html += "<div class='arrow'>→</div>"
        html += (f"<div class='node' style='--accent:{acc}; flex-basis:105px'>"
                 f"<div class='t'>{t}</div><div class='d'>{desc}</div>"
                 f"<div class='s'>{s}</div></div>")
    st.markdown(html + "</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='caption' style='margin-top:.4rem'>Shapes are channels × length, "
                f"measured by running a dummy input through the loaded network. Total "
                f"trainable parameters: <b>{n_params(model):,}</b>.</div>",
                unsafe_allow_html=True)

    # --- tensor-size chart: how length shrinks while channels grow -------------
    stages = ["input", "conv1", "pool1", "conv2", "pool2", "conv3", "avg pool"]
    idxs = [0, 2, 3, 5, 6, 8, 9]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stages, y=[shp[i][1] for i in idxs], name="length (phase bins)",
                             mode="lines+markers+text", line=dict(color=C["flux"], width=2.5),
                             marker=dict(size=9), text=[shp[i][1] for i in idxs],
                             textposition="top center", textfont=dict(color=C["text"])))
    fig.add_trace(go.Scatter(x=stages, y=[shp[i][0] for i in idxs], name="channels (features)",
                             mode="lines+markers+text",
                             line=dict(color=C["planet"], width=2.5, dash="dot"),
                             marker=dict(size=9, symbol="square"),
                             text=[shp[i][0] for i in idxs], textposition="bottom center",
                             textfont=dict(color=C["text"])))
    style(fig, 320, title=dict(text="Where the information goes: length shrinks, features grow",
                               font=dict(size=15)))
    fig.update_yaxes(type="log", title="size (log scale)")
    show(fig, key="sizes")

    # --- plain-language explanation ------------------------------------------
    st.markdown("### What each stage does")
    c1, c2, c3 = st.columns(3)
    cards = [
        ("🔍 Convolutions = pattern detectors",
         "Each filter is a tiny 5-point template slid along the curve. Where the curve "
         "matches the template, the output lights up. Block 1 learns edges (flux starting to "
         "drop / recover); deeper blocks combine those into shapes like <i>U-shaped dip</i> "
         "vs <i>V-shaped dip</i> - a classic clue for telling planets from eclipsing "
         "binaries."),
        ("🗜️ Pooling = summarise & zoom out",
         "Max-pooling keeps the strongest response in each pair of bins, halving the length. "
         "That makes the next layer's 5-point filter cover a wider stretch of the orbit and "
         "makes detection tolerant to small shifts. The final <b>global average pool</b> "
         "squashes each of the 64 feature maps into one number: <i>how much of this pattern "
         "is in the curve?</i>"),
        ("🎯 Classifier = weigh the evidence",
         "A single linear layer weighs the 64 pattern scores into one logit; the sigmoid "
         "turns it into a probability. Dropout randomly hides 30 % of the features during "
         "training so the model can't lean on one feature. At inference (eval mode) dropout "
         "is off and predictions are deterministic."),
    ]
    for col, (t, body) in zip((c1, c2, c3), cards, strict=True):
        col.markdown(f"<div class='card'><h4>{t}</h4><p>{body}</p></div>",
                     unsafe_allow_html=True)

    # --- live feature maps for the star chosen on the Explore page -------------
    st.markdown("### What the filters see")
    idx = int(st.session_state.get("chosen_star", 0))
    kepid = int(d["meta_test"][idx, 0])
    acts = conv_activations(model, d["X_test"][idx], float(d["scale"]))
    layer = st.segmented_control("Conv block", ["Block 1 (16)", "Block 2 (32)", "Block 3 (64)"],
                                 default="Block 1 (16)", key="act_layer")
    layer = layer or "Block 1 (16)"                 # segmented control can be deselected
    li = {"Block 1 (16)": 0, "Block 2 (32)": 1, "Block 3 (64)": 2}[layer]
    a = acts[li]
    xs = np.linspace(-0.5, 0.5, a.shape[1])
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.3, 0.7],
                        vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=PHASE, y=d["X_test"][idx], mode="lines",
                             line=dict(color=C["flux"], width=2), name="input",
                             hovertemplate="phase %{x:.3f}<br>%{y:.5f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Heatmap(z=a, x=xs, y=list(range(a.shape[0])),
                             colorscale=[[0, C["bg"]], [0.5, "#1c5cab"], [1, "#cde2fb"]],
                             colorbar=dict(title="activation", thickness=12),
                             hovertemplate="phase %{x:.3f}<br>filter %{y}<br>"
                                           "activation %{z:.2f}<extra></extra>"),
                  row=2, col=1)
    style(fig, 460, showlegend=False,
          title=dict(text=f"KIC {kepid} - post-ReLU feature maps, {layer}",
                     font=dict(size=15)))
    fig.update_xaxes(title="orbital phase", row=2, col=1)
    fig.update_yaxes(title="flux", tickformat=".4f", row=1, col=1)
    fig.update_yaxes(title="filter #", row=2, col=1)
    show(fig, key="acts")
    st.caption("Each row is one filter's response along the orbit (bright = strong match). "
               "Pick a different star on the Explore page and come back to compare.")


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_star(kepid: int, period: float, epoch: float):
    """Cached wrapper so repeated clicks don't re-download from MAST."""
    return make_global_view(kepid, period, epoch)


def page_try() -> None:
    d, model = load_dataset(), get_model()
    sc = score_all(model)
    hero("Experimental", "Try any Kepler star",
         "Fetch a light curve live from NASA's MAST archive, run it through the same "
         "preprocessing as the training set, and classify it.")
    st.warning("**Experimental.** Downloads from MAST can take 10-60 s or fail when the archive "
               "is busy. Only the first long-cadence quarter is used, as in the training data. "
               "The epoch must be in BKJD (BJD − 2454833), as in the KOI table.", icon="🧪")

    koi = load_koi()
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        kepid = st.number_input("KIC id", min_value=1, value=6922244, step=1, format="%d",
                                help="Default is Kepler-8, host of the hot Jupiter Kepler-8 b.")
    # Offer the KOI table's ephemerides for this star if we have them.
    rows = koi[koi["kepid"] == kepid].reset_index(drop=True) if koi is not None else None
    period0, epoch0, disp = 3.52254, 121.1194, None
    if rows is not None and len(rows):
        with c1:
            pick = st.selectbox(
                "Known KOI signal", range(len(rows)),
                format_func=lambda i: f"P = {rows.koi_period[i]:.4f} d · "
                                      f"{rows.koi_disposition[i].title()}")
        period0, epoch0 = float(rows.koi_period[pick]), float(rows.koi_time0bk[pick])
        disp = rows.koi_disposition[pick]
        key_sfx = f"{kepid}_{pick}"            # new keys -> inputs reset to the KOI values
    else:
        with c1:
            st.caption("Not in the local KOI table - enter period and epoch manually.")
        key_sfx = f"{kepid}_manual"
    with c2:
        period = st.number_input("Period (days)", min_value=0.1, value=period0, step=0.0001,
                                 format="%.5f", key=f"p_{key_sfx}")
    with c3:
        epoch = st.number_input("Epoch (BKJD)", value=epoch0, step=0.0001, format="%.4f",
                                key=f"e_{key_sfx}")

    in_data = any(int(kepid) in set(d[f"meta_{s}"][:, 0].astype(int))
                  for s in ("train", "val", "test"))
    if in_data:
        st.caption("ℹ️ This star is already in the Exo-Detect dataset, so the CNN may have seen "
                   "it in training - treat the result as a demo, not an evaluation.")

    if not st.button("🔭 Fetch & classify", type="primary"):
        return
    try:
        with st.spinner(f"Downloading KIC {kepid} from MAST and folding it..."):
            glob, phase, flux, t, f = fetch_star(int(kepid), float(period), float(epoch))
    except Exception as e:  # network errors, missing data, too few points, ...
        st.error(f"Couldn't fetch or process KIC {kepid}: {e}\n\nMAST may be slow or "
                 f"unavailable - try again in a minute, or try another star.", icon="🛰️")
        return

    prob = float(predict_proba(model, glob, float(d["scale"]))[0])
    pred_cls, pred_name, pred_color = verdict(prob, sc["cnn_thr"])
    snr = box_snr(glob)
    kpi_row([
        kpi("KOI disposition", disp.title() if disp else "unknown", f"KIC {kepid}",
            CLASS_COLOR.get({"CONFIRMED": 1, "FALSE POSITIVE": 0}.get(disp, -1), C["muted"])),
        kpi("CNN verdict", pred_name, f"threshold {sc['cnn_thr']:.3f}", pred_color),
        kpi("CNN P(planet)", f"{prob:.1%}", "live forward pass", C["cnn"]),
        kpi("Baseline SNR", f"{snr:.2f}", f"threshold {sc['base_thr']:.2f}", C["baseline"]),
    ])
    st.write("")

    # Raw flattened light curve (downsampled for speed) and the folded result.
    step = max(1, len(t) // 4000)
    raw = go.Figure(go.Scattergl(x=t[::step], y=f[::step], mode="markers",
                                 marker=dict(size=3, color=C["muted"]),
                                 hovertemplate="t %{x:.2f}<br>%{y:.5f}<extra></extra>"))
    style(raw, 260, title=dict(text="Flattened light curve (time series)", font=dict(size=15)),
          showlegend=False)
    raw.update_xaxes(title="time (BKJD days)")
    raw.update_yaxes(title="flux", tickformat=".4f")
    show(raw, key="raw")

    left, right = st.columns([2.2, 1])
    with left:
        fold = go.Figure()
        sub = max(1, len(phase) // 6000)
        fold.add_trace(go.Scattergl(x=phase[::sub], y=flux[::sub], mode="markers",
                                    marker=dict(size=3, color=C["muted"]), name="folded points",
                                    hoverinfo="skip"))
        fold.add_trace(go.Scatter(x=PHASE, y=glob, mode="lines", name="201-point global view",
                                  line=dict(color=C["flux"], width=2.5)))
        style(fold, 340, title=dict(text="Phase-folded → resampled view", font=dict(size=15)))
        fold.update_xaxes(title="orbital phase")
        fold.update_yaxes(title="flux", tickformat=".4f",
                          range=[np.percentile(flux, 0.5), np.percentile(flux, 99.5)])
        show(fold, key="fold_try")
    with right:
        show(gauge_fig(prob, sc["cnn_thr"], pred_color), key="gauge_try")


# ---------------------------------------------------------------------------
# 5. Navigation
# ---------------------------------------------------------------------------
def main() -> None:
    if not DATASET.exists() or not WEIGHTS.exists():
        st.error(f"Missing {DATASET.name} or {WEIGHTS.name} in {ROOT}. Run the data / "
                 "training scripts first.")
        st.stop()

    with st.sidebar:
        st.markdown("<div style='font-size:1.5rem;font-weight:700'>🪐 Exo-Detect</div>"
                    "<div class='caption'>Kepler transit classifier</div>",
                    unsafe_allow_html=True)

    pages = [
        st.Page(page_overview, title="Overview", icon="🌌", url_path="overview", default=True),
        st.Page(page_explore, title="Explore a light curve", icon="🔭", url_path="explore"),
        st.Page(page_compare, title="Model vs baseline", icon="📊", url_path="compare"),
        st.Page(page_architecture, title="CNN architecture", icon="🧠", url_path="architecture"),
        st.Page(page_try, title="Try any star (beta)", icon="🧪", url_path="try"),
    ]
    nav = st.navigation(pages, position="sidebar")

    with st.sidebar:
        st.divider()
        st.markdown(
            f"<div class='caption'><span class='pill' style='background:{C['planet']}'>planet"
            f"</span> <span class='pill' style='background:{C['fp']}'>false positive</span><br>"
            f"<span class='pill' style='background:{C['cnn']}'>CNN</span> "
            f"<span class='pill' style='background:{C['baseline']}'>baseline</span><br><br>"
            f"Data: NASA Kepler via MAST · labels: KOI cumulative table.</div>",
            unsafe_allow_html=True)
    nav.run()


# Streamlit executes this file as __main__; the guard lets tests import the page
# functions without launching the whole app.
if __name__ == "__main__":
    main()
