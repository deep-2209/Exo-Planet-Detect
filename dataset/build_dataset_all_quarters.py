import os
import time
import warnings
import socket
import numpy as np
import pandas as pd
import lightkurve as lk
from sklearn.model_selection import train_test_split
from astropy.utils.exceptions import AstropyWarning

socket.setdefaulttimeout(120)
warnings.simplefilter("ignore", category=AstropyWarning)

# ----- config -----
N_PER_CLASS = 100        # all-quarters is slow (~1-2 min/star serial); keep small. No resume now.
L_GLOBAL = 201
L_LOCAL = 61
LOCAL_WINDOWS = 3.0

# 1. NASA label table (small snapshot; unrelated to the light-curve downloads)
TAP = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
       "select+kepid,koi_disposition,koi_period,koi_time0bk,koi_duration+"
       "from+cumulative&format=csv")
KOI_CACHE = "../koi_table.csv"     # you're running from the dataset/ subfolder

def load_koi():
    if os.path.exists(KOI_CACHE):
        print(f"Loading KOI table from {KOI_CACHE}")
        return pd.read_csv(KOI_CACHE)
    return pd.read_csv(TAP)

koi = load_koi()

# 2. Binary labels; drop CANDIDATE; need period + epoch + duration
koi = koi[koi["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
koi["label"] = (koi["koi_disposition"] == "CONFIRMED").astype(int)
koi = koi.dropna(subset=["koi_period", "koi_time0bk", "koi_duration"])
koi = koi[koi["koi_period"] < 30]
koi = koi.drop_duplicates(subset="kepid", keep="first")

# 3. Balanced subset
pos_pool, neg_pool = koi[koi.label == 1], koi[koi.label == 0]
n = min(N_PER_CLASS, len(pos_pool), len(neg_pool))
print(f"Available -> confirmed: {len(pos_pool)}, false pos: {len(neg_pool)} | using {n}/class")
pos = pos_pool.sample(n, random_state=42)
neg = neg_pool.sample(n, random_state=42)
subset = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

# 4. One star -> global + local view, folding ALL stitched quarters
def make_views(kepid, period, t0, duration_hours):
    sr = lk.search_lightcurve(f"KIC {int(kepid)}", author="Kepler", exptime="long")
    if len(sr) == 0:
        return None
    lcc = sr.download_all()
    if lcc is None or len(lcc) == 0:
        return None
    lc = lcc.stitch()
    flat = lc.remove_nans().flatten(window_length=401)
    t, f = flat.time.value, flat.flux.value
    phase = ((t - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    order = np.argsort(phase); phase, f = phase[order], f[order]
    m = np.isfinite(f); phase, f = phase[m], f[m]
    if len(f) < L_GLOBAL:
        return None
    glob = np.interp(np.linspace(-0.5, 0.5, L_GLOBAL), phase, f)
    frac_dur = (duration_hours / 24.0) / period
    half = min(LOCAL_WINDOWS * frac_dur, 0.5)
    sel = np.abs(phase) < half
    if sel.sum() < 5:
        return None
    loc = np.interp(np.linspace(-half, half, L_LOCAL), phase[sel], f[sel])
    return glob, loc

# 5. SERIAL, no checkpoint — fresh each run, results kept in memory; errors printed live
rows = [row for _, row in subset.iterrows()]
Xg, Xl, y, meta = [], [], [], []
start = time.time()
for i, row in enumerate(rows):
    try:
        out = make_views(row.kepid, row.koi_period, row.koi_time0bk, row.koi_duration)
        if out is not None:
            g, l = out
            Xg.append(g); Xl.append(l); y.append(int(row.label))
            meta.append([row.kepid, row.koi_period, row.koi_time0bk])
    except Exception as e:
        print(f"\nskip KIC {int(row.kepid)}: {type(e).__name__}: {e}")
    done = i + 1
    elapsed = time.time() - start
    rate = done / elapsed if elapsed > 0 else 0
    eta = (len(rows) - done) / rate / 60 if rate > 0 else 0
    print(f"\r{done}/{len(rows)}  kept {len(Xg)}  {rate:.2f} star/s  ETA {eta:.1f} min    ",
          end="", flush=True)
print()

Xg, Xl, y, meta = np.array(Xg), np.array(Xl), np.array(y), np.array(meta)
print("Global:", Xg.shape, "Local:", Xl.shape,
      "| positives:", int(y.sum()), "negatives:", int((y == 0).sum()))

# 6. Stratified 70/15/15 split on indices
idx = np.arange(len(y))
i_tmp, i_test = train_test_split(idx, test_size=0.15, stratify=y, random_state=42)
i_train, i_val = train_test_split(i_tmp, test_size=0.1765, stratify=y[i_tmp], random_state=42)

np.savez("../dataset_all_quarters.npz",
         X_train=Xg[i_train], Xl_train=Xl[i_train], y_train=y[i_train], meta_train=meta[i_train],
         X_val=Xg[i_val],     Xl_val=Xl[i_val],     y_val=y[i_val],     meta_val=meta[i_val],
         X_test=Xg[i_test],   Xl_test=Xl[i_test],   y_test=y[i_test],   meta_test=meta[i_test])
print("Saved dataset_all_quarters.npz | train/val/test:", len(i_train), len(i_val), len(i_test))