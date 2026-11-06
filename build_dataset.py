import os
import time
import warnings
import numpy as np
import pandas as pd
import lightkurve as lk
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor, as_completed
from astropy.utils.exceptions import AstropyWarning
import socket
socket.setdefaulttimeout(60)

warnings.simplefilter("ignore", category=AstropyWarning)   # silence harmless FITS-unit noise

# ----- config -----
N_PER_CLASS = 1611       
L_GLOBAL = 201           # whole-orbit view length
L_LOCAL = 61             # zoomed transit view length
LOCAL_WINDOWS = 3.0      # local view spans ± this many transit durations
MAX_WORKERS = 1          # parallel downloads; drop to 4 if MAST throttles
PER_STAR_TIMEOUT = 60   # abandon any star whose download hangs longer than this

# 1. NASA label table (KOI cumulative) via TAP, as CSV
TAP = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
       "select+kepid,koi_disposition,koi_period,koi_time0bk,koi_duration+"
       "from+cumulative&format=csv")

KOI_CACHE = "koi_table.csv"

def load_koi():
    if os.path.exists(KOI_CACHE):                 # use the saved snapshot if we have it
        print(f"Loading KOI table from cache: {KOI_CACHE}")
        return pd.read_csv(KOI_CACHE)
    last_err = None
    for attempt in range(1, 6):                   # retry the flaky NASA endpoint
        try:
            df = pd.read_csv(TAP)
            df.to_csv(KOI_CACHE, index=False)     # save for all future runs
            print(f"Fetched KOI table from NASA and cached to {KOI_CACHE}")
            return df
        except Exception as e:
            last_err = e
            print(f"  TAP fetch attempt {attempt}/5 failed ({type(e).__name__}); retrying in 5s...")
            time.sleep(5)
    raise RuntimeError(f"Could not fetch KOI table after 5 attempts: {last_err}")

koi = load_koi()

# 2. Binary labels; drop CANDIDATE; need period + epoch + duration to fold/zoom
koi = koi[koi["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
koi["label"] = (koi["koi_disposition"] == "CONFIRMED").astype(int)
koi = koi.dropna(subset=["koi_period", "koi_time0bk", "koi_duration"])
koi = koi[koi["koi_period"] < 30]                        # >=3 transits per quarter
koi = koi.drop_duplicates(subset="kepid", keep="first")  # one KOI per star -> no leakage

# 3. Balanced subset, clamped to whatever is actually available
pos_pool, neg_pool = koi[koi.label == 1], koi[koi.label == 0]
n = min(N_PER_CLASS, len(pos_pool), len(neg_pool))
print(f"Available -> confirmed: {len(pos_pool)}, false pos: {len(neg_pool)} | using {n}/class")
pos = pos_pool.sample(n, random_state=42)
neg = neg_pool.sample(n, random_state=42)
subset = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

# 4. One star -> fixed-length global view (whole orbit) + local view (zoomed transit)
def make_views(kepid, period, t0, duration_hours):
    sr = lk.search_lightcurve(f"KIC {int(kepid)}", author="Kepler", exptime="long")
    if len(sr) == 0:
        return None
    flat = sr[0].download().remove_nans().normalize().flatten(window_length=401)
    t, f = flat.time.value, flat.flux.value
    phase = ((t - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0                 # center transit at phase 0, range [-0.5, 0.5]
    order = np.argsort(phase); phase, f = phase[order], f[order]
    m = np.isfinite(f); phase, f = phase[m], f[m]
    if len(f) < L_GLOBAL:
        return None
    glob = np.interp(np.linspace(-0.5, 0.5, L_GLOBAL), phase, f)
    frac_dur = (duration_hours / 24.0) / period            # transit width in phase units
    half = min(LOCAL_WINDOWS * frac_dur, 0.5)
    sel = np.abs(phase) < half
    if sel.sum() < 5:                                      # too few in-transit points
        return None
    loc = np.interp(np.linspace(-half, half, L_LOCAL), phase[sel], f[sel])
    return glob, loc

# 5. Parallel download + process (network-bound -> threads), deterministic assembly
def process_row(row):
    last = None
    for attempt in range(2):                      # one retry on failure
        try:
            out = make_views(row.kepid, row.koi_period, row.koi_time0bk, row.koi_duration)
            if out is None:
                return ("skip", int(row.kepid))
            g, l = out
            return ("ok", g, l, int(row.label), [row.kepid, row.koi_period, row.koi_time0bk])
        except Exception as e:
            last = e
            time.sleep(1)
    return ("err", int(row.kepid), str(last))

rows = [row for _, row in subset.iterrows()]
results = [None] * len(rows)
done = kept = 0
start = time.time()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    fut_list = [ex.submit(process_row, rows[i]) for i in range(len(rows))]
    for i, fut in enumerate(fut_list):
        try:
            res = fut.result(timeout=PER_STAR_TIMEOUT)   # <-- hard cap; abandons a hung download
        except Exception as e:
            res = ("err", int(rows[i].kepid), f"timeout:{type(e).__name__}")
        results[i] = res
        done += 1
        if res[0] == "ok":
            kept += 1
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(rows) - done) / rate / 60 if rate > 0 else 0
        print(f"\r{done}/{len(rows)}  kept {kept}  {rate:.1f} star/s  ETA {eta:.1f} min    ",
              end="", flush=True)
print()

# assemble in ORIGINAL order (not completion order) so the split stays reproducible
Xg, Xl, y, meta = [], [], [], []
for res in results:
    if res[0] == "ok":
        _, g, l, lab, m = res
        Xg.append(g); Xl.append(l); y.append(lab); meta.append(m)
    elif res[0] == "err":
        print(f"skip KIC {res[1]}: {res[2]}")

Xg, Xl, y, meta = np.array(Xg), np.array(Xl), np.array(y), np.array(meta)
print("Global:", Xg.shape, "Local:", Xl.shape,
      "| positives:", int(y.sum()), "negatives:", int((y == 0).sum()))

# 6. Stratified 70/15/15 split on INDICES so global/local/meta stay aligned
idx = np.arange(len(y))
i_tmp, i_test = train_test_split(idx, test_size=0.15, stratify=y, random_state=42)
i_train, i_val = train_test_split(i_tmp, test_size=0.1765, stratify=y[i_tmp], random_state=42)

# global view keeps the name X_* (so baseline_bls.py runs unchanged); local view is Xl_*
np.savez("dataset_test.npz",
         X_train=Xg[i_train], Xl_train=Xl[i_train], y_train=y[i_train], meta_train=meta[i_train],
         X_val=Xg[i_val],     Xl_val=Xl[i_val],     y_val=y[i_val],     meta_val=meta[i_val],
         X_test=Xg[i_test],   Xl_test=Xl[i_test],   y_test=y[i_test],   meta_test=meta[i_test])
print("Saved dataset_test.npz | train/val/test:", len(i_train), len(i_val), len(i_test))