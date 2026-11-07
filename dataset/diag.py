import traceback
import numpy as np, pandas as pd, lightkurve as lk

L_GLOBAL, L_LOCAL, LOCAL_WINDOWS = 201, 61, 3.0

koi = pd.read_csv("../koi_table.csv")
koi = koi[koi["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
koi["label"] = (koi["koi_disposition"] == "CONFIRMED").astype(int)
koi = koi.dropna(subset=["koi_period", "koi_time0bk", "koi_duration"])
koi = koi[koi["koi_period"] < 30]
koi = koi.drop_duplicates(subset="kepid", keep="first")
n = min(500, (koi.label == 1).sum(), (koi.label == 0).sum())
pos = koi[koi.label == 1].sample(n, random_state=42)
neg = koi[koi.label == 0].sample(n, random_state=42)
subset = pd.concat([pos, neg]).sample(frac=1, random_state=42).reset_index(drop=True)

row = subset.iloc[0]
print("Testing KIC", int(row.kepid), "| period", row.koi_period,
      "| t0", row.koi_time0bk, "| dur", row.koi_duration)
try:
    sr = lk.search_lightcurve(f"KIC {int(row.kepid)}", author="Kepler", exptime="long")
    print("1. search products:", len(sr))
    lcc = sr.download_all()
    print("2. downloaded quarters:", 0 if lcc is None else len(lcc))
    lc = lcc.stitch()
    print("3. stitched points:", len(lc.flux))
    flat = lc.remove_nans().flatten(window_length=401)
    print("4. flattened points:", len(flat.flux))
    t, f = flat.time.value, flat.flux.value
    phase = ((t - row.koi_time0bk) / row.koi_period) % 1.0
    phase[phase > 0.5] -= 1.0
    order = np.argsort(phase); phase, f = phase[order], f[order]
    m = np.isfinite(f); phase, f = phase[m], f[m]
    print("5. finite folded points:", len(f))
    glob = np.interp(np.linspace(-0.5, 0.5, L_GLOBAL), phase, f)
    print("6. global view: OK")
    frac_dur = (row.koi_duration / 24.0) / row.koi_period
    half = min(LOCAL_WINDOWS * frac_dur, 0.5)
    sel = np.abs(phase) < half
    print("7. local window half:", round(half, 4), "| points in window:", int(sel.sum()))
    loc = np.interp(np.linspace(-half, half, L_LOCAL), phase[sel], f[sel])
    print("8. local view: OK  ->  SUCCESS")
except Exception:
    print("--- FAILED, traceback below ---")
    traceback.print_exc()