import json
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score)

d = np.load("../dataset.npz")
X_val, y_val = d["X_val"], d["y_val"]
X_test, y_test = d["X_test"], d["y_test"]

L = X_test.shape[1]
CENTER = L // 2                      # index 100: where the transit should sit
WIDTHS = [5, 9, 15, 25, 41]          # box widths (phase bins), odd => symmetric

def box_score(row):
    """Best box-dip signal-to-noise over several widths."""
    best = 0.0
    for w in WIDTHS:
        lo, hi = CENTER - w // 2, CENTER + w // 2 + 1
        inside = row[lo:hi]
        outside = np.concatenate([row[:lo], row[hi:]])
        base = np.median(outside)
        sigma = 1.4826 * np.median(np.abs(outside - base))   # robust noise estimate
        if sigma == 0:
            continue
        depth = base - inside.mean()                          # how far the dip drops
        snr = depth / (sigma / np.sqrt(w))                    # dip size vs noise
        best = max(best, snr)
    return best

s_val = np.array([box_score(r) for r in X_val])
s_test = np.array([box_score(r) for r in X_test])

# Pick the cutoff on VALIDATION only: the one with the best F1
best_thr, best_f1 = None, -1
for thr in np.unique(s_val):
    f1 = f1_score(y_val, (s_val >= thr).astype(int), zero_division=0)
    if f1 > best_f1:
        best_thr, best_f1 = thr, f1

pred = (s_test >= best_thr).astype(int)
results = {
    "threshold": float(best_thr),
    "precision": float(precision_score(y_test, pred, zero_division=0)),
    "recall": float(recall_score(y_test, pred, zero_division=0)),
    "f1": float(f1_score(y_test, pred, zero_division=0)),
    "pr_auc": float(average_precision_score(y_test, s_test)),
    "roc_auc": float(roc_auc_score(y_test, s_test)),
    "random_pr_auc": float(y_test.mean()),     # what guessing would score
    "n_test": int(len(y_test)),
}
print(json.dumps(results, indent=2))
with open("../results/baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)