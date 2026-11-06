import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score)

SEEDS = [0, 1, 2, 3, 4]
EPOCHS, BATCH, PATIENCE = 150, 16, 20

# ---------- data (GLOBAL VIEW ONLY) ----------
d = np.load("dataset.npz")
scale = (d["X_train"] - 1.0).std()

def prep(X):
    return torch.tensor((X - 1.0) / scale, dtype=torch.float32).unsqueeze(1)  # (N, 1, 201)

Xtr, Xva, Xte = prep(d["X_train"]), prep(d["X_val"]), prep(d["X_test"])
ytr = torch.tensor(d["y_train"], dtype=torch.float32)
yva = torch.tensor(d["y_val"], dtype=torch.float32)
yte = d["y_test"]

# ---------- model ----------
class TransitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)

def augment(xb):
    flip = torch.rand(xb.size(0)) < 0.5
    xb = torch.where(flip[:, None, None], xb.flip(-1), xb)
    xb = torch.roll(xb, shifts=int(torch.randint(-3, 4, (1,))), dims=-1)
    return xb + 0.05 * torch.randn_like(xb)

def run(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = TransitCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, wait = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train(); perm = torch.randperm(len(ytr)); tl = 0.0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(augment(Xtr[idx])), ytr[idx])
            loss.backward(); opt.step()
            tl += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva), yva).item()
        if epoch % 10 == 0:
            print(f"  seed {seed} epoch {epoch:3d} | train {tl/len(perm):.3f} | val {vl:.3f}")
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        p_val = torch.sigmoid(model(Xva)).numpy()
        p_test = torch.sigmoid(model(Xte)).numpy()
    thr = max(np.unique(p_val),
              key=lambda t: f1_score(d["y_val"], (p_val >= t).astype(int), zero_division=0))
    pred = (p_test >= thr).astype(int)
    return {
        "val_roc_auc": roc_auc_score(d["y_val"], p_val),   # for model selection vs dual-view
        "pr_auc": average_precision_score(yte, p_test),
        "roc_auc": roc_auc_score(yte, p_test),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
    }

runs = []
for s in SEEDS:
    print(f"Seed {s}")
    runs.append(run(s))
    print("  ->", {k: round(v, 3) for k, v in runs[-1].items()})

summary = {k: {"mean": float(np.mean([r[k] for r in runs])),
               "std": float(np.std([r[k] for r in runs]))} for k in runs[0]}
base = json.load(open("results/baseline_results.json"))
print("\n=== Test set (n=%d), mean ± std over %d seeds ===" % (len(yte), len(SEEDS)))
print("  val_roc_auc (model-selection): %.3f ± %.3f"
      % (summary["val_roc_auc"]["mean"], summary["val_roc_auc"]["std"]))
for k in ["pr_auc", "roc_auc", "precision", "recall", "f1"]:
    print(f"  {k:9s} CNN {summary[k]['mean']:.3f} ± {summary[k]['std']:.3f}   | baseline {base[k]:.3f}")
json.dump({"cnn": summary, "baseline": base, "seeds": SEEDS},
          open("results/cnn_results_without_local_view.json", "w"), indent=2)