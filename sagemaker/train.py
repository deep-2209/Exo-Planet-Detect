import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, f1_score)

p = argparse.ArgumentParser()
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--batch", type=int, default=16)
p.add_argument("--patience", type=int, default=20)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--weight_decay", type=float, default=1e-3)
p.add_argument("--seeds", type=int, default=5)
# SageMaker-provided paths, with local fallbacks so it also runs on your machine
p.add_argument("--train_dir",  default=os.environ.get("SM_CHANNEL_TRAINING", "."))
p.add_argument("--model_dir",  default=os.environ.get("SM_MODEL_DIR", "."))
p.add_argument("--output_dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "."))
args = p.parse_args()

d = np.load(os.path.join(args.train_dir, "dataset.npz"))
scale = (d["X_train"] - 1.0).std()
def prep(X):
    return torch.tensor((X - 1.0) / scale, dtype=torch.float32).unsqueeze(1)
Xtr, Xva, Xte = prep(d["X_train"]), prep(d["X_val"]), prep(d["X_test"])
ytr = torch.tensor(d["y_train"], dtype=torch.float32)
yva = torch.tensor(d["y_val"], dtype=torch.float32)
yte = d["y_test"]

class TransitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
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
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, wait = float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train(); perm = torch.randperm(len(ytr))
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(augment(Xtr[idx])), ytr[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva), yva).item()
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= args.patience:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        p_val = torch.sigmoid(model(Xva)).numpy()
        p_test = torch.sigmoid(model(Xte)).numpy()
    thr = max(np.unique(p_val),
              key=lambda t: f1_score(d["y_val"], (p_val >= t).astype(int), zero_division=0))
    pred = (p_test >= thr).astype(int)
    metrics = {
        "val_roc_auc": roc_auc_score(d["y_val"], p_val),
        "pr_auc": average_precision_score(yte, p_test),
        "roc_auc": roc_auc_score(yte, p_test),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "f1": f1_score(yte, pred, zero_division=0),
    }
    return metrics, model

runs, first_model = [], None
for s in range(args.seeds):
    m, model = run(s)
    runs.append(m)
    first_model = first_model or model
    print(f"seed {s}:", {k: round(v, 3) for k, v in m.items()})

summary = {k: {"mean": float(np.mean([r[k] for r in runs])),
               "std": float(np.std([r[k] for r in runs]))} for k in runs[0]}
print("SUMMARY:", json.dumps(summary, indent=2))

# Persist outputs: SM_MODEL_DIR -> model.tar.gz -> S3; output_dir -> S3 too
torch.save(first_model.state_dict(), os.path.join(args.model_dir, "model.pth"))
with open(os.path.join(args.output_dir, "cnn_results.json"), "w") as f:
    json.dump({"cnn": summary, "seeds": args.seeds}, f, indent=2)
print("Saved model.pth and cnn_results.json")