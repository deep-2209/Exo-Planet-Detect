import numpy as np
import matplotlib.pyplot as plt

d = np.load("dataset.npz")
print("Arrays:", d.files)
for k in d.files:
    print(f"  {k}: shape={d[k].shape}, dtype={d[k].dtype}")

X, y = d["X_train"], d["y_train"]
print("\nTrain class counts [FP, planet]:", np.bincount(y))
print("First row, 10 of 201 values:", np.round(X[0][:10], 4))
print("Its label:", y[0], "(1=planet, 0=false positive)")

# See a planet vs a false positive, side by side
p = np.where(y == 1)[0][0]   # first planet
n = np.where(y == 0)[0][0]   # first false positive
plt.figure(figsize=(8, 4))
plt.plot(X[p], label="CONFIRMED (planet)")
plt.plot(X[n], label="FALSE POSITIVE")
plt.axvline(100, color="gray", ls="--", lw=0.8)   # transit center
plt.xlabel("phase bin (0–200, center = transit)"); plt.ylabel("normalized flux")
plt.legend(); plt.tight_layout(); plt.savefig("dataset_rows.png", dpi=150)
print("Saved dataset_rows.png")