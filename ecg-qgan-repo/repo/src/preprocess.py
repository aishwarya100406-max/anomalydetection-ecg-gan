"""
Preprocessing pipeline for hybrid quantum-classical GAN ECG anomaly detection.

Strategy (standard for GAN-based anomaly detection, e.g. BeatGAN / QGAN literature):
  - Train the generator/discriminator ONLY on NORMAL beats (label 0 in MIT-BIH).
  - At test time, anomalies are beats the model reconstructs poorly.
  - Test sets:
      (a) MIT-BIH test split (normal + 4 arrhythmia classes -> binarized normal/abnormal)
      (b) PTBDB normal + abnormal (cross-dataset generalization test)
  - Reduced-data regime: 10% / 25% / 50% / 100% of normal training beats,
    for the "data efficiency" comparison between classical GAN and QGAN.

Each ECG beat is a 187-point time series (already resampled/padded by the
original Kaggle preprocessing of MIT-BIH / PTBDB).
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import os

DATA_DIR = _os.path.join(_ROOT, "data")
OUT_DIR = _os.path.join(_ROOT, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)


def load(path):
    df = pd.read_csv(path, header=None)
    X = df.iloc[:, :-1].values.astype(np.float32) if df.shape[1] == 188 else df.values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(int) if df.shape[1] == 188 else None
    return X, y


def main():
    print("Loading MIT-BIH...")
    Xtr, ytr = load(os.path.join(DATA_DIR, "mitbih_train.csv"))
    Xte, yte = load(os.path.join(DATA_DIR, "mitbih_test.csv"))

    print("Loading PTBDB...")
    Xptb_n, _ = load(os.path.join(DATA_DIR, "ptbdb_normal.csv"))
    Xptb_a, _ = load(os.path.join(DATA_DIR, "ptbdb_abnormal.csv"))

    # ---- Training set: NORMAL beats only, from MIT-BIH train split ----
    normal_mask = ytr == 0
    X_normal = Xtr[normal_mask]
    rng.shuffle(X_normal)
    print(f"MIT-BIH normal training beats available: {X_normal.shape[0]}")

    # Reduced-data regime subsets (for data-efficiency comparison)
    fractions = {"10pct": 0.10, "25pct": 0.25, "50pct": 0.50, "100pct": 1.00}
    for tag, frac in fractions.items():
        n = int(len(X_normal) * frac)
        np.save(os.path.join(OUT_DIR, f"train_normal_{tag}.npy"), X_normal[:n])
        print(f"  saved train_normal_{tag}.npy -> {n} beats")

    # ---- Test set A: MIT-BIH test split, binarized (0=normal, 1=anomaly) ----
    y_bin_mitbih = (yte != 0).astype(int)
    np.save(os.path.join(OUT_DIR, "test_mitbih_X.npy"), Xte)
    np.save(os.path.join(OUT_DIR, "test_mitbih_y.npy"), y_bin_mitbih)
    print(f"MIT-BIH test: {Xte.shape[0]} beats "
          f"({(y_bin_mitbih==0).sum()} normal / {(y_bin_mitbih==1).sum()} anomaly)")

    # ---- Test set B: PTBDB, cross-dataset generalization ----
    X_ptb = np.concatenate([Xptb_n, Xptb_a], axis=0)
    y_ptb = np.concatenate([np.zeros(len(Xptb_n)), np.ones(len(Xptb_a))]).astype(int)
    perm = rng.permutation(len(X_ptb))
    X_ptb, y_ptb = X_ptb[perm], y_ptb[perm]
    np.save(os.path.join(OUT_DIR, "test_ptbdb_X.npy"), X_ptb)
    np.save(os.path.join(OUT_DIR, "test_ptbdb_y.npy"), y_ptb)
    print(f"PTBDB (cross-dataset test): {X_ptb.shape[0]} beats "
          f"({(y_ptb==0).sum()} normal / {(y_ptb==1).sum()} anomaly)")

    print("\nAll preprocessed arrays saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
