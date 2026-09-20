"""
Resumable single-job experiment runner.
Usage: python3 run_one.py <fraction> <classical|quantum>
Writes/merges its result into results/all_results.json
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import pennylane.numpy as pnp
import json
import os
import sys
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import classical_gan as cg
import quantum_gan as qg

PROC_DIR = _os.path.join(_ROOT, "processed")
RESULTS_DIR = _os.path.join(_ROOT, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "all_results.json")
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate(recon_fn, params, threshold, X, y):
    xh = np.array(recon_fn(pnp.array(X), params))
    err = np.mean((X - xh) ** 2, axis=1)
    preds = (err > threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0),
        "roc_auc": roc_auc_score(y, err),
        "mean_recon_error": float(err.mean()),
    }


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def already_done(results, frac, model):
    return frac in results and model in results[frac]


def run_classical(frac, Xte_mit, yte_mit, Xte_ptb, yte_ptb):
    Xtr = np.load(os.path.join(PROC_DIR, f"train_normal_{frac}.npy"))
    t0 = time.time()
    p_cls, hist_cls = cg.train(train_tag=frac, epochs=15, batch_size=64,
                                 lr=0.01, max_batches_per_epoch=150)
    cls_time = time.time() - t0
    cls_params = cg.count_params(p_cls)

    xh_train = np.array(cg.reconstruct(pnp.array(Xtr), p_cls))
    train_err = np.mean((Xtr - xh_train) ** 2, axis=1)
    cls_threshold = np.percentile(train_err, 95)

    cls_mit = evaluate(cg.reconstruct, p_cls, cls_threshold, Xte_mit, yte_mit)
    cls_ptb = evaluate(cg.reconstruct, p_cls, cls_threshold, Xte_ptb, yte_ptb)

    return {
        "n_params": int(cls_params), "train_time_sec": round(cls_time, 1),
        "threshold": float(cls_threshold),
        "mitbih_test": cls_mit, "ptbdb_test": cls_ptb,
    }


def run_quantum(frac, Xte_mit, yte_mit, Xte_ptb, yte_ptb):
    Xtr = np.load(os.path.join(PROC_DIR, f"train_normal_{frac}.npy"))
    t0 = time.time()
    p_q, hist_q = qg.train(train_tag=frac, epochs=25, batch_size=128,
                             lr=0.01, max_batches_per_epoch=45)
    q_time = time.time() - t0
    q_params = qg.count_params(p_q)

    # Now fast enough (batched VQC) to use the FULL training set for threshold calc
    xh_train_q = np.array(qg.reconstruct(pnp.array(Xtr), p_q))
    train_err_q = np.mean((Xtr - xh_train_q) ** 2, axis=1)
    q_threshold = np.percentile(train_err_q, 95)

    # Full test sets, no subsampling needed anymore
    q_mit = evaluate(qg.reconstruct, p_q, q_threshold, Xte_mit, yte_mit)
    q_ptb = evaluate(qg.reconstruct, p_q, q_threshold, Xte_ptb, yte_ptb)

    return {
        "n_params": int(q_params), "train_time_sec": round(q_time, 1),
        "threshold": float(q_threshold),
        "mitbih_test": q_mit, "ptbdb_test": q_ptb,
    }


if __name__ == "__main__":
    frac = sys.argv[1]
    model = sys.argv[2]
    force = len(sys.argv) > 3 and sys.argv[3] == "force"
    assert model in ("classical", "quantum")

    results = load_results()
    if already_done(results, frac, model) and not force:
        print(f"[{frac}][{model}] already done, skipping.")
        sys.exit(0)

    Xte_mit = np.load(os.path.join(PROC_DIR, "test_mitbih_X.npy"))
    yte_mit = np.load(os.path.join(PROC_DIR, "test_mitbih_y.npy"))
    Xte_ptb = np.load(os.path.join(PROC_DIR, "test_ptbdb_X.npy"))
    yte_ptb = np.load(os.path.join(PROC_DIR, "test_ptbdb_y.npy"))

    Xtr = np.load(os.path.join(PROC_DIR, f"train_normal_{frac}.npy"))

    if frac not in results:
        results[frac] = {"n_train_beats": int(len(Xtr))}

    if model == "classical":
        res = run_classical(frac, Xte_mit, yte_mit, Xte_ptb, yte_ptb)
    else:
        res = run_quantum(frac, Xte_mit, yte_mit, Xte_ptb, yte_ptb)

    results[frac][model] = res
    save_results(results)

    print(f"\n[{frac}][{model}] DONE: n_params={res['n_params']} "
          f"MIT-BIH F1={res['mitbih_test']['f1']:.3f} AUC={res['mitbih_test']['roc_auc']:.3f} | "
          f"PTBDB F1={res['ptbdb_test']['f1']:.3f} AUC={res['ptbdb_test']['roc_auc']:.3f}")
