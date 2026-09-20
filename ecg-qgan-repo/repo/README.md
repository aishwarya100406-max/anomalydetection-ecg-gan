# A Hybrid Quantum-Classical GAN for Anomaly Detection in ECG Sensor Data

Code and results for the paper *"A Hybrid Quantum-Classical GAN for Anomaly Detection in ECG Sensor Data."*

An 8-qubit variational quantum circuit (VQC) replaces the classical latent encoder inside an adversarially-regularized autoencoder. Every other component — decoder, discriminator, training objective, optimizer, random seed — is held identical to a classical baseline, so any performance difference is attributable to the quantum substitution alone.

All quantum computation runs on a classical state-vector simulator (PennyLane `default.qubit`). **No quantum hardware is required.**

---

## Key results

| Metric | Classical GAN | Hybrid QGAN |
|---|---|---|
| Total trainable parameters | 18,788 | **14,044** (−25.3%) |
| Generator parameters | 12,739 | **7,995** (−37.2%) |
| MIT-BIH F1 (100% data) | 0.512 | **0.530** |
| MIT-BIH ROC-AUC (10% data) | 0.824 | **0.831** |
| PTBDB cross-dataset ROC-AUC (mean) | 0.550 | **0.567** |
| PTBDB ROC-AUC spread across regimes | 0.044 | **0.013** |

The hybrid model achieves comparable same-dataset performance and higher cross-dataset ROC-AUC at three of four training regimes, using substantially fewer parameters. It does **not** achieve categorical superiority; see the Limitations section of the paper.

---

## Repository structure

```
src/
  preprocess.py        # builds normal-only training sets + test partitions
  classical_gan.py     # classical adversarial autoencoder baseline
  quantum_gan.py       # hybrid QGAN (VQC latent encoder)
  run_one.py           # trains + evaluates one (regime, model) pair
scripts/
  make_framework_fig.py       # Fig. 1 framework diagram
  make_full_architecture.py   # Fig. 2 full architecture diagram
results/
  all_results.json     # all metrics, 4 regimes x 2 models x 2 test sets
  *_history_*.json     # per-epoch training loss curves
  *_params_*.npz       # trained model weights
figures/               # publication figures
```

---

## Data

Not included in this repository (size + licensing). Download the preprocessed
beat-segmented CSVs from Kaggle:

> https://www.kaggle.com/datasets/shayanfazeli/heartbeat

Place `mitbih_train.csv`, `mitbih_test.csv`, `ptbdb_normal.csv`, and
`ptbdb_abnormal.csv` in a `data/` directory at the repository root.

Original sources:
- MIT-BIH Arrhythmia Database (PhysioNet)
- PTB Diagnostic ECG Database (PhysioNet)
- Preprocessing pipeline: Kachuee, Fazeli & Sarrafzadeh, *ECG Heartbeat Classification: A Deep Transferable Representation*, IEEE ICHI 2018.

---

## Reproducing the results

```bash
pip install -r requirements.txt

# 1. Preprocess (writes .npy arrays to processed/)
python src/preprocess.py

# 2. Train + evaluate each configuration
for frac in 10pct 25pct 50pct 100pct; do
    python src/run_one.py $frac classical
    python src/run_one.py $frac quantum
done

# Results are merged into results/all_results.json
```

Runtime on CPU: ~15 s per classical run, ~200 s per quantum run.
Random seed is fixed at 42 throughout.

---

## Method summary

Both models train **only on normal beats**. A beat is flagged anomalous when its
reconstruction error exceeds a threshold τ set at the 95th percentile of the
reconstruction-error distribution over normal *training* beats — no anomaly
labels are used in threshold selection.

For the cross-dataset evaluation, τ is carried over from MIT-BIH to PTBDB
**without recalibration**, so the transfer is genuinely zero-shot. This is
methodologically conservative and is why cross-dataset F1 is low while ROC-AUC
remains informative.

The VQC applies RY angle encoding, then two entangling layers (trainable RY/RZ
rotations per qubit + a CNOT ring), and returns eight Pauli-Z expectation values
as the latent code. Ring topology was chosen for compatibility with
linear-nearest-neighbour hardware connectivity.

---

## Citation

```bibtex
@article{subramanian2026qganecg,
  title   = {A Hybrid Quantum-Classical GAN for Anomaly Detection in ECG Sensor Data},
  author  = {Kavitha D. and Sharma, Harsh and Subramanian, Aishwarya},
  journal = {(under review)},
  year    = {2026}
}
```

---

## License

MIT — see `LICENSE`.
