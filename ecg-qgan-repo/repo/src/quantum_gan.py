"""
Hybrid Quantum-Classical GAN (QGAN) for ECG anomaly detection.

Design choice (matches the QGAN literature pattern, e.g. Hammami et al. 2025,
Bermot et al. 2023): a variational quantum circuit (VQC) replaces the
classical latent-encoding stage of the generator pipeline. The discriminator
and the decoding (reconstruction) stage remain classical, so the comparison
against Step 3's classical baseline isolates the effect of the quantum layer.

Architecture:
  Classical pre-encoder: R^187 -> R^n_qubits   (dimensionality reduction to qubit count)
  VQC (variational quantum circuit): n_qubits -> n_qubits expectation values
                                       (angle-encoding + entangling ansatz, trainable)
  Classical Generator (decoder): R^n_qubits -> R^187   (reconstruct beat)
  Classical Discriminator: R^187 -> [0,1]     (same architecture as Step 3, for fairness)

This keeps qubit count small (n_qubits = LATENT_DIM = 8) so it simulates fast
on CPU with no real quantum hardware needed (PennyLane default.qubit simulator).
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import pennylane as qml
import pennylane.numpy as pnp
from pennylane.optimize import AdamOptimizer
import os
import json
import time

OUT_DIR = _os.path.join(_ROOT, "processed")
RESULTS_DIR = _os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT_DIM = 187
N_QUBITS = 8          # = LATENT_DIM in classical_gan.py, for fair comparison
HIDDEN_DIM = 32
N_LAYERS = 2           # entangling layers in the VQC ansatz
rng = np.random.default_rng(42)

dev = qml.device("default.qubit", wires=N_QUBITS)


def relu(x):
    return pnp.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + pnp.exp(-pnp.clip(x, -30, 30)))


def tanh(x):
    return pnp.tanh(x)


def init_dense(rng, in_dim, out_dim, scale=0.1):
    W = pnp.array(rng.normal(0, scale, size=(in_dim, out_dim)), requires_grad=True)
    b = pnp.array(np.zeros(out_dim), requires_grad=True)
    return W, b


# ---------------------------------------------------------------------
# Variational Quantum Circuit (angle encoding + StronglyEntanglingLayers-style ansatz)
# ---------------------------------------------------------------------
@qml.qnode(dev, interface="autograd")
def vqc_circuit(inputs, weights):
    # inputs shape: (batch, N_QUBITS) -- PennyLane parameter broadcasting executes
    # the whole batch in one vectorized call instead of a per-sample Python loop.
    for i in range(N_QUBITS):
        qml.RY(inputs[:, i], wires=i)
    for layer in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


def vqc_batch(X, weights):
    """Batched VQC execution (parameter broadcasting) -- ~100x faster than a
    per-sample Python loop, since default.qubit vectorizes the whole batch."""
    outs = vqc_circuit(X, weights)          # list of N_QUBITS arrays, each shape (batch,)
    return pnp.stack(outs, axis=1)           # -> shape (batch, N_QUBITS)


class QuantumParams:
    def __init__(self, seed=42):
        r = np.random.default_rng(seed)
        # Classical pre-encoder: 187 -> N_QUBITS (just enough to map into rotation angles)
        self.PE_W1, self.PE_b1 = init_dense(r, INPUT_DIM, N_QUBITS)
        # VQC trainable weights: (layers, qubits, 2 rotation params)
        self.vqc_weights = pnp.array(
            r.uniform(0, 2 * np.pi, size=(N_LAYERS, N_QUBITS, 2)), requires_grad=True
        )
        # Classical Generator (decoder): N_QUBITS -> 32 -> 187
        self.G_W1, self.G_b1 = init_dense(r, N_QUBITS, HIDDEN_DIM)
        self.G_W2, self.G_b2 = init_dense(r, HIDDEN_DIM, INPUT_DIM)
        # Classical Discriminator: identical architecture to Step 3, for a fair comparison
        self.D_W1, self.D_b1 = init_dense(r, INPUT_DIM, HIDDEN_DIM)
        self.D_W2, self.D_b2 = init_dense(r, HIDDEN_DIM, 1)


def pre_encode(x, PE_W1, PE_b1):
    # scaled + bounded so angle-encoding into RY/RZ is sensible
    return pnp.pi * tanh(x @ PE_W1 + PE_b1)


def generate(z, G_W1, G_b1, G_W2, G_b2):
    h = relu(z @ G_W1 + G_b1)
    xh = tanh(h @ G_W2 + G_b2)
    return xh


def discriminate(x, D_W1, D_b1, D_W2, D_b2):
    h = relu(x @ D_W1 + D_b1)
    out = sigmoid(h @ D_W2 + D_b2)
    return out


def reconstruct(x, p):
    angles = pre_encode(x, p.PE_W1, p.PE_b1)
    z = vqc_batch(angles, p.vqc_weights)
    xh = generate(z, p.G_W1, p.G_b1, p.G_W2, p.G_b2)
    return xh


def bce(pred, target):
    eps = 1e-7
    pred = pnp.clip(pred, eps, 1 - eps)
    return -pnp.mean(target * pnp.log(pred) + (1 - target) * pnp.log(1 - pred))


def ae_loss_fn(PE_W1, PE_b1, vqc_weights, G_W1, G_b1, G_W2, G_b2, x_batch, p, lambda_adv=0.1):
    angles = pre_encode(x_batch, PE_W1, PE_b1)
    z = vqc_batch(angles, vqc_weights)
    xh = generate(z, G_W1, G_b1, G_W2, G_b2)
    recon_loss = pnp.mean((x_batch - xh) ** 2)
    d_out = discriminate(xh, p.D_W1, p.D_b1, p.D_W2, p.D_b2)
    adv_loss = bce(d_out, pnp.ones((x_batch.shape[0], 1)))
    return recon_loss + lambda_adv * adv_loss


def d_loss_fn(D_W1, D_b1, D_W2, D_b2, x_batch, xh_batch):
    d_real = discriminate(x_batch, D_W1, D_b1, D_W2, D_b2)
    d_fake = discriminate(xh_batch, D_W1, D_b1, D_W2, D_b2)
    loss_real = bce(d_real, pnp.ones((x_batch.shape[0], 1)))
    loss_fake = bce(d_fake, pnp.zeros((xh_batch.shape[0], 1)))
    return 0.5 * (loss_real + loss_fake)


def count_params(p):
    total = 0
    for arr in [p.PE_W1, p.PE_b1, p.vqc_weights, p.G_W1, p.G_b1, p.G_W2, p.G_b2,
                p.D_W1, p.D_b1, p.D_W2, p.D_b2]:
        total += np.array(arr).size
    return total


def train(train_tag="10pct", epochs=5, batch_size=32, lr=0.01, max_batches_per_epoch=40):
    print(f"\n=== Training QGAN (VQC encoder) on train_normal_{train_tag}.npy ===")
    X = np.load(os.path.join(OUT_DIR, f"train_normal_{train_tag}.npy"))
    n = X.shape[0]
    p = QuantumParams(seed=42)
    print(f"Total QGAN trainable parameters: {count_params(p)}  (n_qubits={N_QUBITS})")

    opt_ae = AdamOptimizer(lr)
    opt_d = AdamOptimizer(lr)

    history = []
    t0 = time.time()
    for epoch in range(epochs):
        perm = rng.permutation(n)
        n_batches = min(max_batches_per_epoch, n // batch_size)
        epoch_ae_loss, epoch_d_loss = 0.0, 0.0
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            x_batch = pnp.array(X[idx])

            xh_batch = reconstruct(x_batch, p)
            xh_batch = pnp.array(np.array(xh_batch))  # detach

            (new_dW1, new_db1, new_dW2, new_db2), d_loss = opt_d.step_and_cost(
                lambda W1, b1, W2, b2: d_loss_fn(W1, b1, W2, b2, x_batch, xh_batch),
                p.D_W1, p.D_b1, p.D_W2, p.D_b2
            )
            p.D_W1, p.D_b1, p.D_W2, p.D_b2 = new_dW1, new_db1, new_dW2, new_db2

            new_ae, ae_loss = opt_ae.step_and_cost(
                lambda pe_w1, pe_b1, vqcw, gw1, gb1, gw2, gb2: ae_loss_fn(
                    pe_w1, pe_b1, vqcw, gw1, gb1, gw2, gb2, x_batch, p),
                p.PE_W1, p.PE_b1, p.vqc_weights, p.G_W1, p.G_b1, p.G_W2, p.G_b2
            )
            (p.PE_W1, p.PE_b1, p.vqc_weights, p.G_W1, p.G_b1, p.G_W2, p.G_b2) = new_ae

            epoch_ae_loss += float(ae_loss)
            epoch_d_loss += float(d_loss)

        epoch_ae_loss /= n_batches
        epoch_d_loss /= n_batches
        history.append({"epoch": epoch, "ae_loss": epoch_ae_loss, "d_loss": epoch_d_loss})
        print(f"  epoch {epoch+1}/{epochs}  ae_loss={epoch_ae_loss:.5f}  d_loss={epoch_d_loss:.5f}  "
              f"elapsed={time.time()-t0:.1f}s")

    save_path = os.path.join(RESULTS_DIR, f"quantum_params_{train_tag}.npz")
    np.savez(save_path,
              PE_W1=np.array(p.PE_W1), PE_b1=np.array(p.PE_b1),
              vqc_weights=np.array(p.vqc_weights),
              G_W1=np.array(p.G_W1), G_b1=np.array(p.G_b1),
              G_W2=np.array(p.G_W2), G_b2=np.array(p.G_b2),
              D_W1=np.array(p.D_W1), D_b1=np.array(p.D_b1),
              D_W2=np.array(p.D_W2), D_b2=np.array(p.D_b2))
    with open(os.path.join(RESULTS_DIR, f"quantum_history_{train_tag}.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved trained params -> {save_path}")
    return p, history


if __name__ == "__main__":
    train(train_tag="10pct", epochs=3, batch_size=32, max_batches_per_epoch=15)
