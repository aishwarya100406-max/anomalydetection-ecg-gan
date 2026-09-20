"""
Classical GAN baseline for ECG anomaly detection (BeatGAN-style adversarial
autoencoder), implemented with autograd-numpy so its optimizer, precision,
and training loop are directly comparable to the hybrid QGAN in Step 4.

Architecture:
  Encoder E:     R^187 -> R^latent_dim          (compress beat to latent code)
  Generator G:   R^latent_dim -> R^187           (reconstruct beat from code)
  Discriminator D: R^187 -> [0,1]                (real beat vs reconstructed beat)

Training (adversarial autoencoder / AnoGAN-family):
  - G (=E+G, the autoencoder) minimizes: reconstruction loss + adversarial loss
    (fool D into thinking the reconstruction is real)
  - D maximizes: correctly distinguish real beats from reconstructions

Anomaly score at test time = reconstruction error ||x - G(E(x))||^2
  (standard scoring rule used in BeatGAN / QGAN anomaly-detection papers)
"""
import os as _os
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

import numpy as np
import pennylane.numpy as pnp
from pennylane.optimize import AdamOptimizer
import os
import json
import time

OUT_DIR = _os.path.join(_ROOT, "processed")
RESULTS_DIR = _os.path.join(_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT_DIM = 187
LATENT_DIM = 8
HIDDEN_DIM = 32
rng = np.random.default_rng(42)


# ---------------------------------------------------------------------
# Simple dense-layer building blocks (manual, autograd-differentiable)
# ---------------------------------------------------------------------
def init_dense(rng, in_dim, out_dim, scale=0.1):
    W = pnp.array(rng.normal(0, scale, size=(in_dim, out_dim)), requires_grad=True)
    b = pnp.array(np.zeros(out_dim), requires_grad=True)
    return W, b


def relu(x):
    return pnp.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + pnp.exp(-pnp.clip(x, -30, 30)))


def tanh(x):
    return pnp.tanh(x)


class ClassicalParams:
    """Holds all trainable parameters for E, G, D as a flat dict."""
    def __init__(self, seed=42):
        r = np.random.default_rng(seed)
        # Encoder: 187 -> 32 -> latent_dim
        self.E_W1, self.E_b1 = init_dense(r, INPUT_DIM, HIDDEN_DIM)
        self.E_W2, self.E_b2 = init_dense(r, HIDDEN_DIM, LATENT_DIM)
        # Generator (decoder): latent_dim -> 32 -> 187
        self.G_W1, self.G_b1 = init_dense(r, LATENT_DIM, HIDDEN_DIM)
        self.G_W2, self.G_b2 = init_dense(r, HIDDEN_DIM, INPUT_DIM)
        # Discriminator: 187 -> 32 -> 1
        self.D_W1, self.D_b1 = init_dense(r, INPUT_DIM, HIDDEN_DIM)
        self.D_W2, self.D_b2 = init_dense(r, HIDDEN_DIM, 1)

    def ae_params(self):
        return [self.E_W1, self.E_b1, self.E_W2, self.E_b2,
                self.G_W1, self.G_b1, self.G_W2, self.G_b2]

    def d_params(self):
        return [self.D_W1, self.D_b1, self.D_W2, self.D_b2]

    def set_ae_params(self, params):
        (self.E_W1, self.E_b1, self.E_W2, self.E_b2,
         self.G_W1, self.G_b1, self.G_W2, self.G_b2) = params

    def set_d_params(self, params):
        (self.D_W1, self.D_b1, self.D_W2, self.D_b2) = params


def encode(x, p):
    h = relu(x @ p.E_W1 + p.E_b1)
    z = h @ p.E_W2 + p.E_b2
    return z


def generate(z, p):
    h = relu(z @ p.G_W1 + p.G_b1)
    xh = tanh(h @ p.G_W2 + p.G_b2)  # beats are roughly in [-1,1] after Kaggle preprocessing/scaling
    return xh


def discriminate(x, p):
    h = relu(x @ p.D_W1 + p.D_b1)
    out = sigmoid(h @ p.D_W2 + p.D_b2)
    return out


def reconstruct(x, p):
    return generate(encode(x, p), p)


# ---------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------
def bce(pred, target):
    eps = 1e-7
    pred = pnp.clip(pred, eps, 1 - eps)
    return -pnp.mean(target * pnp.log(pred) + (1 - target) * pnp.log(1 - pred))


def ae_loss_fn(E_W1, E_b1, E_W2, E_b2, G_W1, G_b1, G_W2, G_b2, x_batch, p, lambda_adv=0.1):
    p.E_W1, p.E_b1, p.E_W2, p.E_b2 = E_W1, E_b1, E_W2, E_b2
    p.G_W1, p.G_b1, p.G_W2, p.G_b2 = G_W1, G_b1, G_W2, G_b2
    xh = reconstruct(x_batch, p)
    recon_loss = pnp.mean((x_batch - xh) ** 2)
    d_out = discriminate(xh, p)
    adv_loss = bce(d_out, pnp.ones((x_batch.shape[0], 1)))  # fool D
    return recon_loss + lambda_adv * adv_loss


def d_loss_fn(D_W1, D_b1, D_W2, D_b2, x_batch, xh_batch, p):
    p.D_W1, p.D_b1, p.D_W2, p.D_b2 = D_W1, D_b1, D_W2, D_b2
    d_real = discriminate(x_batch, p)
    d_fake = discriminate(xh_batch, p)
    loss_real = bce(d_real, pnp.ones((x_batch.shape[0], 1)))
    loss_fake = bce(d_fake, pnp.zeros((xh_batch.shape[0], 1)))
    return 0.5 * (loss_real + loss_fake)


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
def train(train_tag="10pct", epochs=8, batch_size=64, lr=0.01, max_batches_per_epoch=200):
    print(f"\n=== Training classical GAN baseline on train_normal_{train_tag}.npy ===")
    X = np.load(os.path.join(OUT_DIR, f"train_normal_{train_tag}.npy"))
    n = X.shape[0]
    p = ClassicalParams(seed=42)

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

            # --- Update discriminator ---
            xh_batch = reconstruct(x_batch, p)
            xh_batch = pnp.array(np.array(xh_batch))  # detach from AE graph

            new_d, d_loss = opt_d.step_and_cost(
                lambda W1, b1, W2, b2: d_loss_fn(W1, b1, W2, b2, x_batch, xh_batch, p),
                p.D_W1, p.D_b1, p.D_W2, p.D_b2
            )
            p.D_W1, p.D_b1, p.D_W2, p.D_b2 = new_d

            # --- Update autoencoder (E+G) ---
            new_ae, ae_loss = opt_ae.step_and_cost(
                lambda ew1, eb1, ew2, eb2, gw1, gb1, gw2, gb2: ae_loss_fn(
                    ew1, eb1, ew2, eb2, gw1, gb1, gw2, gb2, x_batch, p),
                p.E_W1, p.E_b1, p.E_W2, p.E_b2, p.G_W1, p.G_b1, p.G_W2, p.G_b2
            )
            (p.E_W1, p.E_b1, p.E_W2, p.E_b2,
             p.G_W1, p.G_b1, p.G_W2, p.G_b2) = new_ae

            epoch_ae_loss += float(ae_loss)
            epoch_d_loss += float(d_loss)

        epoch_ae_loss /= n_batches
        epoch_d_loss /= n_batches
        history.append({"epoch": epoch, "ae_loss": epoch_ae_loss, "d_loss": epoch_d_loss})
        print(f"  epoch {epoch+1}/{epochs}  ae_loss={epoch_ae_loss:.5f}  d_loss={epoch_d_loss:.5f}  "
              f"elapsed={time.time()-t0:.1f}s")

    # Save trained params
    save_path = os.path.join(RESULTS_DIR, f"classical_params_{train_tag}.npz")
    np.savez(save_path,
              E_W1=np.array(p.E_W1), E_b1=np.array(p.E_b1),
              E_W2=np.array(p.E_W2), E_b2=np.array(p.E_b2),
              G_W1=np.array(p.G_W1), G_b1=np.array(p.G_b1),
              G_W2=np.array(p.G_W2), G_b2=np.array(p.G_b2),
              D_W1=np.array(p.D_W1), D_b1=np.array(p.D_b1),
              D_W2=np.array(p.D_W2), D_b2=np.array(p.D_b2))
    with open(os.path.join(RESULTS_DIR, f"classical_history_{train_tag}.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved trained params -> {save_path}")
    return p, history


def count_params(p):
    total = 0
    for arr in p.ae_params() + p.d_params():
        total += np.array(arr).size
    return total


if __name__ == "__main__":
    p, hist = train(train_tag="10pct", epochs=5, batch_size=64, max_batches_per_epoch=80)
    print("\nTotal classical trainable parameters:", count_params(p))
