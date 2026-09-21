"""
tas_lib.py -- Core library for the Trustworthy AI Systems final project.

Implements, in pure NumPy (no deep-learning framework required):
  * MNIST loading from the original IDX files
  * A 3-layer fully connected ReLU network 784 -> 64 -> 32 -> 10
  * Manual forward / backward passes and an Adam optimiser
  * FGSM and PGD adversarial attacks
  * Interval Bound Propagation (IBP) forward pass and its exact gradients
  * The IBP ("worst-case logit") training loss described in the assignment

Verification back-ends (DeepPoly/CROWN, DeepZono, MILP) live in `verifiers.py`.

Author: final project submission, course "Trustworthy AI Systems".
"""

from __future__ import annotations

import gzip
import os
import struct

import numpy as np

# --------------------------------------------------------------------------- #
#  Reproducibility
# --------------------------------------------------------------------------- #
SEED = 1405  # Persian year of the deadline; any fixed value works.


def set_seed(seed: int = SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
#  1. Data
# --------------------------------------------------------------------------- #
def _read_idx(path: str) -> np.ndarray:
    """Read an IDX (MNIST) file, transparently handling .gz compression."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as fh:
        magic, = struct.unpack(">I", fh.read(4))
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, fh.read(4 * ndim))
        buf = fh.read()
    return np.frombuffer(buf, dtype=np.uint8).reshape(dims)


def load_mnist(data_dir: str) -> dict:
    """Return MNIST as float32 arrays, images flattened to 784 and scaled to [0,1]."""
    f = {
        "Xtr": "train-images-idx3-ubyte.gz",
        "ytr": "train-labels-idx1-ubyte.gz",
        "Xte": "t10k-images-idx3-ubyte.gz",
        "yte": "t10k-labels-idx1-ubyte.gz",
    }
    out = {}
    for key, name in f.items():
        arr = _read_idx(os.path.join(data_dir, name))
        if key.startswith("X"):
            out[key] = (arr.reshape(len(arr), -1).astype(np.float32) / 255.0)
        else:
            out[key] = arr.astype(np.int64)
    return out


# --------------------------------------------------------------------------- #
#  2. Model
# --------------------------------------------------------------------------- #
LAYER_SIZES = (784, 64, 32, 10)


class MLP:
    """Fully connected ReLU network 784 -> 64 -> 32 -> 10.

    Convention: activations are row vectors, so a layer computes  z = a @ W + b
    with W of shape (fan_in, fan_out).  ReLU is applied after layers 1 and 2;
    layer 3 emits the 10 raw logits (the "one-hot encoded" output of the task).
    """

    def __init__(self, sizes=LAYER_SIZES, rng: np.random.Generator | None = None):
        rng = rng or set_seed()
        self.sizes = tuple(sizes)
        self.W, self.b = [], []
        for fin, fout in zip(self.sizes[:-1], self.sizes[1:]):
            # He (Kaiming) initialisation -- appropriate for ReLU networks.
            self.W.append((rng.standard_normal((fin, fout)) * np.sqrt(2.0 / fin)).astype(np.float32))
            self.b.append(np.zeros(fout, dtype=np.float32))

    # ---------------- parameter (de)serialisation ---------------- #
    @property
    def params(self):
        return self.W + self.b

    def save(self, path: str, **meta):
        d = {f"W{i+1}": w for i, w in enumerate(self.W)}
        d.update({f"b{i+1}": b for i, b in enumerate(self.b)})
        d.update({k: np.asarray(v) for k, v in meta.items()})
        np.savez(path, **d)

    @staticmethod
    def load(path: str) -> "MLP":
        z = np.load(path, allow_pickle=True)
        n = sum(1 for k in z.files if k.startswith("W"))
        net = MLP.__new__(MLP)
        net.W = [z[f"W{i+1}"].astype(np.float32) for i in range(n)]
        net.b = [z[f"b{i+1}"].astype(np.float32) for i in range(n)]
        net.sizes = tuple([net.W[0].shape[0]] + [w.shape[1] for w in net.W])
        return net

    # ---------------- nominal forward / backward ---------------- #
    def forward(self, X: np.ndarray, cache: bool = False):
        """Nominal forward pass.  X: (B, 784).  Returns logits (B, 10)."""
        a = X
        acts, pres = [X], []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            pres.append(z)
            a = np.maximum(z, 0.0) if i < len(self.W) - 1 else z
            acts.append(a)
        return (a, acts, pres) if cache else a

    def backward(self, acts, pres, dlogits):
        """Backprop given dL/dlogits.  Returns (gW, gb) lists."""
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        g = dlogits
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ g
            gb[i] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.W[i].T) * (pres[i - 1] > 0)
        return gW, gb


# --------------------------------------------------------------------------- #
#  3. Losses
# --------------------------------------------------------------------------- #
def softmax_ce(logits: np.ndarray, y: np.ndarray):
    """Mean softmax cross-entropy and dL/dlogits (already averaged over batch)."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    B = len(y)
    loss = float(-np.log(np.clip(p[np.arange(B), y], 1e-12, None)).mean())
    d = p.copy()
    d[np.arange(B), y] -= 1.0
    return loss, d / B


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    return float((logits.argmax(1) == y).mean())


# --------------------------------------------------------------------------- #
#  4. Interval Bound Propagation
# --------------------------------------------------------------------------- #
def input_box(X: np.ndarray, eps: float, lo: float = 0.0, hi: float = 1.0):
    """l-infinity ball of radius eps around X, intersected with the valid pixel box."""
    return np.clip(X - eps, lo, hi), np.clip(X + eps, lo, hi)


def ibp_forward(net: MLP, l0: np.ndarray, u0: np.ndarray, cache: bool = False):
    """Propagate an input box through the network with interval arithmetic.

    Uses the numerically stable midpoint/radius formulation
        mu' = W^T mu + b ,   r' = |W|^T r
    which is exact for affine layers, and the monotone ReLU rule
        [l, u] -> [relu(l), relu(u)] .
    Returns the logit bounds (l3, u3); with cache=True also the tape needed
    for the backward pass.
    """
    tape = []
    l, u = l0, u0
    n = len(net.W)
    for i, (W, b) in enumerate(zip(net.W, net.b)):
        mu = 0.5 * (l + u)
        r = 0.5 * (u - l)
        m = mu @ W + b
        rad = r @ np.abs(W)
        lz, uz = m - rad, m + rad
        if cache:
            tape.append({"mu": mu, "r": r, "lz": lz, "uz": uz})
        if i < n - 1:
            l, u = np.maximum(lz, 0.0), np.maximum(uz, 0.0)
        else:
            l, u = lz, uz
    return (l, u, tape) if cache else (l, u)


def ibp_all_preactivation_bounds(net: MLP, l0: np.ndarray, u0: np.ndarray):
    """Return the list of pre-activation bounds [(l,u), ...] for every layer."""
    out = []
    l, u = l0, u0
    for i, (W, b) in enumerate(zip(net.W, net.b)):
        mu, r = 0.5 * (l + u), 0.5 * (u - l)
        m, rad = mu @ W + b, r @ np.abs(W)
        lz, uz = m - rad, m + rad
        out.append((lz, uz))
        if i < len(net.W) - 1:
            l, u = np.maximum(lz, 0.0), np.maximum(uz, 0.0)
    return out


def worst_case_logits(l3: np.ndarray, u3: np.ndarray, y: np.ndarray):
    """Assemble the worst-case logit vector  z_hat  of the assignment:

        z_hat[c] = z_min[c]           (true class pushed as low as possible)
        z_hat[j] = z_max[j],  j != c  (every wrong class pushed as high as possible)

    Equivalently z_hat = m + s * rad with s = +1 except s[c] = -1.
    """
    B = len(y)
    s = np.ones_like(u3)
    s[np.arange(B), y] = -1.0
    zhat = np.where(s > 0, u3, l3)
    return zhat, s


def ibp_loss_and_grads(net: MLP, X, y, eps, kappa: float):
    """Combined IBP training objective

        L = kappa * CE(f(x), c)  +  (1 - kappa) * CE(z_hat, c)

    Returns (total loss, nominal loss, robust loss, gW, gb, verified_mask).
    Gradients are derived analytically; see report Section 5.2.
    """
    B = len(y)
    idx = np.arange(B)

    # ---- nominal branch ---- #
    logits, acts, pres = net.forward(X, cache=True)
    nat_loss, dlog = softmax_ce(logits, y)
    gW, gb = net.backward(acts, pres, dlog * kappa)

    if kappa >= 1.0 - 1e-12 or eps <= 0:
        return nat_loss, nat_loss, nat_loss, gW, gb, None

    # ---- robust (IBP) branch ---- #
    l0, u0 = input_box(X, eps)
    l3, u3, tape = ibp_forward(net, l0, u0, cache=True)
    zhat, s = worst_case_logits(l3, u3, y)
    rob_loss, dzhat = softmax_ce(zhat, y)
    dzhat = dzhat * (1.0 - kappa)

    # zhat = m + s * rad, so  dm = dzhat  and  drad = s * dzhat
    dm = dzhat
    drad = s * dzhat
    n = len(net.W)
    for i in reversed(range(n)):
        t = tape[i]
        gW[i] = gW[i] + t["mu"].T @ dm + (t["r"].T @ drad) * np.sign(net.W[i])
        gb[i] = gb[i] + dm.sum(axis=0)
        if i == 0:
            break
        dmu = dm @ net.W[i].T
        dr = drad @ np.abs(net.W[i]).T
        # mu = (relu(lz) + relu(uz))/2 ,  r = (relu(uz) - relu(lz))/2
        dl_act = 0.5 * dmu - 0.5 * dr
        du_act = 0.5 * dmu + 0.5 * dr
        prev = tape[i - 1]
        dlz = dl_act * (prev["lz"] > 0)
        duz = du_act * (prev["uz"] > 0)
        dm = dlz + duz            # lz = m - rad, uz = m + rad
        drad = duz - dlz

    verified = (zhat.argmax(1) == y)
    total = kappa * nat_loss + (1.0 - kappa) * rob_loss
    return total, nat_loss, rob_loss, gW, gb, verified


# --------------------------------------------------------------------------- #
#  5. Optimiser
# --------------------------------------------------------------------------- #
class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# --------------------------------------------------------------------------- #
#  6. Attacks
# --------------------------------------------------------------------------- #
def fgsm(net: MLP, X, y, eps, lo=0.0, hi=1.0):
    """Fast Gradient Sign Method:  x_adv = clip(x + eps * sign(grad_x CE))."""
    logits, acts, pres = net.forward(X, cache=True)
    _, dlog = softmax_ce(logits, y)
    g = dlog
    for i in reversed(range(len(net.W))):
        g = g @ net.W[i].T
        if i > 0:
            g = g * (pres[i - 1] > 0)
    return np.clip(X + eps * np.sign(g), lo, hi).astype(np.float32)


def pgd(net: MLP, X, y, eps, steps=50, alpha=None, restarts=3,
        rng: np.random.Generator | None = None, lo=0.0, hi=1.0):
    """Projected Gradient Descent (l-inf) with random restarts.

    Used as a *falsifier*: any successful attack is a definitive
    counterexample to local robustness.
    """
    rng = rng or set_seed()
    alpha = alpha or max(eps / 4.0, 1e-3)
    lb, ub = input_box(X, eps, lo, hi)
    best = X.copy()
    still = np.ones(len(X), dtype=bool)
    for r in range(restarts):
        Xa = X.copy() if r == 0 else np.clip(
            X + rng.uniform(-eps, eps, X.shape).astype(np.float32), lb, ub)
        for _ in range(steps):
            logits, acts, pres = net.forward(Xa, cache=True)
            _, dlog = softmax_ce(logits, y)
            g = dlog
            for i in reversed(range(len(net.W))):
                g = g @ net.W[i].T
                if i > 0:
                    g = g * (pres[i - 1] > 0)
            Xa = np.clip(np.clip(Xa + alpha * np.sign(g), lb, ub), lo, hi).astype(np.float32)
        pred = net.forward(Xa).argmax(1)
        hit = (pred != y) & still
        best[hit] = Xa[hit]
        still &= ~hit
    return best, ~still  # adversarial inputs, and success mask


# --------------------------------------------------------------------------- #
#  7. Gradient check (used to validate the hand-derived IBP gradients)
# --------------------------------------------------------------------------- #
def gradient_check(seed=0, eps=0.1, kappa=0.5, n=4, h=1e-4):
    rng = np.random.default_rng(seed)
    net = MLP((12, 7, 5, 4), rng)
    for i in range(len(net.W)):
        net.W[i] = net.W[i].astype(np.float64)
        net.b[i] = net.b[i].astype(np.float64) + rng.standard_normal(net.b[i].shape) * 0.1
    X = rng.random((n, 12))
    y = rng.integers(0, 4, n)

    L, _, _, gW, gb, _ = ibp_loss_and_grads(net, X, y, eps, kappa)
    worst = 0.0
    for lst, glst in ((net.W, gW), (net.b, gb)):
        for k in range(len(lst)):
            flat = lst[k].ravel()
            gflat = np.asarray(glst[k]).ravel()
            for _ in range(25):
                j = rng.integers(0, flat.size)
                old = flat[j]
                flat[j] = old + h
                Lp = ibp_loss_and_grads(net, X, y, eps, kappa)[0]
                flat[j] = old - h
                Lm = ibp_loss_and_grads(net, X, y, eps, kappa)[0]
                flat[j] = old
                num = (Lp - Lm) / (2 * h)
                den = max(1e-8, abs(num) + abs(gflat[j]))
                worst = max(worst, abs(num - gflat[j]) / den)
    return worst


if __name__ == "__main__":
    print("max relative gradient error:", gradient_check())
