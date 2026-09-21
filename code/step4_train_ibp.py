"""Step 4 -- train the network with Interval Bound Propagation (certified training).

Instead of pushing a single image through the layers, an *input box*
B_eps(x) = [x-eps, x+eps] cap [0,1]^784 is propagated, giving a lower and an
upper bound on each of the ten output logits.  The loss is then evaluated on
the worst-case logit vector

    z_hat[c] = z_min[c]            (true class driven as low as possible)
    z_hat[j] = z_max[j]  (j != c)  (each wrong class driven as high as possible)

so that minimising CE(z_hat, c) directly maximises a *certified* lower bound on
the classification margin.  See Section 5 of the report for the derivation.

Two schedules are used, exactly as recommended by Gowal et al. (2018):
  * eps is ramped linearly from 0 to eps_train,
  * kappa (the weight of the ordinary loss) is annealed from 1 to kappa_end,
because starting directly at the full eps makes the bounds so loose that
training collapses to the trivial constant classifier.
"""
import json
import os
import time

import numpy as np

from tas_lib import (MLP, Adam, accuracy, ibp_forward, ibp_loss_and_grads,
                     input_box, load_mnist, set_seed, worst_case_logits)
from config import CFG


def evaluate_certified(net, X, y, eps, bs=1000):
    """Standard accuracy and IBP-certified accuracy on a whole dataset."""
    ok = cert = 0
    for s in range(0, len(X), bs):
        Xi, yi = X[s:s + bs], y[s:s + bs]
        ok += int((net.forward(Xi).argmax(1) == yi).sum())
        l0, u0 = input_box(Xi, eps)
        l3, u3 = ibp_forward(net, l0, u0)
        zhat, _ = worst_case_logits(l3, u3, yi)
        cert += int((zhat.argmax(1) == yi).sum())
    return ok / len(X), cert / len(X)


def schedules(epoch, n_batches, batch_idx):
    """Linear ramp-up over `ramp_epochs` epochs, applied per batch."""
    w, r = CFG["warmup_epochs"], CFG["ramp_epochs"]
    step = epoch * n_batches + batch_idx
    t0, t1 = w * n_batches, (w + r) * n_batches
    if step <= t0:
        frac = 0.0
    elif step >= t1:
        frac = 1.0
    else:
        frac = (step - t0) / (t1 - t0)
    eps = frac * CFG["eps"] * CFG["eps_train_factor"]
    kappa = 1.0 + frac * (CFG["kappa_end"] - 1.0)
    return eps, kappa


def main():
    rng = set_seed(CFG["seed"])
    d = load_mnist(CFG["data"])
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    eps_eval = CFG["eps"]

    net = MLP(rng=rng)
    opt = Adam(net.params, lr=CFG["lr_ibp"])
    B = CFG["batch"]
    nb = int(np.ceil(len(Xtr) / B))
    history = []
    t0 = time.time()

    for ep in range(CFG["epochs_ibp"]):
        # step-wise learning-rate decay once the epsilon ramp has finished
        for frac in CFG["lr_decay_at"]:
            if ep == int(frac * CFG["epochs_ibp"]):
                opt.lr *= 0.2
                print(f"  (lr -> {opt.lr:.2e})")
        perm = rng.permutation(len(Xtr))
        agg = {"loss": 0.0, "nat": 0.0, "rob": 0.0, "n": 0}
        eps_ep = kap_ep = 0.0
        for bi, s in enumerate(range(0, len(perm), B)):
            idx = perm[s:s + B]
            eps_b, kap_b = schedules(ep, nb, bi)
            eps_ep, kap_ep = eps_b, kap_b
            L, nat, rob, gW, gb, _ = ibp_loss_and_grads(net, Xtr[idx], ytr[idx],
                                                        eps_b, kap_b)
            opt.step(net.params, gW + gb)
            agg["loss"] += L; agg["nat"] += nat; agg["rob"] += rob; agg["n"] += 1

        te_acc, te_cert = evaluate_certified(net, Xte, yte, eps_eval)
        history.append({"epoch": ep + 1, "eps": eps_ep, "kappa": kap_ep,
                        "loss": agg["loss"] / agg["n"],
                        "nat_loss": agg["nat"] / agg["n"],
                        "rob_loss": agg["rob"] / agg["n"],
                        "test_acc": te_acc, "test_ibp_cert_acc": te_cert})
        print(f"epoch {ep+1:3d}  eps={eps_ep:.4f} kappa={kap_ep:.3f}  "
              f"loss={agg['loss']/agg['n']:.4f}  test_acc={te_acc:.4f}  "
              f"IBP-cert={te_cert:.4f}")

    net.save(os.path.join(CFG["models"], "mlp_ibp.npz"),
             arch=np.array(net.sizes), eps_train=CFG["eps"])
    tr_acc, tr_cert = evaluate_certified(net, Xtr, ytr, eps_eval)
    summary = {"history": history, "eps_train": CFG["eps"],
               "final_test_acc": history[-1]["test_acc"],
               "final_test_ibp_cert_acc": history[-1]["test_ibp_cert_acc"],
               "final_train_acc": tr_acc, "final_train_ibp_cert_acc": tr_cert,
               "epochs": CFG["epochs_ibp"], "lr": CFG["lr_ibp"],
               "warmup_epochs": CFG["warmup_epochs"],
               "ramp_epochs": CFG["ramp_epochs"], "kappa_end": CFG["kappa_end"],
               "train_time_s": time.time() - t0}
    with open(os.path.join(CFG["results"], "step4_ibp_training.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nfinal: test acc={summary['final_test_acc']:.4f}  "
          f"IBP-certified acc={summary['final_test_ibp_cert_acc']:.4f}  "
          f"({summary['train_time_s']:.0f}s)")
    return summary


if __name__ == "__main__":
    main()
