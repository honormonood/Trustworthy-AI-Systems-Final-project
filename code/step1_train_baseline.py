"""Step 1 -- design and train the baseline 784-64-32-10 fully connected network.

Per the assignment the network is trained for only one or two epochs, i.e. just
long enough to reach a reasonable accuracy, so that adversarial attacks in
Step 2 remain clearly effective.
"""
import json
import os
import time

import numpy as np

from tas_lib import MLP, Adam, accuracy, load_mnist, set_seed, softmax_ce
from config import CFG

os.makedirs(CFG["models"], exist_ok=True)
os.makedirs(CFG["results"], exist_ok=True)


def main():
    rng = set_seed(CFG["seed"])
    d = load_mnist(CFG["data"])
    Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    print(f"train {Xtr.shape}  test {Xte.shape}")

    net = MLP(rng=rng)
    opt = Adam(net.params, lr=CFG["lr_std"])
    B = CFG["batch"]
    history = []
    t0 = time.time()

    for ep in range(CFG["epochs_std"]):
        perm = rng.permutation(len(Xtr))
        run_loss = run_acc = nb = 0
        for s in range(0, len(perm), B):
            idx = perm[s:s + B]
            X, y = Xtr[idx], ytr[idx]
            logits, acts, pres = net.forward(X, cache=True)
            loss, dlog = softmax_ce(logits, y)
            gW, gb = net.backward(acts, pres, dlog)
            opt.step(net.params, gW + gb)
            run_loss += loss
            run_acc += accuracy(logits, y)
            nb += 1
        te_logits = net.forward(Xte)
        te_acc = accuracy(te_logits, yte)
        tr_acc = accuracy(net.forward(Xtr), ytr)
        history.append({"epoch": ep + 1, "train_loss": run_loss / nb,
                        "train_acc": tr_acc, "test_acc": te_acc})
        print(f"epoch {ep+1}: loss={run_loss/nb:.4f}  train_acc={tr_acc:.4f}  "
              f"test_acc={te_acc:.4f}")

    net.save(os.path.join(CFG["models"], "mlp_baseline.npz"),
             arch=np.array(net.sizes), epochs=CFG["epochs_std"])
    summary = {"history": history,
               "final_test_acc": history[-1]["test_acc"],
               "final_train_acc": history[-1]["train_acc"],
               "n_params": int(sum(p.size for p in net.params)),
               "train_time_s": time.time() - t0,
               "epochs": CFG["epochs_std"], "lr": CFG["lr_std"],
               "batch": CFG["batch"], "seed": CFG["seed"]}
    with open(os.path.join(CFG["results"], "step1_baseline.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"parameters: {summary['n_params']}   time: {summary['train_time_s']:.1f}s")
    return summary


if __name__ == "__main__":
    main()
