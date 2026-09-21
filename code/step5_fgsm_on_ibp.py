"""Step 5 -- re-run FGSM (same epsilon as Step 2) on the IBP-trained model.

The evaluation focuses on exactly the test samples that fooled the baseline
network in Step 2, as required by the assignment, and additionally reports the
attack on the full test set so that the two models can be compared directly.
"""
import json
import os

import numpy as np

from tas_lib import MLP, fgsm, load_mnist, pgd, set_seed
from config import CFG


def main():
    rng = set_seed(CFG["seed"])
    d = load_mnist(CFG["data"])
    Xte, yte = d["Xte"], d["yte"]
    eps = CFG["eps"]

    base = MLP.load(os.path.join(CFG["models"], "mlp_baseline.npz"))
    ibp = MLP.load(os.path.join(CFG["models"], "mlp_ibp.npz"))

    f = np.load(os.path.join(CFG["results"], "fooled_samples.npz"))
    idx, Xf, yf = f["idx"], f["x_clean"], f["y_true"]
    print(f"{len(idx)} samples fooled the baseline model at eps={eps}")

    out = {"eps": eps, "n_previously_fooled": int(len(idx))}

    # --- (a) the previously fooled subset, evaluated on the new model --- #
    clean_new = ibp.forward(Xf).argmax(1)
    out["ibp_clean_acc_on_fooled_subset"] = float((clean_new == yf).mean())

    Xf_adv_new = fgsm(ibp, Xf, yf, eps)          # attack regenerated on the new model
    pred_new = ibp.forward(Xf_adv_new).argmax(1)
    out["ibp_fgsm_acc_on_fooled_subset"] = float((pred_new == yf).mean())
    out["n_still_fooled"] = int((pred_new != yf).sum())
    out["n_recovered"] = int((pred_new == yf).sum())

    # transfer of the *old* adversarial images (crafted on the baseline)
    pred_transfer = ibp.forward(f["x_adv"]).argmax(1)
    out["ibp_acc_on_transferred_adv"] = float((pred_transfer == yf).mean())

    Xf_pgd, _ = pgd(ibp, Xf, yf, eps, steps=100, restarts=5, rng=rng)
    out["ibp_pgd_acc_on_fooled_subset"] = float((ibp.forward(Xf_pgd).argmax(1) == yf).mean())

    # --- (b) full test set, both models, for reference --- #
    for name, net in (("baseline", base), ("ibp", ibp)):
        clean = net.forward(Xte).argmax(1)
        adv = net.forward(fgsm(net, Xte, yte, eps)).argmax(1)
        Xp, _ = pgd(net, Xte, yte, eps, steps=50, restarts=3, rng=rng)
        out[f"{name}_full_clean_acc"] = float((clean == yte).mean())
        out[f"{name}_full_fgsm_acc"] = float((adv == yte).mean())
        out[f"{name}_full_pgd_acc"] = float((net.forward(Xp).argmax(1) == yte).mean())

    # --- (c) FGSM sweep on the new model --- #
    sweep = []
    for e in CFG["eps_grid"]:
        a = ibp.forward(fgsm(ibp, Xte, yte, e)).argmax(1) if e > 0 else ibp.forward(Xte).argmax(1)
        sweep.append({"eps": e, "adv_acc": float((a == yte).mean())})
    out["sweep_ibp"] = sweep

    with open(os.path.join(CFG["results"], "step5_fgsm_on_ibp.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n--- previously fooled subset ({len(idx)} samples) ---")
    print(f"clean accuracy of IBP model      : {out['ibp_clean_acc_on_fooled_subset']*100:.2f}%")
    print(f"FGSM (re-crafted) accuracy       : {out['ibp_fgsm_acc_on_fooled_subset']*100:.2f}%")
    print(f"  -> recovered {out['n_recovered']} / {len(idx)}, still fooled {out['n_still_fooled']}")
    print(f"transferred baseline adv. images : {out['ibp_acc_on_transferred_adv']*100:.2f}%")
    print(f"PGD-100 accuracy                 : {out['ibp_pgd_acc_on_fooled_subset']*100:.2f}%")
    print(f"\n--- full test set at eps={eps} ---")
    for n in ("baseline", "ibp"):
        print(f"{n:9s} clean={out[f'{n}_full_clean_acc']*100:5.2f}%  "
              f"FGSM={out[f'{n}_full_fgsm_acc']*100:5.2f}%  "
              f"PGD={out[f'{n}_full_pgd_acc']*100:5.2f}%")
    return out


if __name__ == "__main__":
    main()
