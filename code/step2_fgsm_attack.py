"""Step 2 -- attack the trained model with FGSM.

Reports, for a grid of perturbation budgets epsilon, how much of the test set is
misclassified, selects the operating point epsilon* used by the remaining steps,
and stores the identity of every test sample that the attack fools (these are
re-used in Step 5).
"""
import json
import os

import numpy as np

from tas_lib import MLP, accuracy, fgsm, load_mnist, pgd, set_seed
from config import CFG


def main():
    rng = set_seed(CFG["seed"])
    d = load_mnist(CFG["data"])
    Xte, yte = d["Xte"], d["yte"]
    net = MLP.load(os.path.join(CFG["models"], "mlp_baseline.npz"))

    clean_pred = net.forward(Xte).argmax(1)
    clean_correct = clean_pred == yte
    clean_acc = float(clean_correct.mean())
    print(f"clean test accuracy: {clean_acc:.4f}")

    rows = []
    for eps in CFG["eps_grid"]:
        if eps == 0:
            acc, err_new = clean_acc, 0.0
        else:
            Xa = fgsm(net, Xte, yte, eps)
            pa = net.forward(Xa).argmax(1)
            acc = float((pa == yte).mean())
            # fraction of the whole test set that was correct and became wrong
            err_new = float((clean_correct & (pa != yte)).mean())
        rows.append({"eps": eps, "adv_acc": acc, "adv_err": 1 - acc,
                     "newly_fooled_frac": err_new})
        print(f"eps={eps:5.3f}  adv acc={acc:.4f}  adv err={1-acc:.4f}  "
              f"newly fooled={err_new*100:.2f}% of test set")

    # ---------------- operating point ---------------- #
    eps = CFG["eps"]
    Xadv = fgsm(net, Xte, yte, eps)
    adv_pred = net.forward(Xadv).argmax(1)
    fooled = clean_correct & (adv_pred != yte)
    fooled_idx = np.where(fooled)[0]

    # PGD, a strictly stronger attack, for reference
    Xpgd, pgd_ok = pgd(net, Xte, yte, eps, steps=50, restarts=3, rng=rng)
    pgd_pred = net.forward(Xpgd).argmax(1)
    pgd_acc = float((pgd_pred == yte).mean())

    np.savez_compressed(
        os.path.join(CFG["results"], "fooled_samples.npz"),
        idx=fooled_idx,
        x_clean=Xte[fooled_idx],
        x_adv=Xadv[fooled_idx],
        y_true=yte[fooled_idx],
        y_adv_pred=adv_pred[fooled_idx],
        eps=np.array(eps),
    )

    summary = {
        "clean_acc": clean_acc,
        "sweep": rows,
        "eps_star": eps,
        "fgsm_adv_acc": float((adv_pred == yte).mean()),
        "fgsm_err_rate": float((adv_pred != yte).mean()),
        "n_fooled": int(fooled.sum()),
        "fooled_frac_of_testset": float(fooled.mean()),
        "fooled_frac_of_correct": float(fooled.sum() / clean_correct.sum()),
        "pgd_adv_acc": pgd_acc,
        "n_fooled_indices_saved": int(len(fooled_idx)),
    }
    with open(os.path.join(CFG["results"], "step2_fgsm.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n--- operating point eps* = {eps} ---")
    print(f"FGSM accuracy      : {summary['fgsm_adv_acc']:.4f}")
    print(f"FGSM error rate    : {summary['fgsm_err_rate']:.4f}")
    print(f"newly fooled       : {summary['n_fooled']} samples "
          f"({summary['fooled_frac_of_testset']*100:.2f}% of the test set, "
          f"{summary['fooled_frac_of_correct']*100:.2f}% of correctly classified ones)")
    print(f"PGD-50 accuracy    : {pgd_acc:.4f}  (stronger attack, reference)")
    return summary


if __name__ == "__main__":
    main()
