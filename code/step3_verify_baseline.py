"""Step 3 -- prove the weakness of the baseline network formally.

The assignment assigns one verification tool per student (ERAN with the
deepzono/deeppoly domain, or CROWN through auto_LiRPA).  Because both tools
implement the same two abstract domains, this submission re-implements both of
them from scratch (`verifiers.py`) so that their outputs can be compared to each
other and, more importantly, to a *complete* MILP verifier that settles every
instance exactly.  A ready-to-run script for the official auto_LiRPA/CROWN tool
and an ONNX export for ERAN are also produced (see `export_models.py`).
"""
import json
import os

import numpy as np

from tas_lib import MLP, load_mnist
from config import CFG
import verification_driver as D


def main(tag="baseline"):
    d = load_mnist(CFG["data"])
    n = CFG["n_verify"]
    X, y = d["Xte"][:n], d["yte"][:n]
    net = MLP.load(os.path.join(CFG["models"], f"mlp_{tag}.npz"))
    eps = CFG["eps"]

    print(f"=== verification of '{tag}' on the first {n} test images, eps={eps} ===")
    res = D.run_full(net, X, y, eps, milp_time_limit=CFG["milp_time_limit"],
                     seed=CFG["seed"], verbose=True)

    print("\ncertified accuracy by domain:")
    for k, v in res["verified_acc"].items():
        print(f"  {k:10s}: {v*100:6.2f}%")
    print(f"complete (MILP) verified accuracy: {res['complete_verified_acc']*100:.2f}%")
    print(f"ROBUST={res['n_robust']}  FALSIFIED={res['n_falsified']}  "
          f"MISCLF={res['n_misclf']}  TIMEOUT={res['n_timeout']}  "
          f"UNKNOWN={res['n_unknown']}")
    print(f"unstable ReLUs / input: mean={np.mean(res['unstable_total']):.1f} "
          f"of {sum(CFG.get('hidden', [64, 32]))}")

    print("\n--- epsilon sweep ---")
    rows = D.sweep(net, X, y, CFG["eps_verify_grid"])
    for r in rows:
        print(f"  eps={r['eps']:5.3f}  pgd={r['pgd_acc']*100:5.1f}%  "
              f"box={r['ver_interval']*100:5.1f}%  zono={r['ver_deepzono']*100:5.1f}%  "
              f"poly={r['ver_deeppoly']*100:5.1f}%")
    res["sweep"] = rows

    with open(os.path.join(CFG["results"], f"step{3 if tag == 'baseline' else 6}_verify_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


if __name__ == "__main__":
    main("baseline")
