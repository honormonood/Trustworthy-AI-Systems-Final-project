"""verification_driver.py -- the analysis pipeline shared by Steps 3 and 6.

For a set of test inputs it produces
  * certified worst-case margins from three sound-but-incomplete abstract
    domains (interval, DeepZono, DeepPoly/CROWN),
  * a falsification pass with PGD (any success is a definitive counterexample),
  * a complete decision for every remaining undecided input via MILP,
so that the final verdict for each input is exact: ROBUST or FALSIFIED.
"""
from __future__ import annotations

import time

import numpy as np

import verifiers as V
from tas_lib import input_box, pgd, set_seed


def certified_margins(net, X, y, eps, methods=("interval", "deepzono", "deeppoly")):
    out, timing, pre_dp = {}, {}, None
    for m in methods:
        t0 = time.time()
        if m == "interval":
            out[m], _ = V.verify_interval(net, X, y, eps)
        elif m == "deepzono":
            out[m] = V.verify_deepzono(net, X, y, eps)
        elif m == "deeppoly":
            out[m], pre_dp = V.verify_deeppoly(net, X, y, eps, return_pre=True)
        elif m == "crown_zero":
            out[m] = V.verify_deeppoly(net, X, y, eps, lam_rule="zero")
        elif m == "crown_parallel":
            out[m] = V.verify_deeppoly(net, X, y, eps, lam_rule="parallel")
        timing[m] = time.time() - t0
    return out, timing, pre_dp


def verified_accuracy(margins, correct):
    """A sample counts as verified only if it is correct AND certified robust."""
    return float((correct & (margins > 0)).mean())


def run_full(net, X, y, eps, milp_time_limit=30.0, pgd_steps=100,
             pgd_restarts=5, seed=0, use_milp=True, verbose=True):
    rng = set_seed(seed)
    n = len(y)
    clean_pred = net.forward(X).argmax(1)
    correct = clean_pred == y

    margins, timing, pre = certified_margins(net, X, y, eps)
    tot_unstable, per_layer = V.unstable_counts(pre)

    # ---- falsification with PGD ---- #
    t0 = time.time()
    Xa, pgd_hit = pgd(net, X, y, eps, steps=pgd_steps, restarts=pgd_restarts, rng=rng)
    timing["pgd"] = time.time() - t0

    status = np.array(["UNKNOWN"] * n, dtype=object)
    decided_by = np.array([""] * n, dtype=object)
    status[~correct] = "MISCLF"
    decided_by[~correct] = "clean"
    status[correct & (margins["deeppoly"] > 0)] = "ROBUST"
    decided_by[correct & (margins["deeppoly"] > 0)] = "deeppoly"
    fals = correct & pgd_hit & (status == "UNKNOWN")
    status[fals] = "FALSIFIED"
    decided_by[fals] = "pgd"

    milp_records = {}
    if use_milp:
        todo = np.where(status == "UNKNOWN")[0]
        if verbose:
            print(f"  MILP needed for {len(todo)}/{n} inputs")
        t0 = time.time()
        for i in todo:
            r = V.verify_milp(net, X[i:i + 1], y[i:i + 1], eps,
                              pre=[(pre[k][0][i:i + 1], pre[k][1][i:i + 1])
                                   for k in range(len(pre))],
                              time_limit=milp_time_limit)[0]
            status[i] = r["status"]
            decided_by[i] = "milp"
            milp_records[int(i)] = {"status": r["status"], "margin": r["margin"],
                                    "time": r["time"]}
            if verbose:
                print(f"    [{i:3d}] {r['status']:10s} margin={r['margin']:+.4f} "
                      f"({r['time']:.1f}s)")
        timing["milp"] = time.time() - t0

    res = {
        "eps": eps,
        "n": n,
        "clean_acc": float(correct.mean()),
        "margins": {k: v.tolist() for k, v in margins.items()},
        "status": status.tolist(),
        "decided_by": decided_by.tolist(),
        "timing": timing,
        "unstable_total": tot_unstable.tolist(),
        "unstable_layer1": per_layer[0].tolist(),
        "unstable_layer2": per_layer[1].tolist(),
        "pgd_success": pgd_hit.tolist(),
        "milp_records": milp_records,
        "verified_acc": {k: verified_accuracy(v, correct) for k, v in margins.items()},
        "n_robust": int((status == "ROBUST").sum()),
        "n_falsified": int((status == "FALSIFIED").sum()),
        "n_misclf": int((status == "MISCLF").sum()),
        "n_timeout": int((status == "TIMEOUT").sum()),
        "n_unknown": int((status == "UNKNOWN").sum()),
    }
    res["complete_verified_acc"] = res["n_robust"] / n
    return res


def sweep(net, X, y, eps_list):
    """Certified accuracy of each incomplete domain over a grid of epsilon."""
    correct = net.forward(X).argmax(1) == y
    rows = []
    for e in eps_list:
        m, t, _ = certified_margins(net, X, y, e)
        _, hit = pgd(net, X, y, e, steps=50, restarts=3)
        row = {"eps": e, "clean_acc": float(correct.mean()),
               "pgd_acc": float((correct & ~hit).mean())}
        for k, v in m.items():
            row[f"ver_{k}"] = verified_accuracy(v, correct)
        rows.append(row)
    return rows
