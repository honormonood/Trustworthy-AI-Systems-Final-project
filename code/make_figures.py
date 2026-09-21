"""make_figures.py -- every figure used in the report."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tas_lib import MLP, fgsm, ibp_forward, input_box, load_mnist, set_seed
from config import CFG
import verifiers as V

R, F = CFG["results"], CFG["figures"]
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"figure.dpi": 160, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
C0, C1 = "#c0392b", "#1f6f8b"


def J(name):
    with open(os.path.join(R, name)) as f:
        return json.load(f)


def fig_training():
    s1, s4 = J("step1_baseline.json"), J("step4_ibp_training.json")
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.8))
    h1 = s1["history"]
    ax[0].plot([r["epoch"] for r in h1], [r["train_acc"] for r in h1], "o-", color=C0, label="train")
    ax[0].plot([r["epoch"] for r in h1], [r["test_acc"] for r in h1], "s--", color=C1, label="test")
    ax[0].set_title("(a) baseline: 2 epochs"); ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("accuracy"); ax[0].set_xticks([1, 2]); ax[0].legend(frameon=False)

    h4 = s4["history"]
    e = [r["epoch"] for r in h4]
    ax[1].plot(e, [r["eps"] for r in h4], color="#666", label=r"$\epsilon$ schedule")
    ax[1].plot(e, [r["kappa"] for r in h4], "--", color="#aaa", label=r"$\kappa$ schedule")
    ax[1].set_title("(b) IBP schedules"); ax[1].set_xlabel("epoch"); ax[1].legend(frameon=False)

    ax[2].plot(e, [r["test_acc"] for r in h4], color=C1, label="test accuracy")
    ax[2].plot(e, [r["test_ibp_cert_acc"] for r in h4], color=C0,
               label=r"IBP-certified acc. @ $\epsilon$=0.1")
    ax[2].set_title("(c) IBP training"); ax[2].set_xlabel("epoch")
    ax[2].set_ylim(0, 1); ax[2].legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_training.png")); plt.close(fig)


def fig_attack_sweep():
    s2, s5 = J("step2_fgsm.json"), J("step5_fgsm_on_ibp.json")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot([r["eps"] for r in s2["sweep"]], [r["adv_acc"] for r in s2["sweep"]],
            "o-", color=C0, label="baseline (Step 1)")
    ax.plot([r["eps"] for r in s5["sweep_ibp"]], [r["adv_acc"] for r in s5["sweep_ibp"]],
            "s-", color=C1, label="IBP-trained (Step 4)")
    ax.axvline(CFG["eps"], color="k", ls=":", lw=1)
    ax.text(CFG["eps"] + .004, .5, r"$\epsilon^\star$", fontsize=9)
    ax.set_xlabel(r"FGSM perturbation budget $\epsilon$")
    ax.set_ylabel("accuracy under attack"); ax.set_ylim(-0.02, 1.0)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_fgsm_sweep.png")); plt.close(fig)


def fig_examples():
    f = np.load(os.path.join(R, "fooled_samples.npz"))
    ibp = MLP.load(os.path.join(CFG["models"], "mlp_ibp.npz"))
    k = 8
    sel = np.linspace(0, len(f["idx"]) - 1, k).astype(int)
    xc, xa = f["x_clean"][sel], f["x_adv"][sel]
    yt, yp = f["y_true"][sel], f["y_adv_pred"][sel]
    xa_new = fgsm(ibp, xc, yt, float(f["eps"]))
    p_new = ibp.forward(xa_new).argmax(1)

    fig, axes = plt.subplots(4, k, figsize=(1.15 * k, 5.0))
    rows = ["clean $x$", r"perturbation", r"$x_{adv}$ (baseline)", r"$x_{adv}$ (IBP model)"]
    for c in range(k):
        for r, img in enumerate([xc[c], (xa[c] - xc[c]), xa[c], xa_new[c]]):
            a = axes[r, c]
            a.imshow(img.reshape(28, 28), cmap="gray" if r != 1 else "bwr",
                     vmin=-float(f["eps"]) if r == 1 else 0,
                     vmax=float(f["eps"]) if r == 1 else 1)
            a.set_xticks([]); a.set_yticks([]); a.grid(False)
            for s in a.spines.values():
                s.set_visible(False)
        axes[0, c].set_title(f"true {yt[c]}", fontsize=8)
        axes[2, c].set_xlabel(f"pred {yp[c]}", fontsize=8, color=C0)
        ok = p_new[c] == yt[c]
        axes[3, c].set_xlabel(f"pred {p_new[c]}", fontsize=8,
                              color="#1e8449" if ok else C0)
    for r, lab in enumerate(rows):
        axes[r, 0].set_ylabel(lab, fontsize=8)
    fig.suptitle(rf"FGSM at $\epsilon={float(f['eps'])}$: samples that fooled the baseline",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_examples.png")); plt.close(fig)


def fig_certified_sweep():
    b, i = J("step3_verify_baseline.json"), J("step6_verify_ibp.json")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True)
    for k, (res, ttl) in enumerate(((b, "(a) baseline model"), (i, "(b) IBP-trained model"))):
        e = [r["eps"] for r in res["sweep"]]
        ax[k].plot(e, [r["pgd_acc"] for r in res["sweep"]], "k^--",
                   label="PGD (upper bound)")
        ax[k].plot(e, [r["ver_deeppoly"] for r in res["sweep"]], "o-", color=C0,
                   label="DeepPoly / CROWN")
        ax[k].plot(e, [r["ver_deepzono"] for r in res["sweep"]], "s-", color=C1,
                   label="DeepZono")
        ax[k].plot(e, [r["ver_interval"] for r in res["sweep"]], "d-", color="#888",
                   label="Interval / IBP")
        ax[k].set_title(ttl); ax[k].set_xlabel(r"$\epsilon$"); ax[k].set_ylim(-0.02, 1.02)
    ax[0].set_ylabel("certified accuracy"); ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_certified_sweep.png")); plt.close(fig)


def fig_unstable_and_margins():
    b, i = J("step3_verify_baseline.json"), J("step6_verify_ibp.json")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.0))
    bins = np.arange(0, 98, 4)
    ax[0].hist(b["unstable_total"], bins=bins, color=C0, alpha=.75, label="baseline")
    ax[0].hist(i["unstable_total"], bins=bins, color=C1, alpha=.75, label="IBP-trained")
    ax[0].set_xlabel("unstable (crossing) ReLUs per input, out of 96")
    ax[0].set_ylabel("number of test inputs"); ax[0].legend(frameon=False)
    ax[0].set_title(r"(a) ReLU instability at $\epsilon=0.1$")

    mb = np.array(b["margins"]["deeppoly"]); mi = np.array(i["margins"]["deeppoly"])
    ax[1].hist(np.clip(mb, -30, 30), bins=40, color=C0, alpha=.75, label="baseline")
    ax[1].hist(np.clip(mi, -30, 30), bins=40, color=C1, alpha=.75, label="IBP-trained")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set_xlabel("certified lower bound on the margin (DeepPoly)")
    ax[1].set_title("(b) certified margins"); ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_unstable_margins.png")); plt.close(fig)


def fig_bound_width():
    d = load_mnist(CFG["data"])
    X, y = d["Xte"][:CFG["n_verify"]], d["yte"][:CFG["n_verify"]]
    l0, u0 = input_box(X, CFG["eps"])
    labels, widths = [], []
    for tag in ("baseline", "ibp"):
        net = MLP.load(os.path.join(CFG["models"], f"mlp_{tag}.npz"))
        l3, u3 = ibp_forward(net, l0, u0)
        pre = V.deeppoly_preactivation_bounds(net, l0, u0)
        widths.append((u3 - l3).mean(1))
        widths.append((pre[-1][1] - pre[-1][0]).mean(1))
        labels += [f"{tag}\nIBP", f"{tag}\nDeepPoly"]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    bp = ax.boxplot(widths, labels=labels, showfliers=False, patch_artist=True)
    for p, c in zip(bp["boxes"], [C0, C0, C1, C1]):
        p.set_facecolor(c); p.set_alpha(.6)
    ax.set_yscale("log"); ax.set_ylabel("mean width of the 10 logit bounds")
    ax.set_title(r"Tightness of the output bounds at $\epsilon=0.1$")
    fig.tight_layout(); fig.savefig(os.path.join(F, "fig_bound_width.png")); plt.close(fig)
    return {l: float(np.median(w)) for l, w in zip(labels, widths)}


if __name__ == "__main__":
    fig_training()
    fig_attack_sweep()
    fig_examples()
    fig_certified_sweep()
    fig_unstable_and_margins()
    stats = fig_bound_width()
    with open(os.path.join(R, "bound_widths.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("figures written to", F)
    print(json.dumps(stats, indent=2))
