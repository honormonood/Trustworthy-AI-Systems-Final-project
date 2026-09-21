"""
verifiers.py -- Formal verification back-ends for the 784-64-32-10 ReLU MLP.

Four sound analyses of increasing precision are provided, plus a complete one:

  1. `verify_interval`  -- Interval Bound Propagation (the box domain).
  2. `verify_deepzono`  -- Zonotope domain, i.e. the DeepZ transformer used by
                           ERAN's `deepzono` option (Singh et al., NeurIPS'18).
  3. `verify_deeppoly`  -- Backward linear relaxation with back-substitution.
                           For feed-forward ReLU networks this abstract domain
                           is exactly ERAN's `deeppoly` and is algorithmically
                           identical to CROWN / auto_LiRPA's `backward` mode;
                           the two differ only in the heuristic used to pick the
                           lower ReLU slope, which is exposed as `lam_rule`.
  4. `verify_milp`      -- Complete (exact) verification via the big-M MILP
                           encoding of Tjeng et al. (ICLR'19), solved with
                           HiGHS through `scipy.optimize.milp`.

All analyses certify the *margin* specification
        min_{j != c}  ( z_c(x') - z_j(x') )  >  0     for all x' in B_eps(x),
which is the property "the classification of x cannot be changed".

Status codes returned by every verifier:
    "ROBUST"    -- proven robust (sound certificate)
    "UNKNOWN"   -- the incomplete analysis could not decide
    "FALSIFIED" -- a concrete counterexample exists (only MILP / attacks)
    "MISCLF"    -- the clean input is already misclassified
    "TIMEOUT"   -- solver limit reached (MILP only)
"""

from __future__ import annotations

import time

import numpy as np

from tas_lib import MLP, input_box, ibp_all_preactivation_bounds

_EPS = 1e-12


# --------------------------------------------------------------------------- #
#  Specification matrices
# --------------------------------------------------------------------------- #
def margin_matrix(y: np.ndarray, n_cls: int = 10) -> np.ndarray:
    """C of shape (B, n_cls-1, n_cls) with rows  e_c - e_j  for every j != c."""
    B = len(y)
    C = np.zeros((B, n_cls - 1, n_cls), dtype=np.float32)
    for b, c in enumerate(y):
        others = [j for j in range(n_cls) if j != c]
        C[b, np.arange(n_cls - 1), others] = -1.0
        C[b, :, c] = 1.0
    return C


# --------------------------------------------------------------------------- #
#  1. Interval / IBP verifier
# --------------------------------------------------------------------------- #
def verify_interval(net: MLP, X, y, eps):
    """Box domain.  Margins are bounded as l_c - u_j (the loosest sound rule)."""
    l0, u0 = input_box(X, eps)
    pre = ibp_all_preactivation_bounds(net, l0, u0)
    l3, u3 = pre[-1]
    B = len(y)
    idx = np.arange(B)
    lc = l3[idx, y][:, None]
    uj = u3.copy()
    uj[idx, y] = -np.inf
    margins = lc - uj                       # (B, 10), true-class entry = +inf
    margins[idx, y] = np.inf
    return margins.min(1), pre


# --------------------------------------------------------------------------- #
#  2. DeepZono (zonotope) verifier
# --------------------------------------------------------------------------- #
def _zono_relu(c, G, l, u):
    """DeepZ ReLU transformer: y = lam*x + mu + mu*eps_new for crossing units."""
    n = len(c)
    cross = (l < 0) & (u > 0)
    active = l >= 0
    lam = np.zeros(n)
    lam[active] = 1.0
    denom = np.where(cross, u - l, 1.0)
    lam = np.where(cross, u / np.maximum(denom, _EPS), lam)
    mu = np.zeros(n)
    mu[cross] = -lam[cross] * l[cross] / 2.0
    c_new = lam * c + mu
    G_new = lam[None, :] * G
    k = int(cross.sum())
    if k:
        extra = np.zeros((k, n))
        extra[np.arange(k), np.where(cross)[0]] = mu[cross]
        G_new = np.vstack([G_new, extra])
    return c_new, G_new


def verify_deepzono(net: MLP, X, y, eps):
    """Per-sample zonotope propagation; returns the certified margin lower bound."""
    l0, u0 = input_box(X, eps)
    box = ibp_all_preactivation_bounds(net, l0, u0)
    out = np.empty(len(y))
    for b in range(len(y)):
        c0 = 0.5 * (l0[b] + u0[b])
        r0 = 0.5 * (u0[b] - l0[b])
        # First affine layer applied directly to the diagonal input generators.
        c = c0 @ net.W[0] + net.b[0]
        G = r0[:, None] * net.W[0]
        for i in range(1, len(net.W)):
            rad = np.abs(G).sum(0)
            # intersect the zonotope's concretisation with the box domain
            lz = np.maximum(c - rad, box[i - 1][0][b])
            uz = np.minimum(c + rad, box[i - 1][1][b])
            c, G = _zono_relu(c, G, lz, uz)
            c = c @ net.W[i] + net.b[i]
            G = G @ net.W[i]
        # Margins are evaluated *inside* the domain, sharing noise symbols.
        cl = y[b]
        best = np.inf
        l3, u3 = box[-1][0][b], box[-1][1][b]
        for j in range(c.shape[0]):
            if j == cl:
                continue
            cm = c[cl] - c[j]
            gm = G[:, cl] - G[:, j]
            best = min(best, max(cm - np.abs(gm).sum(), l3[cl] - u3[j]))
        out[b] = best
    return out


# --------------------------------------------------------------------------- #
#  3. DeepPoly / CROWN verifier
# --------------------------------------------------------------------------- #
def _relu_relaxation(l, u, lam_rule="adaptive"):
    """Return (lam, au, bu): lower slope, upper slope, upper intercept.

    Upper:  relu(z) <= au * z + bu   with au = u/(u-l), bu = -u*l/(u-l) >= 0
    Lower:  relu(z) >= lam * z,      lam in [0, 1]
    """
    l = np.asarray(l, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    au = np.zeros_like(l)
    bu = np.zeros_like(l)
    lam = np.zeros_like(l)

    active = l >= 0
    cross = (l < 0) & (u > 0)
    au[active] = 1.0
    lam[active] = 1.0
    d = np.where(cross, u - l, 1.0)
    au = np.where(cross, u / np.maximum(d, _EPS), au)
    bu = np.where(cross, -u * l / np.maximum(d, _EPS), bu)
    if lam_rule == "adaptive":          # DeepPoly / CROWN-Ada: minimise area
        lam = np.where(cross, (u + l >= 0).astype(np.float64), lam)
    elif lam_rule == "zero":            # always the x-axis
        lam = np.where(cross, 0.0, lam)
    elif lam_rule == "one":
        lam = np.where(cross, 1.0, lam)
    elif lam_rule == "parallel":        # lam = au  (Fast-Lin / CROWN-Lin)
        lam = np.where(cross, au, lam)
    else:
        raise ValueError(lam_rule)
    return lam, au, bu


def _backsubstitute(net, l0, u0, C, pre, k, mode, lam_rule):
    """Bound C @ z_k (pre-activation of layer k) by back-substitution to the input.

    C: (B, S, n_k). Returns (B, S) lower or upper bounds.
    """
    A = C.astype(np.float64)
    d = A @ net.b[k].astype(np.float64)
    A = A @ net.W[k].T.astype(np.float64)
    for i in range(k - 1, -1, -1):
        l, u = pre[i]
        lam, au, bu = _relu_relaxation(l, u, lam_rule)
        lam, au, bu = lam[:, None, :], au[:, None, :], bu[:, None, :]
        pos, neg = np.maximum(A, 0.0), np.minimum(A, 0.0)
        if mode == "lower":
            A = pos * lam + neg * au
            d = d + (neg * bu).sum(-1)
        else:
            A = pos * au + neg * lam
            d = d + (pos * bu).sum(-1)
        d = d + A @ net.b[i].astype(np.float64)
        A = A @ net.W[i].T.astype(np.float64)
    pos, neg = np.maximum(A, 0.0), np.minimum(A, 0.0)
    if mode == "lower":
        return (pos * l0[:, None, :] + neg * u0[:, None, :]).sum(-1) + d
    return (pos * u0[:, None, :] + neg * l0[:, None, :]).sum(-1) + d


def deeppoly_preactivation_bounds(net: MLP, l0, u0, lam_rule="adaptive",
                                  intersect_box=True):
    """Pre-activation bounds for every layer, each obtained by back-substitution.

    Following the reference implementations (ERAN's DeepPoly and auto_LiRPA's
    CROWN), the linear relaxation is intersected with the concrete box domain at
    every layer.  This is necessary because the lower ReLU relaxation lam*z can
    fall below the trivial bound 0, so the two domains are not comparable a
    priori; keeping the tighter of the two is sound and never worse.
    """
    B = len(l0)
    pre = []
    bl, bu = l0, u0                       # running post-activation box bounds
    for k in range(len(net.W)):
        n_k = net.W[k].shape[1]
        I = np.broadcast_to(np.eye(n_k), (B, n_k, n_k))
        lo = _backsubstitute(net, l0, u0, I, pre, k, "lower", lam_rule)
        hi = _backsubstitute(net, l0, u0, I, pre, k, "upper", lam_rule)
        if intersect_box:
            mu, r = 0.5 * (bl + bu), 0.5 * (bu - bl)
            m = mu @ net.W[k] + net.b[k]
            rad = r @ np.abs(net.W[k])
            lo = np.maximum(lo, m - rad)
            hi = np.minimum(hi, m + rad)
        pre.append((lo, hi))
        bl, bu = np.maximum(lo, 0.0), np.maximum(hi, 0.0)
    return pre


def verify_deeppoly(net: MLP, X, y, eps, lam_rule="adaptive", return_pre=False,
                    intersect_box=True):
    """Certified lower bound on the worst-case margin (DeepPoly == CROWN here)."""
    l0, u0 = input_box(X, eps)
    pre = deeppoly_preactivation_bounds(net, l0, u0, lam_rule, intersect_box)
    C = margin_matrix(y, net.W[-1].shape[1])
    k = len(net.W) - 1
    lo = _backsubstitute(net, l0, u0, C, pre[:k], k, "lower", lam_rule).min(1)
    if intersect_box:
        idx = np.arange(len(y))
        l3, u3 = pre[-1]
        uj = u3.copy()
        uj[idx, y] = -np.inf
        box_margin = (l3[idx, y][:, None] - uj)
        box_margin[idx, y] = np.inf
        lo = np.maximum(lo, box_margin.min(1))
    return (lo, pre) if return_pre else lo


# --------------------------------------------------------------------------- #
#  4. Complete MILP verifier (big-M, solved by HiGHS)
# --------------------------------------------------------------------------- #
def _milp_margin(net, x_l, x_u, pre, c_true, j_target, time_limit=30.0,
                 mode="optimise"):
    """Exactly minimise z_c - z_j over the input box.  Returns (value, x*, status)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix, lil_matrix

    W1, W2, W3 = [w.astype(np.float64) for w in net.W]
    b1, b2, b3 = [b.astype(np.float64) for b in net.b]
    n0, n1, n2 = W1.shape[0], W1.shape[1], W2.shape[1]
    (l1, u1), (l2, u2) = pre[0], pre[1]

    cross1 = np.where((l1 < 0) & (u1 > 0))[0]
    cross2 = np.where((l2 < 0) & (u2 > 0))[0]
    # variable layout: x | z1 | a1 | z2 | a2 | d1 | d2
    o_x, o_z1, o_a1 = 0, n0, n0 + n1
    o_z2, o_a2 = n0 + 2 * n1, n0 + 2 * n1 + n2
    o_d1 = n0 + 2 * n1 + 2 * n2
    o_d2 = o_d1 + len(cross1)
    nvar = o_d2 + len(cross2)

    rows, lo_c, hi_c = [], [], []

    def add(row, lo, hi):
        rows.append(row); lo_c.append(lo); hi_c.append(hi)

    # affine layers (equalities)
    A_eq1 = lil_matrix((n1, nvar))
    A_eq1[:, o_z1:o_z1 + n1] = np.eye(n1)
    A_eq1[:, o_x:o_x + n0] = -W1.T
    A_eq2 = lil_matrix((n2, nvar))
    A_eq2[:, o_z2:o_z2 + n2] = np.eye(n2)
    A_eq2[:, o_a1:o_a1 + n1] = -W2.T

    cons = [LinearConstraint(csr_matrix(A_eq1), b1, b1),
            LinearConstraint(csr_matrix(A_eq2), b2, b2)]

    # ReLU encodings
    for (zoff, aoff, doff, l, u, cross, n) in (
            (o_z1, o_a1, o_d1, l1, u1, cross1, n1),
            (o_z2, o_a2, o_d2, l2, u2, cross2, n2)):
        m = len(cross)
        stable_act = np.where(l >= 0)[0]
        if len(stable_act):                       # a = z
            A = lil_matrix((len(stable_act), nvar))
            for r, k in enumerate(stable_act):
                A[r, aoff + k] = 1.0
                A[r, zoff + k] = -1.0
            cons.append(LinearConstraint(csr_matrix(A), 0.0, 0.0))
        if m:
            A1 = lil_matrix((m, nvar))            # a - z >= 0
            A2 = lil_matrix((m, nvar))            # a - u*delta <= 0
            A3 = lil_matrix((m, nvar))            # a - z - l*delta <= -l
            for r, k in enumerate(cross):
                A1[r, aoff + k] = 1.0; A1[r, zoff + k] = -1.0
                A2[r, aoff + k] = 1.0; A2[r, doff + r] = -u[k]
                A3[r, aoff + k] = 1.0; A3[r, zoff + k] = -1.0; A3[r, doff + r] = -l[k]
            cons.append(LinearConstraint(csr_matrix(A1), 0.0, np.inf))
            cons.append(LinearConstraint(csr_matrix(A2), -np.inf, 0.0))
            cons.append(LinearConstraint(csr_matrix(A3), -np.inf, -l[cross]))

    lb = np.concatenate([x_l, l1, np.maximum(l1, 0), l2, np.maximum(l2, 0),
                         np.zeros(len(cross1) + len(cross2))])
    ub = np.concatenate([x_u, u1, np.maximum(u1, 0), u2, np.maximum(u2, 0),
                         np.ones(len(cross1) + len(cross2))])
    integrality = np.zeros(nvar)
    integrality[o_d1:] = 1

    obj = np.zeros(nvar)
    obj[o_a2:o_a2 + n2] = W3[:, c_true] - W3[:, j_target]
    const = b3[c_true] - b3[j_target]

    if mode == "decide":
        # Pure feasibility: does there exist x' in the box with z_j >= z_c ?
        # INFEASIBLE  =>  the property holds for this target class (robust).
        # FEASIBLE    =>  x' is a genuine counterexample.
        # This is markedly faster than minimising the margin, because the
        # solver may stop as soon as infeasibility is proven and never has to
        # close an optimality gap.
        A = csr_matrix(obj.reshape(1, -1))
        cons.append(LinearConstraint(A, -np.inf, -const))
        res = milp(c=np.zeros(nvar), constraints=cons, integrality=integrality,
                   bounds=Bounds(lb, ub),
                   options={"time_limit": time_limit, "presolve": True})
        if res.status == 2:                      # infeasible
            return np.inf, None, "ROBUST"
        if res.status == 0:                      # feasible point found
            return -0.0, res.x[o_x:o_x + n0], "FALSIFIED"
        return None, None, "TIMEOUT"

    res = milp(c=obj, constraints=cons, integrality=integrality,
               bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "presolve": True,
                        "mip_rel_gap": 0.0})
    if res.status == 0:
        return res.fun + const, res.x[o_x:o_x + n0], "OPTIMAL"
    if res.status == 1:
        # time limit: use the best available bound if one was produced
        return -np.inf, None, "TIMEOUT"
    return None, None, f"STATUS_{res.status}"


def verify_milp(net: MLP, X, y, eps, pre=None, prefilter=None,
                time_limit=30.0, verbose=False, mode="decide"):
    """Complete verification of the margin property for a batch of inputs."""
    l0, u0 = input_box(X, eps)
    if pre is None:
        pre = deeppoly_preactivation_bounds(net, l0, u0)
    n_cls = net.W[-1].shape[1]
    results = []
    for b in range(len(y)):
        t0 = time.time()
        pre_b = [(pre[i][0][b], pre[i][1][b]) for i in range(len(pre))]
        clean = net.forward(X[b:b + 1]).argmax(1)[0]
        if clean != y[b]:
            results.append({"status": "MISCLF", "margin": np.nan,
                            "cex": None, "time": 0.0})
            continue
        if prefilter is not None and prefilter[b] > 0:
            results.append({"status": "ROBUST", "margin": float(prefilter[b]),
                            "cex": None, "time": 0.0, "by": "prefilter"})
            continue
        worst, cex, status = np.inf, None, "ROBUST"
        for j in range(n_cls):
            if j == y[b]:
                continue
            val, xstar, st = _milp_margin(net, l0[b], u0[b], pre_b, y[b], j,
                                          time_limit=time_limit, mode=mode)
            if st == "TIMEOUT":
                status = "TIMEOUT"
                break
            if mode == "decide":
                if st == "FALSIFIED":
                    status, cex = "FALSIFIED", xstar
                    break
            else:
                worst = min(worst, val)
                if val < 0:
                    status, cex = "FALSIFIED", xstar
                    break
        results.append({"status": status, "margin": float(worst) if np.isfinite(worst) else np.nan,
                        "cex": cex, "time": time.time() - t0})
        if verbose:
            print(f"  [{b:3d}] {results[-1]['status']:10s} "
                  f"margin={results[-1]['margin']:+.4f}  ({results[-1]['time']:.1f}s)")
    return results


# --------------------------------------------------------------------------- #
#  Utility: count unstable (crossing) ReLUs -- a proxy for verification hardness
# --------------------------------------------------------------------------- #
def unstable_counts(pre):
    tot = None
    per_layer = []
    for (l, u) in pre[:-1]:
        n = ((l < 0) & (u > 0)).sum(1)
        per_layer.append(n)
        tot = n if tot is None else tot + n
    return tot, per_layer
