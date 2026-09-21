"""run_autolirpa.py -- reproduce Steps 3 and 6 with the official CROWN tool.

    pip install torch auto_LiRPA
    python to_pytorch.py ../models/mlp_baseline.npz mlp_baseline.pth
    python run_autolirpa.py mlp_baseline.pth 0.1

This mirrors `verifiers.verify_deeppoly` exactly: `method="backward"` is CROWN,
`method="IBP"` is the box domain, and `method="CROWN-IBP"` is the hybrid.
The certified quantity is the same margin specification
    min_{j != c} ( z_c - z_j ) > 0
encoded through the specification matrix C.
"""
import sys

import numpy as np
import torch
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

from to_pytorch import build_model


def main(pth, eps, n=100):
    from torchvision import datasets, transforms
    model = build_model()
    model.load_state_dict(torch.load(pth, map_location="cpu"))
    model.eval()

    ds = datasets.MNIST(".", train=False, download=True,
                        transform=transforms.ToTensor())
    X = torch.stack([ds[i][0] for i in range(n)]).view(n, -1)
    y = torch.tensor([ds[i][1] for i in range(n)])

    lirpa = BoundedModule(model, torch.empty_like(X))
    ptb = PerturbationLpNorm(norm=np.inf, eps=eps, x_L=(X - eps).clamp(0, 1),
                             x_U=(X + eps).clamp(0, 1))
    bx = BoundedTensor(X, ptb)

    # C[i, j, :] = e_{y_i} - e_{j}  for the nine wrong classes j
    C = torch.zeros(n, 9, 10)
    for i in range(n):
        others = [j for j in range(10) if j != y[i].item()]
        for r, j in enumerate(others):
            C[i, r, y[i]] = 1.0
            C[i, r, j] = -1.0

    correct = (model(X).argmax(1) == y)
    for method in ("IBP", "CROWN-IBP", "backward"):
        lb, _ = lirpa.compute_bounds(x=(bx,), method=method, C=C)
        verified = (lb.min(1).values > 0) & correct
        print(f"{method:10s} certified accuracy: {verified.float().mean()*100:.2f}%")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))
