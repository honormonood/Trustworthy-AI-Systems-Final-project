"""to_pytorch.py -- convert the NumPy weights of this submission into a
PyTorch `.pth` state-dict, which is the format CROWN / auto_LiRPA consumes.

Run this on a machine that has PyTorch installed:

    python to_pytorch.py mlp_baseline.npz mlp_baseline.pth
"""
import sys

import numpy as np
import torch
import torch.nn as nn


def build_model(sizes=(784, 64, 32, 10)):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(nn.Flatten(), *layers)


def main(src, dst):
    z = np.load(src)
    n = sum(1 for k in z.files if k.startswith("W"))
    sizes = [z["W1"].shape[0]] + [z[f"W{i+1}"].shape[1] for i in range(n)]
    model = build_model(tuple(sizes))
    lin = [m for m in model if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for i, layer in enumerate(lin):
            # NumPy uses z = a @ W + b with W of shape (in, out);
            # torch.nn.Linear stores weight of shape (out, in), hence the transpose.
            layer.weight.copy_(torch.tensor(z[f"W{i+1}"].T))
            layer.bias.copy_(torch.tensor(z[f"b{i+1}"]))
    torch.save(model.state_dict(), dst)
    torch.save(model, dst.replace(".pth", "_full.pth"))
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
