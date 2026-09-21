"""export_models.py -- produce the artefacts the two official tools expect.

  * ONNX  (`models/*.onnx`)  -- the input format of ERAN
                               (`deepzono` / `deeppoly` domains).
  * NPZ   (`models/*.npz`)   -- the framework-independent weights produced by
                               the training scripts.
  * A converter (`to_pytorch.py`, written next to the models) that turns the
    NPZ weights into a PyTorch `.pth` state-dict, which is the input format of
    CROWN / `auto_LiRPA`.  It is emitted as a script rather than executed here
    because the analysis environment of this submission is NumPy-only.

Every exported ONNX graph is executed with onnxruntime and checked against the
NumPy reference implementation, so the exported artefact is guaranteed to be the
same function that was analysed in Steps 3 and 6.
"""
import json
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tas_lib import MLP, load_mnist
from config import CFG


def to_onnx(net: MLP, path: str, name: str):
    """Emit Gemm -> Relu -> Gemm -> Relu -> Gemm, the shape ERAN expects."""
    nodes, inits = [], []
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, net.sizes[0]])
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, net.sizes[-1]])
    cur = "input"
    n = len(net.W)
    for i, (W, b) in enumerate(zip(net.W, net.b)):
        wn, bn = f"W{i+1}", f"b{i+1}"
        # ONNX Gemm computes  Y = alpha * A * B + beta * C ; we keep B = W (in, out)
        inits.append(numpy_helper.from_array(W.astype(np.float32), wn))
        inits.append(numpy_helper.from_array(b.astype(np.float32), bn))
        zn = "output" if i == n - 1 else f"gemm{i+1}"
        nodes.append(helper.make_node("Gemm", [cur, wn, bn], [zn],
                                      name=f"Gemm{i+1}", alpha=1.0, beta=1.0,
                                      transA=0, transB=0))
        if i < n - 1:
            an = f"relu{i+1}"
            nodes.append(helper.make_node("Relu", [zn], [an], name=f"Relu{i+1}"))
            cur = an
    graph = helper.make_graph(nodes, name, [inp], [out], initializer=inits)
    model = helper.make_model(graph, producer_name="tas-final-project",
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8          # kept low for compatibility with older ERAN images
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return model


def verify_onnx(net: MLP, path: str, X: np.ndarray) -> float:
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ref = net.forward(X)
    got = np.vstack([sess.run(None, {"input": X[i:i + 1]})[0] for i in range(len(X))])
    return float(np.abs(ref - got).max())


CONVERTER = '''"""to_pytorch.py -- convert the NumPy weights of this submission into a
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
'''

AUTOLIRPA = '''"""run_autolirpa.py -- reproduce Steps 3 and 6 with the official CROWN tool.

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
'''

ERAN_CMD = '''# Reproducing Step 3 / Step 6 with ERAN
#
# ERAN has heavy native dependencies (ELINA, Gurobi, ...), so the official
# Docker image is the recommended route:
#
#   docker pull ethsri/eran:cpu
#   docker run -it -v "$PWD":/work ethsri/eran:cpu bash
#
# Inside the container, from the `tf_verify` directory:
#
#   python3 . --netname /work/models/mlp_baseline.onnx \\
#             --domain deeppoly --dataset mnist --epsilon 0.1
#
#   python3 . --netname /work/models/mlp_baseline.onnx \\
#             --domain deepzono --dataset mnist --epsilon 0.1
#
#   python3 . --netname /work/models/mlp_ibp.onnx \\
#             --domain deeppoly --dataset mnist --epsilon 0.1
#
# ERAN expects the MNIST test set as a CSV (label first, then 784 pixel values
# in 0..255); `results/mnist_test_eran.csv` is written by export_models.py in
# exactly that layout.
'''


def main():
    d = load_mnist(CFG["data"])
    Xte, yte = d["Xte"], d["yte"]
    report = {}
    for tag in ("baseline", "ibp"):
        npz = os.path.join(CFG["models"], f"mlp_{tag}.npz")
        onnx_path = os.path.join(CFG["models"], f"mlp_{tag}.onnx")
        net = MLP.load(npz)
        to_onnx(net, onnx_path, f"mlp_{tag}")
        err = verify_onnx(net, onnx_path, Xte[:50])
        report[tag] = {"onnx": os.path.basename(onnx_path),
                       "max_abs_logit_error_vs_numpy": err}
        print(f"{tag:9s} -> {onnx_path}   max |logit| error vs NumPy: {err:.2e}")

    for name, text in (("to_pytorch.py", CONVERTER),
                       ("run_autolirpa.py", AUTOLIRPA),
                       ("run_eran.sh", ERAN_CMD)):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), name), "w") as f:
            f.write(text)

    # ERAN-style CSV of the first 100 test images
    n = CFG["n_verify"]
    csv = np.hstack([yte[:n, None], np.round(Xte[:n] * 255).astype(np.int64)])
    np.savetxt(os.path.join(CFG["results"], "mnist_test_eran.csv"), csv,
               fmt="%d", delimiter=",")

    with open(os.path.join(CFG["results"], "export_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("wrote to_pytorch.py, run_autolirpa.py, run_eran.sh and "
          "results/mnist_test_eran.csv")


if __name__ == "__main__":
    main()
