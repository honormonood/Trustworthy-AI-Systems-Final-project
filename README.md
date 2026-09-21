# Trustworthy AI Systems — Final Project

Adversarial attacks, formal verification and certified training of a
`784 → 64 → 32 → 10` fully connected ReLU classifier on MNIST.

 
---

## Headline results (ε = 0.1, ℓ∞)

| | baseline (Step 1) | IBP-trained (Step 4) |
|---|---|---|
| clean test accuracy | **95.77 %** | **95.63 %** |
| accuracy under FGSM | 8.92 % | **88.02 %** |
| accuracy under PGD-50 | 3.13 % | **86.28 %** |
| certified — Interval / IBP | 0 % | 82 % |
| certified — DeepZono | 0 % | 82 % |
| certified — DeepPoly / CROWN | 0 % | 82 % |
| **complete (MILP) verified** | **0 %** | **82 %** |
| falsified with an explicit counterexample | 98 / 100 | 17 / 100 |
| unstable ReLUs per input (of 96) | 81.7 | 12.8 |

Of the **8685** test images that FGSM flipped on the baseline, the certified
model classifies **7828 (90.13 %)** correctly even when the attack is
re-crafted against it.

---

## Layout

```
code/     all source (pure NumPy — no deep-learning framework required)
data/     the four original MNIST IDX files (MD5-verified)
models/   mlp_baseline.{npz,onnx}   mlp_ibp.{npz,onnx}
results/  step*.json, fooled_samples.npz, mnist_test_eran.csv, figures/
report/   report.tex and TAS_Final_Project_Report.pdf
```

### Source files

| file | contents |
|---|---|
| `code/tas_lib.py` | MNIST loading, MLP, manual backprop, Adam, FGSM, PGD, IBP forward pass and its analytic gradients |
| `code/verifiers.py` | four verifiers: interval, DeepZono (zonotope), DeepPoly/CROWN, complete MILP |
| `code/verification_driver.py` | the combined analysis pipeline shared by Steps 3 and 6 |
| `code/step1…step6_*.py` | the six steps of the assignment, in order |
| `code/run_all.py` | runs everything end to end (~7 min, 1 CPU core) |
| `code/export_models.py` | ONNX export + onnxruntime validation + tool driver scripts |
| `code/to_pytorch.py` | converts the NumPy weights into a PyTorch `.pth` state-dict |
| `code/run_autolirpa.py` | reproduces Steps 3/6 with the official CROWN (`auto_LiRPA`) tool |
| `code/run_eran.sh` | the ERAN commands for `deeppoly` / `deepzono` on the ONNX models |
| `code/make_figures.py` | every figure in the report |

---

## Reproducing

```bash
pip install -r code/requirements.txt
cd code && python run_all.py            # ~7 minutes, no GPU needed
cd ../report && pdflatex report.tex     # twice, for cross-references
```

Everything is seeded (`seed = 1405`, set in `code/config.py`, where all
hyper-parameters live).

### Environment actually used

Ubuntu 24.04 (kernel 6.18), single CPU core, **no GPU**.
Python 3.12.3 · NumPy 2.4.4 · SciPy 1.17.1 · Matplotlib 3.10.8 ·
ONNX 1.22.0 · ONNX Runtime 1.24.4.

---


```bash
# CROWN / auto_LiRPA
pip install torch auto_LiRPA
python to_pytorch.py ../models/mlp_baseline.npz mlp_baseline.pth
python run_autolirpa.py mlp_baseline.pth 0.1

# ERAN
docker pull ethsri/eran:cpu
bash run_eran.sh        # (contains the exact commands; see the file)
```

The exported ONNX graphs were executed with onnxruntime and agree with the
analysed NumPy network to within `1.2e-5` (`results/export_check.json`).

---

## Known limitation

One baseline input (test index 82) is **undecided** at ε = 0.1 within the 30 s
MILP budget: DeepPoly cannot certify it, and a strong targeted PGD attack
(30 restarts × 400 steps × 9 target classes) cannot push its margin below
+2.41, so it is most likely robust. It is reported honestly as `TIMEOUT`
rather than counted either way; since it is a single input, the conclusion
(certified accuracy ∈ [0 %, 1 %]) is unaffected.
