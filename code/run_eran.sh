# Reproducing Step 3 / Step 6 with ERAN
#
# ERAN has heavy native dependencies (ELINA, Gurobi, ...), so the official
# Docker image is the recommended route:
#
#   docker pull ethsri/eran:cpu
#   docker run -it -v "$PWD":/work ethsri/eran:cpu bash
#
# Inside the container, from the `tf_verify` directory:
#
#   python3 . --netname /work/models/mlp_baseline.onnx \
#             --domain deeppoly --dataset mnist --epsilon 0.1
#
#   python3 . --netname /work/models/mlp_baseline.onnx \
#             --domain deepzono --dataset mnist --epsilon 0.1
#
#   python3 . --netname /work/models/mlp_ibp.onnx \
#             --domain deeppoly --dataset mnist --epsilon 0.1
#
# ERAN expects the MNIST test set as a CSV (label first, then 784 pixel values
# in 0..255); `results/mnist_test_eran.csv` is written by export_models.py in
# exactly that layout.
