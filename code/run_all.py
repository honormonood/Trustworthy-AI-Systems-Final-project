"""run_all.py -- reproduce the entire submission end to end.

    python run_all.py

Approximate wall-clock cost on a single CPU core (no GPU):
    Step 1  train baseline .................   1 s
    Step 2  FGSM attack + sweep ............  20 s
    Step 3  verify baseline ................  60 s   (incl. one 30 s MILP timeout)
    Step 4  IBP certified training .........  170 s
    Step 5  FGSM on the certified model ....  25 s
    Step 6  verify certified model .........  15 s
    export + figures .......................  15 s
"""
import time

import step1_train_baseline
import step2_fgsm_attack
import step3_verify_baseline
import step4_train_ibp
import step5_fgsm_on_ibp
import export_models
import make_figures

STEPS = [
    ("Step 1  baseline training", lambda: step1_train_baseline.main()),
    ("Step 2  FGSM attack", lambda: step2_fgsm_attack.main()),
    ("Step 3  verify baseline", lambda: step3_verify_baseline.main("baseline")),
    ("Step 4  IBP certified training", lambda: step4_train_ibp.main()),
    ("Step 5  FGSM on certified model", lambda: step5_fgsm_on_ibp.main()),
    ("Step 6  verify certified model", lambda: step3_verify_baseline.main("ibp")),
    ("export models (ONNX + tool scripts)", lambda: export_models.main()),
]

if __name__ == "__main__":
    t0 = time.time()
    for name, fn in STEPS:
        print("\n" + "=" * 78)
        print(f"### {name}")
        print("=" * 78)
        fn()
    print("\n" + "=" * 78 + "\n### figures\n" + "=" * 78)
    make_figures.fig_training()
    make_figures.fig_attack_sweep()
    make_figures.fig_examples()
    make_figures.fig_certified_sweep()
    make_figures.fig_unstable_and_margins()
    make_figures.fig_bound_width()
    print(f"\nall done in {time.time() - t0:.0f} s")
    print("build the report with:  cd ../report && pdflatex report.tex")
