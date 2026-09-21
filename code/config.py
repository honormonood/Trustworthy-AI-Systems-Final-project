"""Central configuration shared by every step of the pipeline."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

CFG = {
    "seed": 1405,
    "data": os.path.join(_ROOT, "data"),
    "models": os.path.join(_ROOT, "models"),
    "results": os.path.join(_ROOT, "results"),
    "figures": os.path.join(_ROOT, "results", "figures"),

    # ---- Step 1: standard training ----
    "epochs_std": 2,
    "lr_std": 1e-3,
    "batch": 128,

    # ---- Step 2: attack ----
    "eps_grid": [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3],
    "eps": 0.1,            # the operating point epsilon* used in Steps 3-6

    # ---- Step 3/6: verification ----
    "n_verify": 100,       # first N test images (the ERAN/CROWN convention)
    "milp_time_limit": 30.0,
    "eps_verify_grid": [0.01, 0.02, 0.03, 0.05, 0.075, 0.1],

    # ---- Step 4: IBP training ----
    "epochs_ibp": 150,
    "lr_ibp": 5e-4,
    "warmup_epochs": 3,    # pure natural training before the ramp-up starts
    "ramp_epochs": 40,     # linear ramp of epsilon 0 -> eps_train, kappa 1 -> kappa_end
    "kappa_end": 0.5,
    "lr_decay_at": [0.7, 0.9],   # fractions of training at which lr is x0.2
    "eps_train_factor": 1.0,   # train at exactly the epsilon of Step 2
}
