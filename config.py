import hashlib
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch


warnings.filterwarnings("ignore")


SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

N_GPU  = torch.cuda.device_count()
DEVICE = "cuda" if N_GPU > 0 else "cpu"
USE_AMP_OK = (N_GPU > 0)
print(f"GPUs detected: {N_GPU} | base device: {DEVICE}")
for i in range(N_GPU):
    print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")

CONFIG = {
    "data_dir":      Path("Dataset"),
    "cache_dir":     Path("./cache"),
    "target_file":   "spectra_normalized.xlsx",
    "smiles_file":   "SMILES.xlsx",
    "morgan_sizes":  [2048],
    "n_meta_target": 5,
    "n_meta_morgan": 2,

    "dedup_canonical": True,
    "clip_negative":   True,
    "chemprop_atom_filter": True,
    "dedup_spectra":      True,
    "spectra_dup_cosine": 0.999,

    "eval_mode":       "tune_eval",
    "test_frac":       0.10,
    "val_frac":        0.10,
    "n_eval_folds":    5,
    "n_eval_repeats":  1,
    "split_schemes":   ["random", "cluster"],
    "cluster_cutoff":  0.35,
    "n_outer_folds":   5,
    "inner_val_frac":  0.18,
    "selection_metric": "sid",
    "stratify_resolution": False,

    "pca_min":       16,
    "pca_max":       256,
    "xgb_pca_max":   256,

    "mlp_trials":    50,
    "rf_trials":     50,
    "cnn_trials":    50,
    "smilestransformer_trials": 50,
    "mordred_trials":    50,
    "mordred_rf_trials": 50,
    "mordred_max_nan":   0.2,
    "mpnn_trials":        50,
    "attentivefp_trials": 50,
    "gin_trials":         50,
    "graph_parallel":     False,
    "graph_gpu":          0,
    "morgan_search": False,

    "loss":              "sid",
    "output_activation": "softplus",
    "target_norm":       "sum",
    "floor":             1e-8,

    "headline_metric":   "SIS_b10",
    "grid_step_cm":      2.0,

    "max_epochs":    200,
    "patience":      20,
    "use_amp":       True,

    "use_checkpoints": True,
    "ckpt_dir":      Path("./checkpoints"),
    "seed":          SEED,
    "save_models":    True,
    "save_all_preds": True,
    "save_splits":    True,
}
CONFIG["cache_dir"].mkdir(exist_ok=True)
CONFIG["ckpt_dir"].mkdir(exist_ok=True)

for _cand in [Path("Dataset"), Path("../Dataset"), Path("../../Dataset")]:
    if (_cand / CONFIG["target_file"]).exists():
        CONFIG["data_dir"] = _cand
        break
else:
    raise FileNotFoundError("Could not find spectra_normalized.xlsx in Dataset/, ../Dataset/, ../../Dataset/.")
print("data_dir resolved to:", CONFIG["data_dir"].resolve())

def _cfg_fingerprint(cfg):
    keys = ["loss","output_activation","target_norm","dedup_canonical","clip_negative",
            "chemprop_atom_filter","floor","morgan_search","xgb_pca_max","pca_max","pca_min",
            "eval_mode","test_frac","val_frac","n_outer_folds","inner_val_frac","cluster_cutoff","seed",
            "selection_metric","n_eval_folds","stratify_resolution","dedup_spectra","spectra_dup_cosine"]
    blob = json.dumps({k: cfg[k] for k in keys}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]
CFG_HASH = _cfg_fingerprint(CONFIG)
FLOOR = CONFIG["floor"]
RUN_SEED = SEED
print("CONFIG fingerprint:", CFG_HASH, "| log-floor:", FLOOR, "| eval_mode:", CONFIG["eval_mode"])
