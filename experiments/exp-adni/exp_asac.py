import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from data_util import load_adni_data

from cvar_sensing.dataset import split_data

# Add repo root to PYTHONPATH to ensuer we can access experiments module:
path = os.path.realpath(os.path.realpath(Path(__file__).parent.as_posix()) + "../../..")
sys.path.append(path)

from experiments.exp_config import conda_path, venv_asac

# setup dirs
predictor_model_dir = Path("predictor")
asac_model_dir = Path("asac")

# load data
dataset = load_adni_data()

with open(predictor_model_dir / "best_drop_rate", encoding="utf-8") as f:
    best_droprate = f.read()

metrics = pd.read_csv(predictor_model_dir / f"drop_rate={best_droprate}.csv", index_col=0)
best_seed = metrics.roc.argmax()
predictor_idx = metrics.loc[best_seed, "model_id"]

print(f"seed of best predictor: {best_seed}")  # noqa: T201
metric = metrics.loc[best_seed]
print(f"accuracy: roc={metric.roc:.4f}, prc={metric.prc:.4f}")  # noqa: T201

train_set, test_set = split_data(dataset, seed=best_seed)

# save the training and test data to asac dir
np.savez_compressed(asac_model_dir / "train_set", **train_set[:])
np.savez_compressed(asac_model_dir / "test_set", **test_set[:])

cmd = f"bash ./exp_asac.sh {conda_path} {venv_asac}".split(" ")
log_path = asac_model_dir / "exp_asac_sh.log"
with open(log_path, "w") as f:
    proc = subprocess.Popen(cmd, cwd=asac_model_dir, stdout=f, stderr=f)
    proc.wait()
