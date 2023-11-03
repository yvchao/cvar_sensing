import argparse
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data_util import load_synthetic_data
from model_config import net_config, sim_config, train_config

from cvar_sensing.dataset import split_data
from cvar_sensing.sensing import Sensing
from cvar_sensing.train import batch_call, fit, load_model, sampler, save_model

parser = argparse.ArgumentParser("Active Sensing Demo")
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

# initialize device
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

# setup dirs
predictor_model_dir = Path("predictor")
ras_model_dir = Path("ras")

# load data
dataset = load_synthetic_data()
net_config["x_dim"] = dataset.x_dim
net_config["y_dim"] = dataset.y_dim


with open(predictor_model_dir / "best_drop_rate", encoding="utf-8") as f:
    best_droprate = f.read()

metrics = pd.read_csv(predictor_model_dir / f"drop_rate={best_droprate}.csv", index_col=0)
best_seed = metrics.roc.argmax()
predictor_idx = metrics.loc[best_seed, "model_id"]

print(f"seed of best predictor: {best_seed}")  # noqa: T201
metric = metrics.loc[best_seed]
print(f"accuracy: roc={metric.roc:.4f}, prc={metric.prc:.4f}")  # noqa: T201

train_set, test_set = split_data(dataset, seed=best_seed)


def fit_sensing(
    train_set, net_config, train_config, predictor_idx, seed=0, model_dir=Path("sensing"), sampler_config=None
):
    if sampler_config is not None:

        def my_sampler(itr, model, dataset, device):
            return sampler(itr, model, dataset, device=device, **sampler_config)

    else:
        my_sampler = None

    model_dir = Path(model_dir)

    torch.manual_seed(seed)
    model = Sensing(sim_config=sim_config, **net_config)
    model.predictor = load_model(model.predictor, predictor_model_dir / f"Predictor_{predictor_idx}.pt")
    model = model.to(device)

    loss_weights = {"actor": 1.0, "critic": 1.0}
    params = [
        {"params": model.pi.parameters(), "lr": 0.05},
        {"params": model.critic.parameters(), "lr": 0.1},
        {"params": model.encoder.parameters(), "lr": 0.1},
        {"params": model.predictor.parameters(), "lr": 0.0},
    ]

    fit(model, train_set, loss_weights, test_set=None, params=params, sampler=my_sampler, device=device, **train_config)

    model_id = uuid.uuid1()
    model_name = f"{model.name}_{model_id}"  # save model with a randomly generated uuid
    save_model(model, path=model_dir, name=model_name)
    return model, model_id


rns_model_dir = Path("rns")
model, rns_id = fit_sensing(
    train_set, net_config, train_config, predictor_idx, model_dir=rns_model_dir, sampler_config=None, seed=0
)


def evaluate_q_dist(dataset, model_path, model_id, eval_mode=False, seed=0):
    torch.random.manual_seed(seed)
    model_path = Path(model_path)

    model = Sensing(sim_config=sim_config, **net_config)
    model = load_model(model, model_path / f"ActiveSensing_{model_id}.pt").to(device)
    if model_path.name == "baseline":
        model.max_dt = 1.0
        model.min_dt = 1.0
        model.pi.adaptive_interval = False

    if eval_mode:
        model.eval()
    else:
        model.train()

    ret = batch_call(model.simulate, dataset, device=device)

    q_vals = torch.sum(ret["cost"] * ret["mask"], dim=-1).detach().numpy()
    return q_vals


ras_id = pd.read_csv(f"./ras/lambda={sim_config['lambda']}.csv", index_col=0).loc[0, "uuid"]
baseline_id = pd.read_csv("./baseline/delta=1.0.csv", index_col=0).loc[0, "uuid"]

q_ras = []
for seed in range(5):
    q_vals = evaluate_q_dist(test_set, "ras", ras_id, seed=seed)
    q_ras.append(q_vals)
q_ras = np.stack(q_ras, axis=-1).mean(axis=-1)

q_rns = []
for seed in range(5):
    q_vals = evaluate_q_dist(test_set, "rns", rns_id, seed=seed)
    q_rns.append(q_vals)
q_rns = np.stack(q_rns, axis=-1).mean(axis=-1)

q_baseline = []
for seed in range(5):
    q_vals = evaluate_q_dist(test_set, "baseline", baseline_id, seed=seed)
    q_baseline.append(q_vals)
q_baseline = np.stack(q_baseline, axis=-1).mean(axis=-1)

np.savez_compressed("q_pi.npz", q_ras=q_ras, q_baseline=q_baseline, q_rns=q_rns)
