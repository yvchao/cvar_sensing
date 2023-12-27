import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data_util import load_synthetic_data
from model_config import net_config, sim_config
from tqdm import auto

from cvar_sensing.dataset import split_data
from cvar_sensing.sensing import Sensing
from cvar_sensing.train import dict_to_device, load_model, ndarray_to_tensor
from cvar_sensing.utils import evaluate, get_auc_scores, step_interp

parser = argparse.ArgumentParser("Run evaluation")
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()

# initialize device
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

# setup dirs
predictor_model_dir = Path("predictor")

# load data
dataset = load_synthetic_data()
net_config["x_dim"] = dataset.x_dim
net_config["y_dim"] = dataset.y_dim

with open(predictor_model_dir / "best_drop_rate", encoding="utf-8") as f:
    best_droprate = f.read().strip()

metrics = pd.read_csv(predictor_model_dir / f"drop_rate={best_droprate}.csv", index_col=0)
best_seed = metrics.roc.argmax()
predictor_idx = metrics.loc[best_seed, "model_id"]

print(f"seed of best predictor: {best_seed}")  # noqa: T201
metric = metrics.loc[best_seed]
print(f"accuracy: roc={metric.roc:.4f}, prc={metric.prc:.4f}")  # noqa: T201

train_set, test_set = split_data(dataset, seed=best_seed)

# full observation
sim_config["max_dt"] = 0.2
sim_config["min_dt"] = 0.2


def calculate_delay_oracle(batch, ret, threshold=0.3, label_index=1):
    def find_where(a):
        (indicies,) = torch.where(a)
        if len(indicies) == 0:
            return -1
        else:
            return indicies[0].item()

    def batch_find_where(arr):
        indicies = [find_where(a) for a in arr]
        return np.array(indicies)

    obs_t = ret["obs_t"]
    pred_y = ret["pred_y"]
    eval_t = torch.linspace(0, 3, 200)
    x_max = 1.0
    p = np.exp(-3 * np.square(x_max - batch["x"][:, :, 0]))  # ground truth
    eval_p = np.stack([np.interp(eval_t.cpu(), batch["t"][i], p[i]) for i in range(len(p))])
    eval_p = torch.from_numpy(eval_p)
    pred_y_int = step_interp(eval_t, obs_t, pred_y)

    indicies = batch_find_where(pred_y_int[:, :, label_index] >= threshold)
    policy_detect_t = eval_t[indicies]
    indicies = batch_find_where(eval_p[:, :] >= threshold)
    oracle_detect_t = eval_t[indicies]
    oracle_detect_t[indicies == -1] = np.nan
    delay_t = policy_detect_t - oracle_detect_t
    return delay_t


def get_performance(test_set, model):
    stats = []
    for seed in range(5):
        metric = evaluate(test_set, model, device=device, seed=seed, delay_eval=calculate_delay_oracle)
        stats.append(metric)
    stats = pd.DataFrame(stats)
    return stats


def get_fo_model(model_path: Path | str, net_config, sim_config):
    model_path = Path(model_path)
    model = Sensing(sim_config=sim_config, **net_config)
    model.predictor = load_model(model.predictor, model_path)  # directly load the predictor
    model = model.to(device)
    model.pi.full_obs = True
    return model


model = get_fo_model(predictor_model_dir / f"Predictor_{predictor_idx}.pt", net_config, sim_config)
metric = get_performance(test_set, model)
metric.to_csv("fo_evaluation.csv")

# restore the simulation settting.
sim_config["min_dt"] = 0.2
sim_config["max_dt"] = 1.0

ras_model_dir = Path("ras")


def get_saved_model(model_path: Path | str, net_config, sim_config):
    model_path = Path(model_path)

    model = Sensing(sim_config=sim_config, **net_config)
    model = load_model(model, model_path)
    model = model.to(device)
    return model


metrics = []
files = list(ras_model_dir.glob("*.csv"))
for f in auto.tqdm(files):
    lambda_ = re.findall(r"(?<==)[0-9]*[.]?[0-9]+(?=\.csv)", f.name)[0]
    lambda_ = float(lambda_)
    record = pd.read_csv(f, index_col=0)
    sim_config["lambda"] = lambda_
    for seed in record.index:
        model_id = record.loc[seed, "uuid"]
        model = get_saved_model(ras_model_dir / f"ActiveSensing_{model_id}.pt", net_config, sim_config)
        metric = get_performance(test_set, model)
        metric["lambda"] = lambda_
        metric["seed"] = seed
        metrics.append(metric)
metrics = pd.concat(metrics)
metrics.to_csv("ras_evaluation.csv")


nll_model_dir = Path("nll")

metrics = []
files = list(nll_model_dir.glob("*.csv"))
for f in auto.tqdm(files):
    lambda_ = re.findall(r"(?<==)[0-9]*[.]?[0-9]+(?=\.csv)", f.name)[0]
    lambda_ = float(lambda_)
    record = pd.read_csv(f, index_col=0)
    sim_config["lambda"] = lambda_

    for seed in record.index:
        model_id = record.loc[seed, "uuid"]
        model = get_saved_model(nll_model_dir / f"ActiveSensing_{model_id}.pt", net_config, sim_config)
        metric = get_performance(test_set, model)
        metric["lambda"] = lambda_
        metric["seed"] = seed
        metrics.append(metric)
metrics = pd.concat(metrics)
metrics.to_csv("nll_evaluation.csv")


baseline_model_dir = Path("baseline")

metrics = []
files = list(baseline_model_dir.glob("*.csv"))
for f in auto.tqdm(files):
    delta = re.findall(r"(?<==)[0-9]*[.]?[0-9]+(?=\.csv)", f.name)[0]
    delta = float(delta)
    record = pd.read_csv(f, index_col=0)
    sim_config["min_dt"] = delta
    sim_config["max_dt"] = delta

    for seed in record.index:
        model_id = record.loc[seed, "uuid"]
        model = get_saved_model(baseline_model_dir / f"ActiveSensing_{model_id}.pt", net_config, sim_config)
        metric = get_performance(test_set, model)
        metric["delta"] = delta
        metric["seed"] = seed
        metrics.append(metric)
metrics = pd.concat(metrics)
metrics.to_csv("baseline_evaluation.csv")


predictor_model_dir = Path("predictor")


def evaluate_asac_test(test_set, seed, lambda_):
    npz = np.load(f"asac/models/seed={seed},lambda={lambda_}.npz")
    cost = np.array([1, 0.1, 1.0, 1.0])
    ret = {}
    ret["obs_t"] = test_set[:]["t"]
    ret["obs_x"] = npz["testX"]
    # ret['pred_y'] = np.stack([npz['testY_hat'],1-npz['testY_hat']],axis=-1)
    ret["pred_y"] = npz["testY_hat"]
    ret["obs_m"] = npz["testG_hat"]
    ret["r_m"] = ret["obs_m"] @ cost
    ret["mask"] = test_set[:]["mask"]
    ret = ndarray_to_tensor(ret)

    roc, prc = get_auc_scores(test_set, ret)
    cost = torch.mean(torch.sum(ret["r_m"] * ret["mask"], dim=-1)).item()

    batch = test_set[:]
    batch = ndarray_to_tensor(batch)
    batch = dict_to_device(batch)

    delays = {}
    for threshold in [0.1, 0.3, 0.5, 0.7]:
        ds = calculate_delay_oracle(test_set[:], ret, threshold=threshold, label_index=1)
        delays[f"delay(p>={threshold})"] = np.nanmean(np.abs(ds)).item()

    return roc, prc, cost, delays


dfs = []

for seed in [0, 1, 2, 3, 4]:
    for lambda_ in [0.1, 0.01, 0.005, 0.001]:
        df = pd.DataFrame()
        m = "ASAC"
        roc, prc, cost, delays = evaluate_asac_test(test_set, seed, lambda_)
        df.loc[m, "roc"] = roc
        df.loc[m, "prc"] = prc
        df.loc[m, "cost"] = cost
        df.loc[m, "lambda"] = lambda_
        for k, v in delays.items():
            df.loc[m, k] = v
        df = df.rename_axis("method").reset_index()
        dfs.append(df)

metrics = pd.concat(dfs)
metrics.to_csv("asac_evaluation.csv")
