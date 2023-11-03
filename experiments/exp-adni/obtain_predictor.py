import argparse
import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from data_util import load_adni_data, split_data
from model_config import net_config, train_config

from cvar_sensing.predictor import Predictor, take_from_sequence
from cvar_sensing.train import (
    batch_call,
    fit,
    get_auc_scores,
    load_model,
    predictor_evaluation,
    save_model,
)

parser = argparse.ArgumentParser("Active Sensing Demo")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--epochs", type=int, default=200)
args = parser.parse_args()

# initialize device
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
epochs = args.epochs

# setup dirs
predictor_model_dir = Path("predictor")


# fit the predictor with different data splits
def fit_predictor(dataset, net_config, model_dir=Path("predictor"), seed=0):
    model_dir = Path(model_dir)
    # load dataset
    train_set, valid_set, test_set = split_data(dataset, seed=seed, validation=True)

    torch.manual_seed(seed)
    predictor = Predictor(fit_obs=True, **net_config).to(device)
    loss_weights = {"bce": 1.0, "obs_bce": 1.0}
    fit(
        predictor,
        train_set,
        loss_weights,
        test_set=valid_set,
        test_metric="bce",
        epochs=epochs,
        tolerance=7,
        device=device,
        **train_config,
    )

    model_id = uuid.uuid1()
    model_name = f"{predictor.name}_{model_id}"
    save_model(predictor, path=model_dir, name=model_name)
    predictor = load_model(predictor, model_dir / f"{model_name}.pt").to(device)
    roc, prc = predictor_evaluation(predictor, test_set, train_config["batch_size"], device=device)
    performance = {"roc": roc, "prc": prc, "model_id": model_id}
    return predictor, performance


# search for the best predictor
metrics = pd.DataFrame(columns=["drop_rate", "roc", "prc", "model_id"])

for drop_rate in [0.0, 0.3, 0.5, 0.7]:
    # load data
    dataset = load_adni_data(drop_rate=drop_rate)
    net_config["x_dim"] = dataset.x_dim
    net_config["y_dim"] = dataset.y_dim

    for seed in range(5):
        predictor, metric = fit_predictor(dataset, net_config, model_dir=predictor_model_dir, seed=seed)
        metrics.loc[seed, ["roc", "prc", "model_id"]] = metric
        metrics.loc[seed, "drop_rate"] = drop_rate
    metrics.to_csv(predictor_model_dir / f"drop_rate={drop_rate}.csv")


def evaluation_dropout(model, dataset, batch_size=100, device=None, eval_mode=True):
    if eval_mode:
        model.eval()
    else:
        model.train()

    def func(batch):
        t = batch["t"]
        x = batch["x"]
        y = batch["y"]
        mask = batch["mask"]

        obs_t = batch["obs_t"]
        obs_mask = batch["obs_mask"]
        obs_x = model.obtain_obs_data(obs_t, obs_mask, t, x)
        obs_y = take_from_sequence(t[0], y, obs_t)
        obs_y_mask = take_from_sequence(t[0], mask[:, :, None], obs_t)[:, :, 0]

        pred_y = model.predict(obs_t, obs_x)

        ret = {}
        ret["true_y"] = obs_y
        ret["pred_y"] = pred_y
        ret["mask"] = obs_y_mask
        return ret

    with torch.no_grad():
        ret = batch_call(func, dataset, device=device, batch_size=batch_size)

    roc, auc = get_auc_scores(ret)
    return roc, auc


# we evaluate the predictor performance with data drop rate of 0.7
dataset = load_adni_data(drop_rate=0.7)

performance = []
for f in predictor_model_dir.glob("*.csv"):
    drop_rate = re.findall(r"(?<==)[0-9]*[.]?[0-9]+(?=\.csv)", f.name)[0]
    metrics = pd.read_csv(f, index_col=0)
    best_seed = metrics.roc.argmax()
    best_model = metrics.loc[best_seed, "model_id"]

    train_set, valid_set, test_set = split_data(dataset, seed=best_seed, validation=True)
    predictor = Predictor(fit_obs=True, **net_config)
    predictor = load_model(predictor, predictor_model_dir / f"Predictor_{best_model}.pt").to(device)
    rocs = []
    for i in range(5):
        roc, prc = evaluation_dropout(predictor, test_set, train_config["batch_size"], device=device)
        rocs.append(roc)
    roc = np.mean(rocs)
    performance.append({"drop_rate": drop_rate, "roc": roc})

performance = pd.DataFrame(performance)
best_idx = performance.roc.argmax()
best_droprate = performance.loc[best_idx, "drop_rate"]
print(f"best drop rate: {best_droprate}")  # noqa: T201
print(performance)  # noqa: T201

metrics = pd.read_csv(predictor_model_dir / f"drop_rate={best_droprate}.csv", index_col=0)
best_seed = metrics.roc.argmax()
best_model = metrics.loc[best_seed, "model_id"]
print(f"seed of best predictor: {best_seed}")  # noqa: T201
metric = metrics.loc[best_seed]
print(f"accuracy: roc={metric.roc:.4f}, prc={metric.prc:.4f}")  # noqa: T201

with open(predictor_model_dir / "best_drop_rate", "w", encoding="utf-8") as f:
    f.write(f"{best_droprate}")
