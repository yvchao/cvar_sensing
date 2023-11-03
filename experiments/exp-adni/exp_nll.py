import argparse
import uuid
from pathlib import Path

import pandas as pd
import torch
from data_util import load_adni_data
from model_config import net_config, sim_config, train_config

from cvar_sensing.dataset import split_data
from cvar_sensing.sensing import Sensing
from cvar_sensing.train import fit, load_model, sampler, save_model

parser = argparse.ArgumentParser("Active Sensing Demo")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--coeff", type=float, default=300)
args = parser.parse_args()

# initialize device
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
lambda_ = args.coeff
epochs = args.epochs

# setup dirs
predictor_model_dir = Path("predictor")
nll_model_dir = Path("nll")

# load data
dataset = load_adni_data()
net_config["x_dim"] = dataset.x_dim
net_config["y_dim"] = dataset.y_dim

# non-risk-aversion
sampler_config = None

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
    model = Sensing(sim_config=sim_config, nll_reward=True, **net_config)  # use nll reward
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


sim_config["lambda"] = lambda_
train_config["epochs"] = epochs

record = pd.DataFrame(columns=["lambda", "uuid"])

for seed in range(3):
    model, model_id = fit_sensing(
        train_set,
        net_config,
        train_config,
        predictor_idx,
        model_dir=nll_model_dir,
        sampler_config=sampler_config,
        seed=seed,
    )

    record.loc[seed, "lambda"] = sim_config["lambda"]
    record.loc[seed, "uuid"] = model_id
record.to_csv(nll_model_dir / f"lambda={lambda_}.csv")
