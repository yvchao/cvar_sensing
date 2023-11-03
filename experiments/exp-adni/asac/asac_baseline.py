# ASAC
import argparse
import json
import os
import pathlib
import random
import sys

import numpy as np
import tensorflow as tf
from ASAC import ASAC

# Predictor after selection
from Predictor_G import Predictor_G
from sklearn.model_selection import train_test_split
from tqdm import auto

parser = argparse.ArgumentParser("ASAC")
parser.add_argument("--test-split", type=float, default=0.2)
parser.add_argument("--niters", type=int, default=2000)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

epochs = args.niters
seed = args.seed

tf.reset_default_graph()

os.environ["PYTHONHASHSEED"] = str(seed)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

random.seed(seed)
tf.set_random_seed(seed)
tf.random.set_random_seed(seed)
np.random.seed(seed)

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
# The results are always changing. I have no idea how to fix it currently.


# set up working directory
asac_dir = pathlib.Path("models")
asac_dir.mkdir(exist_ok=True)


def load_dataset():
    train_set_file = np.load("train_set.npz")
    test_set_file = np.load("test_set.npz")

    data_x, data_y = train_set_file["x"].astype("float32"), train_set_file["y"].astype("float32")
    data_mask = train_set_file["mask"].astype("float32")
    data_y[data_mask != 1] = 0  # deal with missing values
    data_x = np.nan_to_num(data_x)
    train_idx, valid_idx = train_test_split(
        np.arange(len(data_y)),
        test_size=args.test_split,
        shuffle=True,
        # stratify=np.nanmax(data_y[:, :, 0], axis=1),
        random_state=args.seed,
    )

    train_set = data_x[train_idx], data_y[train_idx]
    valid_set = data_x[valid_idx], data_y[valid_idx]
    data_x, data_y = test_set_file["x"].astype("float32"), test_set_file["y"].astype("float32")
    data_mask = test_set_file["mask"].astype("float32")
    data_x[data_mask != 1] = 0  # deal with missing values
    data_x = np.nan_to_num(data_x)
    test_set = data_x, data_y
    return train_set, valid_set, test_set


costs = np.array([1.0, 1.0, 0.5, 0.5])


for lambda_ in [0.1, 0.01, 0.005, 0.001]:
    train_set, valid_set, test_set = load_dataset()

    trainX, trainY = train_set
    validX, validY = valid_set
    testX, testY = test_set
    print(f"run ASAC (lambda={lambda_}) ...")
    adjusted = 0.0001 * lambda_
    trainG_hat, validG_hat, testG_hat = ASAC(trainX, trainY, validX, testX, costs, iterations=epochs, lambda_=adjusted)

    print("fit predictor with selection mask ...")

    validY_hat = Predictor_G(trainX, validX, trainY, trainG_hat, validG_hat, iterations=epochs)
    testY_hat = Predictor_G(trainX, testX, trainY, trainG_hat, testG_hat, iterations=epochs)

    results = {
        "testX": testX,
        "testY": testY,
        "testY_hat": testY_hat,
        "testG_hat": testG_hat,
        "validX": validX,
        "validY": validY,
        "validY_hat": validY_hat,
        "validG_hat": validG_hat,
    }

    np.savez_compressed(asac_dir / f"seed={args.seed},lambda={lambda_}.npz", **results)
