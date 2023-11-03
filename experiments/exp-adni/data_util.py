import numpy as np

from cvar_sensing.dataset import Dataset
from cvar_sensing.train import get_one_hot


def load_adni_data(obs_length=10, drop_rate=0.7):
    # load data
    file = np.load("data_with_orig.npz")
    t = file["t"].astype("float32")
    x = file["x"].astype("float32")
    mask = file["mask"].astype("float32")

    y = file["y"]
    y[mask != 1] = -1
    y = y.astype("int")[:, :, 0]

    # pick four features
    features = ["FDG", "AV45", "Hippocampus", "Entorhinal"]
    sel_mask = np.in1d(file["feat_list"], features)
    x = x[:, :, sel_mask]

    y = get_one_hot(y, 3)
    y[mask != 1, :] = np.nan

    # prepare dataset
    data_config = {
        "obs_length": obs_length,
        "obs_prob": 1 - drop_rate,
    }

    dataset = Dataset(t, x, y, mask, data_config)
    return dataset
