import numpy as np

from cvar_sensing.dataset import Dataset
from cvar_sensing.train import get_one_hot


def load_synthetic_data(obs_length=10, drop_rate=0.7):
    # load data
    file = np.load("synthetic_data.npz")
    t = file["t"].astype("float32")
    x = file["x"].astype("float32")
    y = file["y"].astype("int")[:, :, 0]
    mask = file["mask"].astype("float32")

    y = get_one_hot(y, 2)

    # prepare dataset
    data_config = {
        "obs_length": obs_length,
        "obs_prob": 1 - drop_rate,
    }

    dataset = Dataset(t, x, y, mask, data_config)
    return dataset
