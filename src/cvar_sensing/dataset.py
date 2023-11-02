import numpy as np
import torch


def split_data(dataset, test_split=0.2, seed=0, validation=False):
    torch.random.manual_seed(seed)

    train_set, test_set = torch.utils.data.random_split(dataset, [1 - test_split, test_split])
    if validation:
        train_set, valid_set = torch.utils.data.random_split(train_set, [1 - test_split, test_split])
        return train_set, valid_set, test_set
    else:
        return train_set, test_set


class Dataset(torch.utils.data.Dataset):
    def update_property(self, property, value):
        setattr(self, property, value)

    def __init__(self, t, x, y, mask, obs_config={}, seed=0):
        super().__init__()
        self.t = t
        self.x = x
        self.y = y
        self.mask = mask
        self.obs_config = obs_config
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

        self.sample_size = len(x)
        self.x_dim = x.shape[-1]
        self.y_dim = y.shape[-1]
        self.t_length = t.shape[-1]
        self.obs_T = t.max() - t.min()

        for k in self.obs_config.keys():
            self.update_property(k, self.obs_config[k])

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        t = self.t[index]
        mask = self.mask[index]
        x = self.x[index]
        y = self.y[index]

        alpha = 3 * np.ones(self.obs_length)

        batch_dim = len(y.shape)
        if batch_dim == 3:
            batch_size = len(y)
            dt = self.rng.dirichlet(alpha, size=(batch_size,))
            obs_t = t[:, [0]] + self.obs_T * np.cumsum(dt, axis=-1)
            obs_mask = self.rng.binomial(1, self.obs_prob, size=(batch_size, self.obs_length, self.x_dim))
        else:
            dt = self.rng.dirichlet(alpha)
            obs_t = t[0] + self.obs_T * np.cumsum(dt, axis=-1)
            obs_mask = self.rng.binomial(1, self.obs_prob, size=(self.obs_length, self.x_dim))

        ret = {
            "t": t,
            "x": x,
            "y": y,
            "mask": mask,
            "obs_t": obs_t.astype("float32"),
            "obs_mask": obs_mask.astype("float32"),
        }
        return ret
