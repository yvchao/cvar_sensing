import torch
import torch.nn.functional as tf

from .constants import newdim
from .encoders import NCDE, take_from_sequence
from .nn import MLP
from .utils import batch_interp_nd


class Predictor(torch.nn.Module):
    def __init__(
        self, x_dim: int, y_dim: int, *, hidden_size: int, num_layer: int, fit_obs: bool = False, resolution=0.3
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.name = "Predictor"
        self.resolution = resolution
        self.fit_obs = fit_obs

        self.encoder = NCDE(
            self.x_dim,
            self.hidden_size,
            hidden_size=self.hidden_size,
            num_layer=self.num_layer,
            resolution=self.resolution,
        )
        self.mlp = MLP(self.hidden_size, self.y_dim, self.hidden_size, self.num_layer)

    def get_logits(self, t, x, *, t_eval=None, bound=5.0):
        embed = self.encoder(t, x, t_eval=t_eval)
        logits = self.mlp(embed)
        logits = torch.clamp(logits, -bound, bound)
        return logits

    def predict(self, t, x, *, t_eval=None):
        logits = self.get_logits(t, x, t_eval=t_eval)
        pred_y = torch.sigmoid(logits)
        return pred_y

    # def _bce_loss(self, pred_y, y, mask):
    #     pred_y = pred_y[mask == 1.0]
    #     y = y[mask == 1.0]
    #     loss = tf.binary_cross_entropy(pred_y, y)
    #     return loss

    def _bce_loss_logits(self, logits, y, mask):
        logits = logits[mask == 1.0]
        y = y[mask == 1.0]
        loss = tf.binary_cross_entropy_with_logits(logits, y)
        return loss

    def obtain_obs_data(self, obs_t, obs_mask, t, x):
        obs_x = batch_interp_nd(obs_t, t, x)
        obs_x[obs_mask != 1.0] = torch.nan
        return obs_x

    def forward(self, batch):
        t = batch["t"]
        x = batch["x"]
        y = batch["y"]
        mask = batch["mask"]

        logits = self.get_logits(t, x)

        losses = {}
        losses["bce"] = self._bce_loss_logits(logits, y, mask)

        if self.fit_obs:
            obs_t = batch["obs_t"]
            obs_mask = batch["obs_mask"]
            obs_x = self.obtain_obs_data(obs_t, obs_mask, t, x)
            obs_y = take_from_sequence(t[0], y, obs_t)
            obs_y_mask = take_from_sequence(t[0], mask[:, :, newdim], obs_t)[:, :, 0]
            obs_logits = self.get_logits(obs_t, obs_x)
            losses["obs_bce"] = self._bce_loss_logits(obs_logits, obs_y, obs_y_mask)
        return losses
