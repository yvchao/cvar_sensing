# from timeit import default_timer as timer

import torch

from .constants import EPS
from .encoders import NCDE, RNN, obtain_interpolator
from .nn import MLP
from .policy import Policy
from .predictor import Predictor
from .utils import batch_interp_nd

# from loguru import logger


def plogp_q(p, q):
    return p * (torch.log(p + EPS) - torch.log(q + EPS))


def d_kl(p, q):
    # p,q: ... x y_dim for categorical distribution
    # d: ...
    d = torch.sum(plogp_q(p, q), dim=-1) + torch.sum(plogp_q(1 - p, 1 - q), dim=-1)
    return d


def d_js(p, q):
    m = 0.5 * (p + q)
    d = 0.5 * (d_kl(p, m) + d_kl(q, m))
    return d


def step_interp1d(eval_t, t, y):
    idx = torch.searchsorted(t, eval_t, side="right")
    idx = torch.clip(idx - 1, min=0)
    return y[idx]


step_interp = torch.vmap(step_interp1d, in_dims=(None, 0, 0))


sim_config = {
    "min_dt": 0.1,
    "max_dt": 0.5,
    "cost": [1.0, 1.0, 1.0],
    "lambda": 1.0,
    "null_visit": 10.0,
}


class Sensing(torch.nn.Module):
    def __init__(
        self,
        x_dim,
        y_dim,
        *,
        hidden_size=10,
        num_layer=1,
        predictor=None,
        sim_config=sim_config,
        adjoint=False,
        gamma=0.99,
        rnn_encoder=False,
        resolution=0.4,
        nll_reward=False,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.adjoint = adjoint
        self.gamma = gamma
        self.resolution = resolution
        self.nll_reward = nll_reward

        self.name = "ActiveSensing"

        for k in sim_config:
            setattr(self, k, sim_config[k])

        if rnn_encoder:
            self.encoder = RNN(self.x_dim, self.hidden_size, self.hidden_size, self.num_layer)
        else:
            self.encoder = NCDE(
                self.x_dim,
                self.hidden_size,
                hidden_size=self.hidden_size,
                num_layer=self.num_layer,
                resolution=self.resolution,
            )

        if self.max_dt == self.min_dt:
            adaptive_interval = False
        else:
            adaptive_interval = True
        self.pi = Policy(
            self.hidden_size, self.x_dim, self.hidden_size, self.num_layer, adaptive_interval=adaptive_interval
        )
        self.critic = MLP(self.hidden_size, 1, self.hidden_size, self.num_layer)
        if predictor is not None:
            self.predictor = predictor
        else:
            self.predictor = Predictor(
                self.x_dim,
                self.y_dim,
                hidden_size=self.hidden_size,
                num_layer=self.num_layer,
                fit_obs=True,
                resolution=self.resolution,
            )

    def simulate(self, batch):
        t = batch["t"]
        x = batch["x"]
        mask = batch["mask"]
        valid_t = t[mask == 1]
        t_max = valid_t.max()
        t_min = valid_t.min()

        _, interpolator = obtain_interpolator(t, x, resolution=self.resolution)

        length = torch.ceil((t_max - t_min) / self.min_dt).to(int) + 2
        batch_size, _, _ = x.shape

        # time-first
        obs_t = torch.full((length, batch_size), -1.0, device=x.device)
        obs_x = torch.full((length, batch_size, self.x_dim), torch.nan, device=x.device)
        obs_m = torch.full((length, batch_size, self.x_dim), 0.0, device=x.device)
        pred_y = torch.full((length, batch_size, self.y_dim), 0.0, device=x.device)
        value = torch.full((length, batch_size), 0.0, device=x.device)

        log_p_m = torch.full((length, batch_size), 0.0, device=x.device)
        log_p_delta = torch.full((length, batch_size), 0.0, device=x.device)

        # no previous observation at time -1
        h_t = obs_t[:0].transpose(0, 1).contiguous()
        h_x = obs_x[:0].transpose(0, 1).contiguous()
        z = self.encoder(h_t, h_x)

        # decide the subset of feature to measure
        m, _, log_p_m_t, log_p_delta_t = self.pi(z)
        log_p_m[1] = log_p_m_t.sum(dim=-1)
        log_p_delta[0] = log_p_delta_t
        value[0] = self.critic(z)[:, 0]

        # start measurement at time 0
        t = interpolator.interval[0].expand(batch_size).contiguous()
        # logger.debug(f"z(0)={z[0,0].item():.3f}")
        for i in range(1, length):
            if (t > t_max).all():
                last_val, _ = torch.max(obs_t, dim=0)
                obs_t[i:] = last_val[None, :] + self.min_dt * torch.arange(1, length - i + 1, device=x.device).view(
                    length - i, 1
                )  # .expand(length - i, batch_size)
                break

            # collect new data point
            full_x = interpolator.evaluate(t)
            full_x = torch.diagonal(full_x).transpose(0, 1)
            # full_x = torch.stack([interpolator.evaluate(t_i)[i] for i, t_i in enumerate(torch.unbind(t))], dim=0)

            # only keep selected features via m (no need to update since there is no new observation yet)
            obs_x[i, m == 1] = full_x[m == 1].detach()
            obs_t[i] = t.detach()
            obs_m[i] = m.detach()

            # obtain current observation history
            h_t = obs_t[: i + 1].transpose(0, 1).contiguous()
            h_x = obs_x[: i + 1].transpose(0, 1).contiguous()

            # evolve latent state z
            z = self.encoder(h_t, h_x)[:, -1]
            # logger.debug(f"z({t[0].item():.3f})={z[0,0].item():.3f}")
            # determine the observation interval and the selected features for next time

            m, delta, log_p_m_t, log_p_delta_t = self.pi(z)
            if i + 1 < length:
                log_p_m[i + 1] = log_p_m_t.sum(dim=-1)
            log_p_delta[i] = log_p_delta_t
            value[i] = self.critic(z)[:, 0]

            delta_t = self.min_dt + (self.max_dt - self.min_dt) * delta

            # update observation time and previous observation
            t = t + delta_t

        # batch-first
        obs_t = obs_t.transpose(0, 1).contiguous()
        obs_x = obs_x.transpose(0, 1).contiguous()
        obs_m = obs_m.transpose(0, 1).contiguous()

        # predict outcome at current time
        pred_y = self.predictor.predict(obs_t, obs_x)

        # pred_y = pred_y.transpose(0, 1).contiguous()
        value = value.transpose(0, 1).contiguous()
        log_p_m = log_p_m.transpose(0, 1).contiguous()
        log_p_delta = log_p_delta.transpose(0, 1).contiguous()
        # log_p = torch.nan_to_num(log_p_m + log_p_delta,nan=0.0, posinf=1.0, neginf=-1.0)
        log_p = log_p_m + log_p_delta

        t = batch["t"]
        data_mask = batch["mask"]

        # t_max = t[torch.arange(len(t)), data_mask.sum(dim=-1).to(int)-1][:,None]
        _, idx = last_nonzero(data_mask, dim=-1)
        t_e = t[torch.arange(len(t)), -idx - 1][:, None]
        t_s = t[:, [0]]
        mask = (obs_t >= t_s) & (obs_t <= t_e)
        # mask = (obs_t>=X.interval[0])&(obs_t<=X.interval[-1])

        if self.nll_reward:
            eval_y = batch_interp_nd(obs_t, batch["t"], batch["y"])
            eval_y[eval_y != 1.0] = 0.0
            ll = torch.sum(torch.log(pred_y + EPS) * eval_y, dim=-1)
            r_y = -ll
        else:
            # evaluate oracle prediction at finer grain
            eval_t = torch.linspace(t_min, t_max, length, device=x.device)
            # predicted y (oracle)
            eval_y = self.predictor.predict(batch["t"], batch["x"], t_eval=eval_t)

            # cost in prediction error
            r_y = self.calculate_r_y(obs_t, pred_y, eval_t, eval_y)
        # cost in measurement
        r_m = torch.matmul(obs_m, torch.tensor(self.cost, device=obs_m.device))
        r_null = 1.0 * (obs_m == 0).all(dim=-1)
        cost = r_m + getattr(self, "lambda", 1.0) * r_y + self.null_visit * r_null

        # no cost beyond [0,T]
        cost[~mask] = 0.0

        ret = {
            "obs_t": obs_t,
            "obs_x": obs_x,
            "obs_m": obs_m,
            "pred_y": pred_y,
            "pred_y_baseline": eval_y,
            "log_p": log_p,
            "log_p_m": log_p_m,
            "log_p_delta": log_p_delta,
            "cost": cost,
            "value": value,
            "mask": mask,
            "r_y": r_y,
            "r_m": r_m,
            "r_null": r_null,
        }

        return ret

    # def step_interp(self, t, y, eval_t):
    #     ys = []
    #     for i in range(len(y)):
    #         idx = torch.searchsorted(t[i], eval_t, side="right") - 1
    #         idx = torch.clip(idx, min=0)
    #         yi = y[i, idx]
    #         ys.append(yi)
    #     pred_y = torch.stack(ys)
    #     return pred_y

    @torch.compile
    def calculate_r_y(self, obs_t, pred_y, eval_t, eval_y):
        r_y = torch.zeros(obs_t.shape, device=obs_t.device)
        for j in range(len(r_y)):
            idx = torch.searchsorted(obs_t[j], eval_t, side="right")
            for i in range(obs_t.shape[1]):
                rs = d_js(eval_y[j, idx == i + 1], pred_y[j, [i]])
                r_y[j, i] = torch.sum(rs)
        r_y = r_y * (eval_t[-1] - eval_t[0]) / len(eval_t)
        return r_y

    def forward(self, batch, fit_predictor=False):
        ret = self.simulate(batch)
        _, length = ret["cost"].shape
        reward = torch.zeros_like(ret["cost"])
        for i in range(1, length + 1):
            last_i = min(length - i + 1, length - 1)
            reward[:, length - i] = ret["cost"][:, length - i] + self.gamma * reward[:, last_i]
        reward = reward.detach()
        # R = torch.cumsum(ret['cost'].flip(dim=-1),dim=-1).flip(dim=-1).detach()
        reward = (reward - reward[ret["mask"]].mean()) / (reward[ret["mask"]].std() + EPS)
        advantage = (reward - ret["value"]).detach()

        actor_loss = ret["log_p"] * advantage * ret["mask"]
        actor_loss = torch.nan_to_num(actor_loss, nan=0.0, posinf=0.0, neginf=0.0)
        # print(actor_loss.max(),(torch.square(ret['value'] - R) * ret['mask']).max())
        losses = {}
        losses["actor"] = actor_loss.mean()
        losses["critic"] = (torch.square(ret["value"] - reward) * ret["mask"]).mean()
        # batch['obs_t'] = ret['obs_t']
        # batch['obs_mask'] = ret['obs_m']
        if fit_predictor:
            for k, v in self.predictor(batch).items():
                losses[k] = v

        return losses


def last_nonzero(x, dim=0):
    x = x.flip(1)
    nonz = x > 0
    return ((nonz.cumsum(dim) == 1) & nonz).max(dim)
