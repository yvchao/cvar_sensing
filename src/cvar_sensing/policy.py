import torch

from .constants import EPS
from .nn import MLP


class Policy(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        x_dim: int,
        hidden_size: int,
        num_layers: int,
        full_obs: bool = False,
        adaptive_interval=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.x_dim = x_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.full_obs = full_obs
        self.adaptive_interval = adaptive_interval

        self.pi_m = MLP(self.input_size, self.x_dim, self.hidden_size, self.num_layers)
        self.pi_delta = MLP(self.input_size, 2, self.hidden_size, self.num_layers)

    def forward(self, embed):
        # feature selection mask
        logits = self.pi_m(embed)
        p = torch.sigmoid(logits)
        if self.training:
            # p = p + (torch.clamp(p, 0.1, 0.9) - p).detach()
            p = torch.clamp(p, 0.1, 0.9)  # + p - p.detach()

        bern = torch.distributions.Bernoulli(p)

        # beta_params = torch.square(self.pi_delta(embed)) + EPS
        # alpha, beta = 1.0 + beta_params[:, 0], 1.0 + 1.0 * beta_params[:, 1]

        beta_params = self.pi_delta(embed)
        c = beta_params[:, 0] ** 2 + 10
        w = torch.sigmoid(beta_params[:, 1])

        if self.training:
            c = torch.clamp(c, 0, 100)  # + c - c.detach()
            w = torch.clamp(w, 0.1, 0.9)  #  + w - w.detach()

        alpha, beta = 1.0 + c * w, 1.0 + 1.0 * c * (1 - w)

        beta_dist = torch.distributions.Beta(alpha, beta)

        if self.training:
            m = bern.sample()
            if self.adaptive_interval:
                delta = beta_dist.sample()
            else:
                delta = ((alpha - 1.0) / (alpha + beta - 2.0)).detach()  # mode
        else:
            m = 1.0 * (p > 0.5)  # binary
            delta = (alpha - 1.0) / (alpha + beta - 2.0)  # mode

        if self.full_obs:
            m = torch.ones_like(m)

        log_p_m = bern.log_prob(m)
        if self.adaptive_interval:
            delta = torch.clamp(delta, EPS, 1 - EPS)
            log_p_delta = beta_dist.log_prob(delta)
        else:
            log_p_delta = torch.zeros((len(embed),), dtype=embed.dtype, device=embed.device)
        return m, delta, log_p_m, log_p_delta
