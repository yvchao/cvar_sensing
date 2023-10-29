import torch
import torchcde

from .constants import nan_rep
from .nn import GRU, MLP
from .utils import prepare_time_series


def prepare_input(t, x, resolution=1.0):
    t, x = prepare_time_series(t, x, resolution=resolution, fill_value=nan_rep)
    # TODO: prepend and postpend values of x[0] and x[1] to avoid extrapolation
    # Done. No need to prepend as this is handled by interpolation step and we never evaluate at t<0.
    x = torch.cat([x, x[:, [-1]]], dim=-2)
    t = torch.cat([t, t[:, [-1]] + resolution], dim=-1)
    return t, x


def obtain_interpolator(t, x, resolution):
    t, x = prepare_input(t, x, resolution)
    coeffs = torchcde.linear_interpolation_coeffs(x, t=t[0])
    interpolator = torchcde.LinearInterpolation(coeffs, t=t[0])
    return t[0], interpolator


# @torch.compile
def take_from_sequence(t, y, t_eval):
    idx = torch.searchsorted(t, t_eval)[:, :, None]
    _, _, y_dim = y.shape
    y = torch.gather(y, 1, idx.expand(*idx.shape[:-1], y_dim))
    return y


class LatentDynamics(torch.nn.Module):
    def __init__(self, x_size, z_size, hidden_size=1, num_layer=3):
        super().__init__()
        self.z_size = z_size
        self.x_size = x_size
        # derivative is bounded to [-1,1]
        self.dynamic = MLP(z_size, z_size * x_size, hidden_size, num_layer, output_func=torch.nn.Tanh())

    def forward(self, t, z):
        return self.dynamic(z).view(*z.shape[:-1], self.z_size, self.x_size) - 1e-3 * z.unsqueeze(-1).expand(
            (*z.shape, self.x_size)
        )


class NCDE(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        hidden_size: int,
        num_layer: int,
        adjoint: bool = False,
        method: str = "rk4",
        resolution: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.adjoint = adjoint
        self.method = method
        self.resolution = resolution

        # input_size + time dimension
        self.func = LatentDynamics(self.input_size, self.output_size, self.hidden_size, self.num_layer)
        self.initialize = MLP(self.input_size, self.output_size, self.hidden_size, self.num_layer)

    def forward(self, t, x, *, t_eval=None, z0=None):
        batch_size, length, x_dim = x.shape
        if length == 0:
            z = self.initialize(torch.full((batch_size, x_dim), nan_rep, device=x.device))
        else:
            t_merge, interpolator = obtain_interpolator(t, x, self.resolution)
            if z0 is None:
                z0 = self.initialize(interpolator.evaluate(interpolator.interval[0]))

            if t_eval is None:
                z_unified = torchcde.cdeint(
                    interpolator,
                    func=self.func,
                    z0=z0,
                    t=t_merge,
                    adjoint=self.adjoint,
                    method=self.method,
                )
                t_min, _ = torch.min(t, dim=-1, keepdim=True)
                t_eval = t - t_min
                z = take_from_sequence(t_merge, z_unified, t_eval)
            else:
                t_min, _ = torch.min(t_eval, dim=-1, keepdim=True)
                t_eval = t_eval - t_min
                z = torchcde.cdeint(
                    interpolator,
                    func=self.func,
                    z0=z0,
                    t=t_eval,
                    adjoint=self.adjoint,
                    method=self.method,
                )

        return z


class RNN(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layer: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer

        # input_size + time dimension
        self.rnn = GRU(self.input_size + 1, self.hidden_size, 1)
        self.mlp = MLP(self.hidden_size, self.output_size, self.hidden_size, self.num_layer)

    def forward(self, t, x):
        # fill nan values in x
        nan_mask = torch.isnan(x)
        x_masked = x.clone()
        x_masked[nan_mask] = nan_rep

        # combine time and feature
        dt = torch.zeros_like(t)
        dt[:, 1:] = t[:, 1:] - t[:, :-1]
        xt = torch.cat([dt[:, :, None], x_masked], dim=-1)

        # obtain embedding
        state = self.rnn(xt)
        embed = self.mlp(state)
        return embed
