import numpy as np
import torch

from .constants import nan_rep, newdim


def fillna1d(t, x):
    """
    t: 1-D array
    x: 1-D array
    """
    nans = np.isnan(x)
    if np.all(nans):
        return np.full_like(x, nan_rep)

    x = x.copy()
    x[nans] = np.interp(t[nans], t[~nans], x[~nans])
    return x


def interp1d(eval_t, t, x):
    """
    eval_t: 1-D array
    t: 1-D array
    x: 1-D array
    """
    eval_x = np.interp(eval_t, t, x)
    return eval_x


def interp(eval_t, t, x_ndim, axis=0):
    """
    eval_t: 1-D array
    t: 1-D array
    x_ndim: n-D array
    axis: int, time dimension
    """
    x_ndim = np.apply_along_axis(lambda x: interp1d(eval_t, t, x), axis, x_ndim)
    return x_ndim


def fillna(t, x_ndim, axis=0):
    """
    t: 1-D array
    x_ndim: n-D array
    axis: int, time dimension
    """
    x_ndim = np.apply_along_axis(lambda x: fillna1d(t, x), axis, x_ndim)
    return x_ndim


def take_from_array(eval_t, t, y):
    """
    eval_t: 2-D array of batch x time
    t: 1-D array of time points
    y: 3-D array of batch x time x feature
    """
    idx = np.searchsorted(t, eval_t, side="right")[:, :, None]
    # y = np.concatenate([y[:, [0], :], y], axis=1)
    idx = np.clip(idx - 1, a_min=0)
    eval_y = np.take_along_axis(y, idx, 1)
    return eval_y


def take_from_tensor(eval_t, t, y):
    """
    eval_t: 2-D tensor of batch x time
    t: 1-D tensor of time points
    y: 3-D tensor of batch x time x feature
    """
    idx = torch.searchsorted(t, eval_t, side="right")[:, :, newdim]
    _, _, y_dim = y.shape
    # y = torch.cat([y[:, [0], :], y], dim=1)
    idx = torch.clamp(idx - 1, min=0)
    eval_y = torch.gather(y, 1, idx.expand(*idx.shape[:-1], y_dim))
    return eval_y


# from here
# https://github.com/pytorch/pytorch/issues/50334#issuecomment-1247611276
def interp1d_tensor(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[:, 1:] - fp[:, :-1]) / (xp[:, 1:] - xp[:, :-1])  # slope
    b = fp[:, :-1] - (m.mul(xp[:, :-1]))

    indicies = torch.sum(torch.ge(x[:, :, None], xp[:, None, :]), -1) - 1  # torch.ge:  x[i] >= xp[i] ? true: false
    indicies = torch.clamp(indicies, 0, m.shape[-1] - 1)

    line_idx = torch.arange(len(indicies), device=indicies.device).view(-1, 1)
    line_idx = line_idx.expand(indicies.shape)
    return m[line_idx, indicies].mul(x) + b[line_idx, indicies]


interp_tensor = torch.vmap(interp1d_tensor, in_dims=(None, None, -1), out_dims=-1)
