import torch
from torch import Tensor


# from here
# https://github.com/pytorch/pytorch/issues/50334#issuecomment-1000917964
def interp_1d(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
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
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1

    left = indicies < 0
    right = indicies >= len(m)
    inside = (~left) & (~right)

    indicies = torch.clamp(indicies, 0, len(m) - 1)

    f = m[indicies] * x + b[indicies]

    f = f * inside + left * fp[0] + right * fp[-1]

    return f


def batch_interp_1d(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
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
    m = (fp[:, 1:] - fp[:, :-1]) / (xp[:, 1:] - xp[:, :-1] + torch.finfo(xp.dtype).tiny)  # slope
    b = fp[:, :-1] - (m.mul(xp[:, :-1]))

    indicies = torch.sum(torch.ge(x[:, :, None], xp[:, None, :]), -1) - 1  # torch.ge:  x[i] >= xp[i] ? true: false

    left = indicies < 0
    right = indicies >= m.shape[-1]
    inside = (~left) & (~right)

    indicies = torch.clamp(indicies, 0, m.shape[-1] - 1)

    line_idx = torch.arange(len(indicies), device=indicies.device).view(-1, 1)
    line_idx = line_idx.expand(indicies.shape)

    f = m[line_idx, indicies].mul(x) + b[line_idx, indicies]
    f = f * inside + left * fp[:, [0]] + right * fp[:, [-1]]
    return f


batch_interp_nd = torch.vmap(batch_interp_1d, in_dims=(None, None, -1), out_dims=-1)


def fillna_1d(xp: Tensor, fp: Tensor, fill_value: float = 0.0) -> Tensor:
    """Fill missing values in a batch of 1-d time-series

    Parameters
    ----------
    xp : 2-d tensor
        Time stamps of observations in fp
    fp : 2-d tensor
        Observed values with missingness
    fill_value : float
        Value to replace the missingness in fp

    Returns
    -------
    2-d tensor
        fp with missing values filled
    """

    # first put all valid values into the front of each row
    mask = ~torch.isnan(fp)
    justified_mask, _ = torch.sort(1.0 * mask, dim=-1, descending=True)

    fp_ = torch.full(fp.shape, fill_value, device=xp.device)
    xp_ = torch.full(xp.shape, xp.max() + 1, device=xp.device)  # we also need to rearange time stampes
    fp_[justified_mask != 0] = fp[mask]
    xp_[justified_mask != 0] = xp[mask]

    # then forwar fill the end of each row with the last valid value

    # numner of valid values at each row along dimension of axis
    last_valid_idx = torch.sum(justified_mask, dim=-1, dtype=int) - 1
    last_valid_idx = torch.clamp(last_valid_idx, min=0, max=fp.shape[-1] - 1)
    # take the last valid value at each row along dimension of axis
    impute_val = fp_.gather(dim=-1, index=last_valid_idx.unsqueeze(-1)).expand(fp_.shape)
    fp_ = torch.where(justified_mask == 1, fp_, impute_val)

    filled = batch_interp_1d(xp, xp_, fp_)
    return filled


def fillna_nd(xp, fp, fill_value=0.0, dim=-1):
    return torch.stack([fillna_1d(xp, fp_i, fill_value=fill_value) for fp_i in torch.unbind(fp, dim=dim)], dim=dim)
