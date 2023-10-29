import torch

from .interpolation import fillna_nd


def merge_series(t, x, resolution):
    # first, we merge all time stamps into a global axis
    t_min, _ = torch.min(t, dim=-1, keepdim=True)
    t = t - t_min
    t_max = t.max()
    steps = int(t_max / resolution)
    t_merge = torch.linspace(0, t_max, max(steps, 10), device=t.device)

    # then, we merge values in x by putting them to the closest bin
    idx = torch.bucketize(t, t_merge)  # bin label

    marker = torch.zeros(idx.shape[0], idx.max() + 1, idx.shape[1], device=t.device)  # weight for calculating average
    marker[torch.arange(len(idx))[:, None], idx, torch.arange(idx.size(1))] = 1
    nan_mask = torch.isnan(x)
    num = marker @ torch.nan_to_num(x)
    den = marker @ (1.0 - 1.0 * nan_mask)
    x_merge = num / den  # we calculate nanmean manually here

    # only return the relevant time stamps
    sel_idx = torch.unique(idx)
    t_merge = t_merge[sel_idx].unsqueeze(0).expand((len(x), -1))
    x_merge = x_merge[:, sel_idx]
    return t_merge, x_merge


def prepare_time_series(t, x, resolution=1.0, fill_value=0.0):
    # t: batch_size x time_size
    # x: batch_size x time_size x feature_size

    # we first merge time series in [t, x] into unified axis
    t, x = merge_series(t, x, resolution)

    # after that, we interpolate any missing values in x
    if torch.isnan(x).any():
        x_filled = fillna_nd(t, x, fill_value=fill_value, dim=-1)
    else:
        x_filled = x

    return t, x_filled
