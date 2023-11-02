import numpy as np
import scipy.stats
import torch
from sklearn import metrics

from ..train import batch_call
from .interpolation import step_interp


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, h, m - h, m + h


def get_auc_scores(dataset, ret):
    true_y = dataset[:]["y"]
    t = dataset[:]["t"]
    mask = dataset[:]["mask"]
    pred_y = ret["pred_y"].numpy()
    pred_t = ret["obs_t"].numpy()

    interp_y = np.zeros_like(true_y)
    for i in range(len(pred_t)):
        idx = np.searchsorted(pred_t[i], t[i], side="right") - 1
        interp_y[i] = pred_y[i, idx]

    pred_y = interp_y[mask == 1]
    true_y = true_y[mask == 1]
    _, y_dim = pred_y.shape
    roc = np.zeros((y_dim,))
    prc = np.zeros((y_dim,))
    for i in range(y_dim):
        roc[i] = metrics.roc_auc_score(true_y, pred_y)
        prc[i] = metrics.average_precision_score(true_y, pred_y)
    return roc.mean(), prc.mean()


def calculate_delay(batch, ret, threshold=0.3, label_index=0):
    data_t = batch["t"]
    data_mask = batch["mask"]
    t_valid = data_t[data_mask == 1]
    t_min = t_valid.min()
    t_max = t_valid.max()
    obs_t = ret["obs_t"]
    pred_y_baseline = ret["pred_y_baseline"]
    pred_y = ret["pred_y"]

    _, num_sample, _ = pred_y_baseline.shape
    baseline_t = torch.linspace(0, t_max - t_min, steps=num_sample)
    eval_t = torch.linspace(0, t_max - t_min, steps=200)
    y_baseline = step_interp(
        eval_t, baseline_t.unsqueeze(0).expand((len(pred_y_baseline), -1)), pred_y_baseline
    ).detach()
    y_policy = step_interp(eval_t, obs_t, pred_y).detach()

    indicies = batch_find_where(y_policy[:, :, label_index] >= threshold)
    policy_detect_t = eval_t[indicies]
    indicies = batch_find_where(y_baseline[:, :, label_index] >= threshold)
    baseline_detect_t = eval_t[indicies]
    baseline_detect_t[indicies == -1] = np.nan
    delay_t = policy_detect_t - baseline_detect_t
    return delay_t


def find_where(a):
    (indicies,) = torch.where(a)
    if len(indicies) == 0:
        return -1
    else:
        return indicies[0].item()


def batch_find_where(arr):
    indicies = [find_where(a) for a in arr]
    return np.array(indicies)


def evaluate(dataset, model, seed=0, delay_eval=calculate_delay, eval_mode=False, device=None):
    torch.random.manual_seed(seed)
    if eval_mode:
        model.eval()
    else:
        model.train()

    with torch.no_grad():
        ret = batch_call(model.simulate, dataset, device=device)

    metrics = {}
    roc, prc = get_auc_scores(dataset, ret)
    metrics["roc"] = roc
    metrics["prc"] = prc
    metrics["cost"] = torch.mean(torch.sum(ret["r_m"] * ret["mask"], dim=-1)).item()

    for threshold in [0.1, 0.3, 0.5, 0.7]:
        delays = delay_eval(dataset[:], ret, threshold=threshold)
        metrics[f"delay(p>={threshold})"] = np.nanmean(np.abs(delays)).item()

    return metrics
