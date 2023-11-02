from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn import metrics
from tqdm import auto

from .meters import AggregateMeters


def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape) + [nb_classes])


def calculate_loss(losses, weights, device=None):
    total_loss = torch.tensor(0.0, device=device)
    for k, v in weights.items():
        loss = losses.get(k, torch.tensor(0.0)).to(device)
        total_loss += v * loss
    return total_loss


def dict_to_device(dictionary, device=None):
    for k, v in dictionary.items():
        dictionary[k] = v.to(device)
    return dictionary


def ndarray_to_tensor(batch):
    for k, v in batch.items():
        batch[k] = torch.from_numpy(v.astype("float32"))
    return batch


def batch_to_device(batch, device):
    for k, v in batch.items():
        batch[k] = v.to(device)
    return batch


def clone_state_dict(model):
    state_dict = {}
    for key in model.state_dict():
        state_dict[key] = model.state_dict()[key].clone()
    return state_dict


def save_model(model, path=Path("saved_models"), name=None):
    path.mkdir(exist_ok=True)
    state_dict = model.state_dict()
    filename = f"{model.name}.pt" if name is None else f"{name}.pt"
    torch.save(state_dict, path / filename)


def load_model(model, filename, device=None):
    state_dict = torch.load(filename, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device)


def batch_call(func, dataset, device=None, batch_size=100):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rets = {}
    for batch in loader:
        batch = dict_to_device(batch, device=device)
        ret = func(batch)
        for k, v in ret.items():
            vs = rets.get(k, [])
            vs.append(v)
            rets[k] = vs

    for k, v in rets.items():
        max_length = np.max([vi.shape[1] for vi in v])
        if v[0].ndimension() == 2:
            merged = torch.zeros((len(dataset), max_length))
        else:
            merged = torch.zeros((len(dataset), max_length, v[0].shape[-1]))

        start = 0
        for vi in v:
            size = vi.shape[0]
            length = vi.shape[1]
            merged[start : start + size, :length] = vi
            start = start + size

        rets[k] = merged

    return rets


def predictor_evaluation(model, dataset, batch_size=100, device=None, eval_mode=True):
    if eval_mode:
        model.eval()
    else:
        model.train()

    def func(batch):
        t = batch["t"]
        x = batch["x"]
        pred_y = model.predict(t, x)

        ret = {}
        ret["true_y"] = batch["y"]
        ret["pred_y"] = pred_y
        ret["mask"] = batch["mask"]
        return ret

    with torch.no_grad():
        ret = batch_call(func, dataset, device=device, batch_size=batch_size)

    roc, auc = get_auc_scores(ret)
    return roc, auc


def get_auc_scores(ret):
    pred_y = ret["pred_y"][ret["mask"] == 1]
    true_y = ret["true_y"][ret["mask"] == 1]
    _, y_dim = pred_y.shape
    roc = np.zeros((y_dim,))
    prc = np.zeros((y_dim,))
    for i in range(y_dim):
        roc[i] = metrics.roc_auc_score(true_y, pred_y)
        prc[i] = metrics.average_precision_score(true_y, pred_y)
    return roc.mean(), prc.mean()


def train(model, loss_weights, loader, optimizer, device=None):
    total_losses = {}
    num_sample = 0.0
    for batch in loader:
        optimizer.zero_grad()
        batch = dict_to_device(batch, device)
        losses = model(batch)
        loss = calculate_loss(losses, loss_weights, device)
        loss.backward()
        optimizer.step()

        batch_size = len(batch["x"])
        for k in losses:
            total_losses[k] = total_losses.get(k, 0.0) + batch_size * losses[k].cpu().item()
        num_sample += batch_size

    for k in total_losses:
        total_losses[k] = total_losses[k] / num_sample
    return total_losses


def test(model, loader, device=None):
    total_losses = {}
    if loader is None:
        return total_losses

    num_sample = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = dict_to_device(batch, device)
            losses = model(batch)

            batch_size = len(batch["x"])
            for k in losses:
                total_losses[k] = total_losses.get(k, 0.0) + batch_size * losses[k].cpu().item()
            num_sample += batch_size

    for k in total_losses:
        total_losses[k] = total_losses[k] / num_sample
    return total_losses


def fit(
    model,
    dataset,
    loss_weights,
    *,
    test_set=None,
    test_metric=None,
    epochs=100,
    params={},
    lr=0.01,
    batch_size=50,
    device=None,
    eval_func=None,
    sampler=None,
    tolerance=None,
):
    if params == {}:
        params = model.parameters()

    optimizer = optim.AdamW(params, lr=lr)

    loss_meters = AggregateMeters(loss_weights.keys(), momentum=0.6)
    test_loss_meters = AggregateMeters(loss_weights.keys(), momentum=0.6)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if test_set is not None:
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    else:
        test_loader = None

    best_validation_loss = np.inf
    no_improvement_count = 0
    best_model = {}

    with auto.trange(epochs, position=0) as tbar:
        for itr in tbar:
            if eval_func is not None:
                eval_func(itr, model, test_set, device=device)

            if sampler is not None:
                subset, updated = sampler(itr, model, dataset, device=device)
                if updated:
                    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)

            losses = train(model, loss_weights, loader, optimizer, device)
            loss_meters.update(losses)

            test_losses = test(model, test_loader, device)
            if test_metric is not None and test_losses != {}:
                test_loss_meters.update(test_losses)
                loss_valid = test_losses.get(test_metric, np.inf)

                if best_validation_loss > loss_valid:
                    best_validation_loss = loss_valid
                    no_improvement_count = 0
                    best_model = clone_state_dict(model)
                else:
                    no_improvement_count += 1

                if tolerance is not None and no_improvement_count >= tolerance:
                    break
            else:
                pass

            msg = ",".join([f"{k}:{v:.3f}" for k, v in loss_meters.report().items()])
            test_msg = ",".join([f"{k}:{v:.3f}" for k, v in test_loss_meters.report().items()])
            summary = f"tr:{msg}|te:{msg}" if test_msg != "" else f"tr:{msg}"
            tbar.set_description(summary)

    if best_model != {}:
        model.load_state_dict(best_model)


def sampler(itr, model, dataset, *, interval=100, **kwargs):
    if itr % interval == 0:
        subset = adversarial_sampling(dataset, model, **kwargs)
        return subset, True
    else:
        return dataset, False


def adversarial_sampling(dataset, model, alpha=0.3, batch_size=100, device=None):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # model = model.eval()
    q_vals = []
    with torch.no_grad():
        for batch in loader:
            batch = dict_to_device(batch, device)
            ret = model.simulate(batch)
            q = torch.sum(ret["cost"] * ret["mask"], dim=-1)  # we need to exclude unrelated steps
            q_vals.append(q)
    q = torch.cat(q_vals, dim=0)
    q_alpha = torch.quantile(q, q=1 - alpha)
    (idx,) = torch.where(q > q_alpha)
    subset = torch.utils.data.Subset(dataset, idx.cpu())
    return subset


def reweight_sampling(dataset, model, alpha=0.3, batch_size=100, device=None):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.eval()
    q_vals = []
    with torch.no_grad():
        for batch in loader:
            batch = dict_to_device(batch, device)
            ret = model.simulate(batch)
            q = ret["cost"].sum(dim=-1)
            q_vals.append(q)
    q = torch.cat(q_vals, dim=0)
    q_alpha = torch.quantile(q, q=1 - alpha)

    labels = 1.0 * (q > q_alpha)
    (idx,) = np.where(labels == 1)
    idx_tail = np.random.choice(idx, size=(len(q) // 2,))

    (idx,) = np.where(labels == 0)
    idx_major = np.random.choice(idx, size=(len(q) // 2,))

    idx = np.concatenate([idx_tail, idx_major])

    subset = torch.utils.data.Subset(dataset, idx)
    return subset
