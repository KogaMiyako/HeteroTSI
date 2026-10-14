import torch
import numpy as np

def sample_mask(
    shape, p=0.0015, p_noise=0.05, max_seq=1, min_seq=1, rng=None
):  # block missing, point missing
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers
    mask = rand(shape) < p
    for col in range(mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, shape[0] - 1)
        mask[idxs, col] = True
    mask = mask | (rand(mask.shape) < p_noise)
    return mask.astype("uint8")

def missed_eval_np(predict, true, mask):
    """
    predict: [samples, seq_len, feature_dim]
    true: [samples, seq_len, feature_dim]
    mask: [samples, seq_len, feature_dim]
    """
    predict, true = np.asarray(predict), np.asarray(true)
    mae = np.sum(np.absolute(predict - true) * (1 - mask)) / (np.sum(1 - mask) + 1e-5)
    mse = np.sum((predict - true) ** 2 * (1 - mask)) / (np.sum(1 - mask) + 1e-5)
    rmse = np.sqrt(
        np.sum((predict - true) ** 2 * (1 - mask)) / (np.sum(1 - mask) + 1e-5)
    )
    mape = np.sum(np.absolute((predict - true) * (1 - mask))) / (
        np.sum(np.absolute(true * (1 - mask))) + 1e-5
    )
    R2 = 1 - mse / (
        np.sum((true - np.mean(true)) ** 2 * (1 - mask)) / (np.sum(1 - mask) + 1e-5)
    )
    return mae, rmse, mape, mse, R2

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )

def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))

def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler  # (B,L,K)
    forecast = forecast * scaler + mean_scaler  # (B,nsample,L,K)

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)
