#!/usr/bin/env python3
"""Prepare chronological splits from a raw [time, channel] Numpy array."""
import argparse
from pathlib import Path
import pickle
import numpy as np

DATASETS = {"ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7,
            "ExchangeRate": 8, "Weather": 21, "METR-LA": 207, "Electricity": 321}


def prepare(dataset, input_path, output):
    x = np.load(input_path, allow_pickle=False)
    if x.ndim != 2 or x.shape[1] != DATASETS[dataset] or np.isinf(x).any():
        raise ValueError("Expected finite-or-NaN [time, channel] data with correct channels")
    r_train, r_val = (0.6, 0.2) if dataset.startswith("ETT") else (0.7, 0.1)
    n_train, n_val = int(len(x) * r_train), int(len(x) * r_val)
    splits = [x[:n_train], x[n_train:n_train + n_val], x[n_train + n_val:]]
    if min(len(s) for s in splits) < 48:
        raise ValueError("Splits are too short for the training window protocol")
    mean, std = np.nanmean(splits[0], axis=0), np.nanstd(splits[0], axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Each channel needs finite training observations")
    std[std < 1e-8] = 1.0
    output = Path(output)
    names = ["train_set.pkl", "val_set.pkl", "test_set.pkl", "scaler.pkl"]
    if any((output / name).exists() for name in names):
        raise FileExistsError("Prepared data files already exist")
    output.mkdir(parents=True, exist_ok=True)
    for name, split in zip(names, splits):
        with (output / name).open("wb") as f:
            pickle.dump(((split - mean) / std).astype(np.float32), f)
    with (output / "scaler.pkl").open("wb") as f:
        pickle.dump([mean, std], f)
    print(dataset, [s.shape for s in splits], output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=DATASETS, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    prepare(a.dataset, a.input, a.output_dir)
