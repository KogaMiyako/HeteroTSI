import argparse
import copy
import itertools
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .data import generate_train_dataloader, generate_val_test_dataloader
from .engine import train, evaluate, seed_torch
from .model import MTSCI_Freq

DATASETS = {"ETTh1": 7, "ETTh2": 7, "ETTm1": 7, "ETTm2": 7,
            "ExchangeRate": 8, "Weather": 21, "METR-LA": 207, "Electricity": 321}
CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


class LimitedLoader:
    """Limit the number of batches yielded by a data loader."""
    def __init__(self, loader, limit):
        self.loader, self.limit = loader, limit

    def __len__(self):
        return min(len(self.loader), self.limit)

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.limit)


def parser():
    p = argparse.ArgumentParser(description="Train or evaluate HeteroDiff")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--pattern", required=True, choices=["point", "block"])
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--nsample", type=int, default=100)
    p.add_argument("--checkpoint", type=Path, help="Evaluate only; skip training")
    p.add_argument("--threads", type=int, help="CPU intra-op threads")
    p.add_argument("--smoke", action="store_true",
                   help="5 epochs, one batch per split, 2 samples")
    return p


def run(args):
    if args.nsample < 1:
        raise ValueError("nsample must be positive")
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(args.threads)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu")
    config_path = args.config or CONFIG_ROOT / f"{args.dataset}_{args.pattern}.yaml"
    config = yaml.safe_load(config_path.read_text())
    if args.smoke:
        config["train"]["epochs"] = 5
        args.nsample = 2
        print("SMOKE: 5 epochs, 1 batch/split, 2 samples", flush=True)
    data_dir = args.data_dir.resolve()
    for name in ["train_set.pkl", "val_set.pkl", "test_set.pkl", "scaler.pkl"]:
        if not (data_dir / name).is_file():
            raise FileNotFoundError(data_dir / name)
    output = args.output_dir.resolve()
    release = CONFIG_ROOT.parent
    if output == release or release in output.parents:
        raise ValueError("Output directory must be outside the repository")
    if (output / "metrics.json").exists() or (output / "model_best.pth").exists():
        raise FileExistsError(f"Experiment output files already exist: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_torch(args.seed)
    ratio = 0.20 if args.pattern == "point" else 0.05
    batch_size = config["train"]["batch_size"]
    train_loader = generate_train_dataloader(str(data_dir), 24, ratio, args.pattern, batch_size)
    val_loader = generate_val_test_dataloader(str(data_dir), 24, ratio, args.pattern, batch_size, "val")
    test_loader = generate_val_test_dataloader(str(data_dir), 24, ratio, args.pattern, batch_size, "test")
    if args.smoke:
        train_loader, val_loader, test_loader = [LimitedLoader(x, 1) for x in
                                                  [train_loader, val_loader, test_loader]]
    if min(len(train_loader), len(val_loader), len(test_loader)) == 0:
        raise ValueError("Each split must contain at least one valid window")
    with (data_dir / "scaler.pkl").open("rb") as f:
        mean, std = pickle.load(f)
    if np.shape(mean) != (DATASETS[args.dataset],) or np.shape(std) != np.shape(mean):
        raise ValueError("Scaler channel count does not match dataset")
    mean, std = [torch.from_numpy(x).to(args.device) for x in [mean, std]]
    effective = {"dataset": args.dataset, "pattern": args.pattern, "seed": args.seed,
                 "missing_ratio": ratio, "nsample": args.nsample, "smoke": args.smoke,
                 "max_batches_per_split": 1 if args.smoke else None,
                 "device": args.device, "config": copy.deepcopy(config)}
    (output / "effective_config.json").write_text(json.dumps(effective, indent=2) + "\n")
    model = MTSCI_Freq(config, args.device, target_dim=DATASETS[args.dataset], seq_len=24).to(args.device)
    checkpoint = args.checkpoint
    started = time.time()
    train_time = 0
    if checkpoint is None:
        _, _, train_time = train(model, config["train"], args, train_loader,
                                valid_loader=val_loader, foldername=str(output), current_time="best")
        checkpoint = output / "model_best.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=args.device))
    metrics = evaluate(model, test_loader, args.nsample, std, mean, str(output))
    metrics.update({"train_time": train_time, "elapsed_time": time.time() - started,
                    "dataset": args.dataset, "pattern": args.pattern, "seed": args.seed,
                    "nsample": args.nsample, "smoke": args.smoke,
                    "epochs": effective["config"]["train"]["epochs"],
                    "checkpoint": str(checkpoint.resolve())})
    if not all(np.isfinite(metrics[k]) for k in ["mae", "rmse", "mape", "mse", "r2", "CRPS"]):
        raise RuntimeError("Nonfinite evaluation metric")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    run(parser().parse_args())
