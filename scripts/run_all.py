#!/usr/bin/env python3
"""Run 8 datasets x 2 patterns, with explicit data/output roots."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "ExchangeRate", "Weather", "METR-LA", "Electricity"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seeds", nargs="+", type=int, default=[1])
    p.add_argument("--threads", type=int)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    for dataset in DATASETS:
        data = a.data_root.resolve() / dataset
        if (data / "data").is_dir():
            data = data / "data"
        for pattern in ["point", "block"]:
            for seed in a.seeds:
                command = [sys.executable, str(ROOT / "main.py"), "--dataset", dataset,
                           "--pattern", pattern, "--data-dir", str(data), "--output-dir",
                           str(a.output_root.resolve() / f"{dataset}_{pattern}" / f"seed{seed}"),
                           "--seed", str(seed), "--device", a.device]
                if a.smoke:
                    command += ["--smoke"]
                if a.threads:
                    command += ["--threads", str(a.threads)]
                print(subprocess.list2cmdline(command), flush=True)
                if not a.dry_run:
                    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
