# HeteroTSI

Official implementation of **HeteroTSI: Channel-Heterogeneity-Aware Diffusion for Multivariate Time Series Imputation**.

HeteroTSI is a conditional diffusion method that combines Data-Adaptive Channel
Embedding (DACE), Affinity-Gated Channel Attention (AGCA), and Channel
Interpolation Consistency (CIC) for multivariate time series imputation.

## Installation

Use Python 3.10 and PyTorch 1.13.1. For CUDA 11.6:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu116 -r requirements.txt
```

## Data preparation

Supported datasets: ETTh1, ETTh2, ETTm1, ETTm2, ExchangeRate, Weather, METR-LA,
and Electricity. Place prepared data under `/data/prepared/<Dataset>/`:

```text
/data/prepared/ETTh1/
├── train_set.pkl
├── val_set.pkl
├── test_set.pkl
└── scaler.pkl
```

Each split contains a standardized float32 NumPy array of shape `[time, channels]`,
with original missing values stored as NaN. `scaler.pkl` contains `[mean, std]`,
two per-channel NumPy arrays. To prepare a raw `[time, channels]` NumPy array:

```bash
python scripts/prepare_data.py --dataset ETTh1 --input /data/ETTh1.npy --output-dir /data/prepared/ETTh1
```

## Run experiments

Train and evaluate one dataset with point missingness:

```bash
python main.py --dataset ETTh1 --pattern point --data-dir /data/prepared/ETTh1 --output-dir /results/ETTh1-point --device cuda:0 --seed 1
```

Use `--pattern block` for block missingness. The corresponding configuration in
`configs/` is selected automatically. Output directories must be outside the repository.

Run all eight datasets under both missingness patterns:

```bash
python scripts/run_all.py --data-root /data/prepared --output-root /results/heterotsi --device cuda:0
```

MAE and RMSE are saved in `metrics.json` under each experiment's output directory.

## Citation

If you use this code in your research, please cite **HeteroTSI: Channel-Heterogeneity-Aware Diffusion for Multivariate Time Series Imputation**.

## Acknowledgements and license

This implementation builds on [MTSCI](https://github.com/JeremyChou28/MTSCI).
The code is available under the [MIT License](LICENSE).
