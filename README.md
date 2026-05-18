# RAPID-Hydrology: Pre-training Hydro-Meteorological Representations with Diffusion for Enhanced Causal Prediction

**RAPID-Hydrology** (Relative-position Autoregressive Pre-training with Diffusion for Hydrology) is a diffusion-based generative pre-training and transfer learning framework designed for hydrological time series forecasting in data-scarce basins.

> Paper: *RAPID-Hydrology: Pre-training Hydro-Meteorological Representations with Diffusion for Enhanced Causal Prediction* (IEEE TGRS)

## Overview

Accurate hydrological variable prediction at the basin scale is fundamental to water resource management, yet remains challenging due to complex temporal dependencies and limited in-situ observations. RAPID-Hydrology addresses these challenges through a two-stage learning paradigm:

1. **Knowledge Condensation (Pre-training)**: A domain-specific diffusion model, **HydroDiffuser**, is pre-trained on large-scale hydro-meteorological reanalysis data via a conditional denoising reconstruction task. The model employs an *Attenuation-aware Relative Position Causal Encoder (RPCE-Encoder)* with Rotary Position Embedding (RoPE) and causal masking, paired with a Denoising Diffusion Probabilistic Model (DDPM) decoder. After pre-training, the diffusion decoder is discarded, and only the RPCE-Encoder is retained as a transferable feature extractor.

2. **Knowledge Rapid Injection (Fine-tuning)**: The pre-trained encoder is transferred to the target basin (e.g., Danjiangkou Reservoir Basin) with limited observations. A lightweight prediction head is appended, and the model is fine-tuned using a *Dual Sliding-Window Sampling* strategy to maximize data utilization.

## Key Features

- **Causality-sensitive temporal modeling**: RoPE-based relative position encoding naturally captures the decaying influence of historical events with increasing time lag, incorporating hydrological physical priors
- **Diffusion-based generative pre-training**: Learns robust, transferable representations by reconstructing progressively perturbed target sequences conditioned on encoded context
- **Data-scarce basin adaptation**: Transfers pre-trained knowledge to target basins with limited observations via lightweight fine-tuning
- **Multi-variable joint prediction**: Simultaneously predicts four hydro-meteorological variables — Precipitation (Pre), Mean Temperature (Tm), Wind Speed (Win)

## Quick Start

### 1. Environment Setup

```bash
# Python 3.10+
conda create -n rapid-hydrology python=3.10 -y
conda activate rapid-hydrology

# Clone repository
git clone https://github.com/Kilimetarlop/RAPID-Hydrology.git
cd RAPID-Hydrology

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Data consists of CSV files, each representing a daily time series at a fixed geographic grid point. Filenames follow the convention `{latitude}_{longitude}.csv`.

**CSV format requirements**:
- The last four columns must be: `Pre`, `Tm`, `Win`, `Rhu`
- The first column is typically a date (YYYYMMDD format)
- One time step per row

Example (`dataset/train_dataset/31.5_110.0.csv`):
```
,Pre,Tm,Win,Rhu
19610101,0.0,0.356,2.475,64.662
19610102,0.0,0.151,1.650,71.172
...
```

**Dataset splitting** (using `split_data.py`):
```bash
python split_data.py
```
Edit the configuration variables at the top of the script: `source_dir` (raw data directory), `keep_rows` (number of rows for training set), and output directories.

### 3. Pre-training (Self-supervised)

```bash
python run.py \
    --task_name pretrain \
    --model RAPID-Hydrology \
    --data Water \
    --root_path ./dataset \
    --data_path train_dataset \
    --features M \
    --enc_in 4 --c_out 4 \
    --seq_len 336 --patch_len 12 --stride 12 \
    --d_model 512 --d_ff 2048 --n_heads 8 --e_layers 2 --d_layers 1 \
    --train_epochs 10 --batch_size 32 --learning_rate 0.0001 \
    --time_steps 1000 --scheduler cosine --mask_ratio 0.5 \
    --use_norm 1 --num_workers 4
```

Checkpoints are saved to `./outputs/pretrain_checkpoints/Water/ckpt_best.pth`.

### 4. Fine-tuning (Supervised Prediction)

```bash
python run.py \
    --task_name finetune \
    --model RAPID-Hydrology \
    --data Water \
    --root_path ./dataset \
    --data_path train_dataset \
    --features M \
    --enc_in 4 --c_out 4 \
    --seq_len 336 --pred_len 96 \
    --patch_len 12 --stride 12 \
    --d_model 512 --d_ff 2048 --n_heads 8 --e_layers 2 --d_layers 1 \
    --train_epochs 20 --batch_size 32 --learning_rate 0.0001 \
    --use_norm 1 --num_workers 4 \
    --pretrain_checkpoints ./outputs/pretrain_checkpoints \
    --transfer_checkpoints ckpt_best.pth
```

Test results are output to `./outputs/test_results/Water/`:
- `detailed_predictions.csv` — per-sample, per-timestep, per-feature predictions vs ground truth
- `sample_metrics.csv` — per-sample aggregated MSE/MAE
- `feature_metrics.csv` — per-feature aggregated metrics

### 5. Standalone Inference

```bash
python standalone_test_save.py \
    --csv_dir ./dataset/test_dataset \
    --input_len 336 --pred_len 96 \
    --features M \
    --ckpt ./outputs/checkpoints/<setting>/checkpoint.pth \
    --model RAPID-Hydrology \
    --d_model 512 \
    --out_dir ./outputs/inference_results
```

## Key Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `--task_name` | pretrain / finetune | pretrain |
| `--model` | Model name | RAPID-Hydrology |
| `--data` | Dataset identifier | Water |
| `--root_path` | Data root directory | ./dataset |
| `--data_path` | Data subdirectory | train_dataset |
| `--enc_in` | Number of input features | 4 |
| `--c_out` | Number of output features | 4 |
| `--seq_len` | Input sequence length | 336 |
| `--pred_len` | Prediction sequence length | 96 |
| `--patch_len` | Patch length | 12 |
| `--stride` | Patch stride | 12 |
| `--d_model` | Model dimension | 512 |
| `--time_steps` | Diffusion steps (pretrain only) | 1000 |
| `--mask_ratio` | Decoder partial mask ratio (pretrain only) | 0.5 |
| `--transfer_checkpoints` | Pretrained weights for fine-tuning | ckpt_best.pth |

## Evaluation Metrics

Test-set evaluation uses MSE and MAE.

The output file `detailed_predictions.csv` can be consumed by scripts in `visual/` to generate:
- **ACC** (Anomaly Correlation Coefficient): prediction anomaly correlation at the timestep level
- **MSSS** (Mean Squared Skill Score): prediction skill score at the spatial grid level
- **MAPE** (Mean Absolute Percentage Error): percentage error spatial distribution

## Project Structure

```
RAPID-Hydrology/
├── run.py                           # Main entry point (pre-training + fine-tuning)
├── standalone_test_save.py          # Standalone inference script
├── split_data.py                    # Dataset splitting utility
├── requirements.txt
├── README.md
├── docs/
│   └── USAGE.md                     # Detailed usage guide
├── models/
│   └── rapid_hydrology.py           # RAPID-Hydrology model definition
├── layers/
│   ├── rapid_hydrology_encdec.py    # Core encoder-decoder modules (RPCE-Encoder, Diffusion, DenoisingPatchDecoder)
│   ├── Embed.py                     # Patch embedding and positional encoding (including Rotary Position Encoding)
│   ├── SelfAttention_Family.py      # Attention variants
│   └── Transformer_EncDec.py        # Standard Transformer encoder-decoder blocks
├── exp/
│   ├── exp_basic.py                 # Base experiment class
│   └── exp_rapid_hydrology.py       # RAPID-Hydrology training/evaluation loop
├── data_provider/
│   ├── data_factory.py              # Sliding-window dataset (Dataset_MyData) and data provider
│   ├── data_loader.py               # Standard dataset base classes
│   └── multi_folder_csv_provider.py # Multi-folder CSV data provider
├── utils/
│   ├── tools.py                     # EarlyStopping, weight transfer, learning rate adjustment
│   ├── metrics.py                   # MSE, MAE, RMSE, and other evaluation metrics
│   ├── losses.py                    # Loss functions
│   ├── masking.py                   # Causal and partial attention masks
│   ├── timefeatures.py              # Time feature encoding
│   └── augmentations.py             # Data augmentation utilities
├── visual/
│   ├── 1_.py                        # ACC time series analysis
│   ├── 2_.py                        # MSSS spatial analysis + noise comparison
│   └── 3_MAPE.py                    # MAPE spatial analysis + noise comparison
└── dataset/
    ├── train_dataset/               # Training data (CSV)
    └── test_dataset/                # Test data (CSV)
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{zhang2026rapid,
  title={RAPID-Hydrology: Pre-training Hydro-Meteorological Representations with Diffusion for Enhanced Causal Prediction},
  author={Zhang, Xinbin and Tang, Tiantian and Gui, Guan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026}
}
```

## License

This project is released under the MIT License.
