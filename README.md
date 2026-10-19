# RAPID-Hydrology: Pretraining Hydro-Meteorological Representations With Diffusion for Enhanced Causal Prediction

Core implementation of **RAPID-Hydrology**, published in **IEEE Transactions on Geoscience and Remote Sensing**, vol. 64, Art. no. 4212113, 2026.

**Xinbin Zhang, Tiantian Tang, and Guan Gui**

[IEEE Xplore](https://ieeexplore.ieee.org/document/11684875/) | [DOI](https://doi.org/10.1109/TGRS.2026.3732149) | [Citation](#citation)

## Overview

RAPID-Hydrology is a large-source/small-target transfer framework for long-horizon hydro-meteorological forecasting in basins with limited local observations. It extends the **TimeDART diffusion-autoregressive backbone** through national-scale hydro-meteorological pretraining, relative-position causal encoding, and sample-efficient target-basin adaptation.

1. **Knowledge condensation (pretraining).** HydroDiffuser trains a patch-based causal encoder and a conditional diffusion decoder on large-scale national hydro-meteorological sequences. The Relative Position Causal Encoding (RPCE) encoder combines Rotary Position Embedding (RoPE) with causal attention masking. Conditional denoising learns representations for subsequent transfer.
2. **Knowledge rapid injection (fine-tuning).** The diffusion decoder is discarded. The pretrained encoder and a lightweight prediction head are jointly fine-tuned using the limited Danjiangkou Reservoir basin record. Coupled lookback and forecast windows generate chronologically ordered supervised pairs. Downstream forecasting is deterministic and requires no iterative diffusion sampling.

Here, **causal** means temporally ordered information flow without access to future observations at each encoder position. RoPE encodes relative temporal offsets; it does not by itself impose monotonically decaying attention weights or identify physical causal relations.

## Key Features

- **Hydrology-oriented representation transfer:** national-scale source pretraining followed by adaptation to a data-scarce basin.
- **Relative-position causal encoding:** RoPE represents temporal lags, while causal masking preserves forecasting order.
- **Diffusion during pretraining:** the denoising objective trains the transferable encoder; downstream inference uses the encoder and prediction head.
- **Sample-efficient adaptation:** paired lookback and forecast windows extract overlapping supervised examples from a limited local record.
- **Long-horizon prediction:** the paper evaluates deterministic 365-step forecasts for precipitation, mean temperature, and wind speed.

## Published Results

For the paper's 12-month average **precipitation** results, RAPID-Hydrology reduces MSE, MAE, and RMSE by **98.32%**, **95.77%**, and **86.08%**, respectively, relative to CNN-BiLSTM-MTS, the strongest evaluated from-scratch comparator. These reductions refer to this specific comparison. Additional evaluations include sMAPE, MSSS, ACC, spatial skill maps, and prediction-residual diagnostics.

The repository's CSV interface uses four feature columns (`Pre`, `Tm`, `Win`, and `Rhu`), while the published forecasting results focus on the first three variables. The commands below are configurable usage examples with a 96-step forecast horizon, rather than the complete configuration for reproducing the paper's 365-step experiments.

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

If you use this implementation, please cite the published paper:

> X. Zhang, T. Tang, and G. Gui, “RAPID-Hydrology: Pretraining hydro-meteorological representations with diffusion for enhanced causal prediction,” *IEEE Transactions on Geoscience and Remote Sensing*, vol. 64, Art. no. 4212113, 2026, doi: 10.1109/TGRS.2026.3732149.

```bibtex
@article{zhang2026rapid,
  author  = {Zhang, Xinbin and Tang, Tiantian and Gui, Guan},
  title   = {{RAPID-Hydrology}: Pretraining Hydro-Meteorological Representations With Diffusion for Enhanced Causal Prediction},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  year    = {2026},
  volume  = {64},
  pages   = {4212113--4212113},
  doi     = {10.1109/TGRS.2026.3732149},
  url     = {https://ieeexplore.ieee.org/document/11684875/}
}
```

The BibTeX page field follows IEEE Xplore's exported article-number format. In formatted prose, use **Art. no. 4212113**.

## License

This project is released under the MIT License.
