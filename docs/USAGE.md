# RAPID-Hydrology — Detailed Usage Guide

## 1. Data Preprocessing

### Data Format

Each CSV file represents a daily meteorological time series at a fixed geographic location. Filenames follow the convention `{latitude}_{longitude}.csv`.

**Required columns** (last 4 columns):

| Column | Variable | Unit |
|--------|----------|------|
| Pre  | Precipitation | mm |
| Tm   | Mean Temperature | degC |
| Win  | Wind Speed | m/s |
| Rhu  | Relative Humidity | % |

### Data Splitting

Use `split_data.py` to split raw long sequences chronologically into training and test sets.

Edit the configuration variables at the top of the script:
- `source_dir`: directory containing raw CSV files
- `target_parent_A`: training set output directory
- `target_parent_B`: test set output directory
- `keep_rows`: number of rows to keep in the training set (default 15088)

```bash
python split_data.py
```

### Data Loading Strategies

The project provides two data loading approaches:

**Approach A: DirectLoad** (root-level `data_factory.py`)
- Each CSV serves as an independent sample, directly slicing the first `input_len + pred_len` rows as one (x, y) pair
- Suitable for scenarios with many grid points, each with moderate temporal length

**Approach B: SlidingWindow** (`data_provider/data_factory.py`)
- Each CSV generates multiple (x, y) pairs via a sliding window
- Test set defaults to the last window only (`test_no_sliding=True`) to prevent data leakage
- Suitable for scenarios with a single station and very long sequences

To switch between approaches, modify the import source in `exp/exp_rapid_hydrology.py` line 1.

## 2. Model Architecture

RAPID-Hydrology uses a Causal Transformer with Rotary Position Embedding (RoPE) as the encoder (RPCE-Encoder), combined with a Denoising Diffusion Probabilistic Model (DDPM) as the decoder during pre-training. The parameter count is approximately 10–20M, suitable for single-GPU training.

During inference, only the RPCE-Encoder and the lightweight prediction head are used — the diffusion decoder is discarded after pre-training.

## 3. Custom Training

### Common Hyperparameter Adjustments

```bash
# Longer input sequence
--seq_len 512 --patch_len 16 --stride 16

# Larger model
--d_model 768 --d_ff 3072 --n_heads 12

# Longer prediction horizon
--pred_len 192

# Adjust learning rate and training epochs
--learning_rate 0.00005 --train_epochs 30 --patience 5
```

### Using Custom Datasets

1. Prepare CSV files in the format described above under `dataset/<your_data>/`
2. Specify `--root_path ./dataset --data_path <your_data> --data <YourDataName>`
3. Ensure `--enc_in` and `--c_out` match the number of feature columns

## 4. Standalone Inference

```bash
python standalone_test_save.py \
    --csv_dir ./dataset/test_dataset \
    --input_len 336 \
    --pred_len 96 \
    --features M \
    --ckpt ./outputs/pretrain_checkpoints/Water/ckpt_best.pth \
    --model RAPID-Hydrology \
    --d_model 512 \
    --out_dir ./outputs/inference
```

### Output File Description

`detailed_predictions.csv`:

| Column | Description |
|--------|-------------|
| sample_id | Sample index (corresponding to CSV file index) |
| timestep | Prediction timestep (0 ~ pred_len-1) |
| feature_index | Feature index (0=Pre, 1=Tm, 2=Win, 3=Rhu) |
| target | Ground truth value |
| prediction | Predicted value |

An `pred_true_lastwindow.npz` file (NumPy format) is also generated for convenient loading.

## 5. Result Visualization

Before running, edit the path variables in each script to point to the actual prediction result files and data directories.

```bash
# ACC time series analysis (prediction anomaly correlation curves per feature + noise comparison)
python visual/1_.py

# MSSS spatial analysis (prediction skill score heatmaps by month + noise robustness)
python visual/2_.py

# MAPE spatial analysis (percentage error heatmaps by month + noise robustness)
python visual/3_MAPE.py
```

## 6. FAQ

**Q: CUDA out of memory**

Reduce `--batch_size` or `--d_model`, or shorten `--seq_len`.

**Q: ImportError: No module named 'data_provider.data_factory'**

Add the project root directory to PYTHONPATH:
```bash
export PYTHONPATH=.    # Linux/Mac
set PYTHONPATH=.       # Windows
```

**Q: Pre-training loss not decreasing**

Verify that data normalization is enabled (`--use_norm 1`). Try lowering `--learning_rate` or increasing `--batch_size`.

**Q: How to resume training from a saved checkpoint**

In fine-tuning mode, specify the pretrained weight path:
```bash
--pretrain_checkpoints ./outputs/pretrain_checkpoints \
--transfer_checkpoints ckpt_best.pth
```
