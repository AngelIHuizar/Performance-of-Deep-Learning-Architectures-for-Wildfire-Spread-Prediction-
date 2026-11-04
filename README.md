# Performance-of-Deep-Learning-Architectures-for-Wildfire-Spread-Prediction-
Code for next-day wildfire spread prediction on the modified NDWS dataset: MobileNetV2, U-Net, ConvLSTM and xLSTM, with fire-size stratification, seeds, bootstrap statistics and variable ablation.
Code accompanying the article **"Performance of Deep Learning Architectures for Wildfire Spread Prediction Considering Fire Size and Environmental Variables"** (*International Journal of Combinatorial Optimization Problems and Informatics*, IJCOPI; under revision). The article extends the MCPR 2026 conference paper (Huizar Bretado et al., 2026).

The repository trains and evaluates four architectures on the *modified Next Day Wildfire Spread* dataset, and reproduces the analyses of the article: fixed validation-selected thresholds, three training seeds, per-sample metrics with paired bootstrap, stratification by fire size, a persistence baseline and an input-variable ablation.

> **Scope.** ConvLSTM and xLSTM are evaluated as **single-step (T = 1)** models, because the dataset has no date or location per sample and temporal continuity between samples cannot be verified. The results therefore say nothing about the benefit of recurrent memory over real time series.

## Repository layout

```
shared/     configuration, data partitions, unified evaluation, variable groups
models/     training scripts (one per architecture; seed given on the command line)
analysis/   scripts that produce the tables and figures of the article
legacy/     earlier versions of analyses, kept for provenance (not needed to reproduce the article)
dev/        data-inspection and debugging utilities
```

| Name in the article | Training script | Framework |
|---|---|---|
| MobileNetV2 | `models/train_autoencoder.py` (MobileNetV2 encoder + convolutional decoder) | TensorFlow / Keras |
| U-Net | `models/train_unet.py` | TensorFlow / Keras |
| ConvLSTM | `models/train_convlstm.py` | TensorFlow / Keras |
| xLSTM | `models/train_xlstm.py` | PyTorch (`xlstm` package) |

## Data

The experiments use the **Modified Next Day Wildfire Spread** dataset (Hulsey, 2024), derived from the Next Day Wildfire Spread dataset (Huot et al., 2022):

1. Download it from <https://www.kaggle.com/datasets/georgehulsey/modified-next-day-wildfire-spread>.
2. Put the `*.tfrecord` files in one folder. The partitions are resolved from the file names (`train`, `eval`, `test`; see `shared/partitions.py`).
3. Point the code to that folder: `export WILDFIRE_DATASET_PATH=/path/to/folder` (Windows PowerShell: `$env:WILDFIRE_DATASET_PATH="C:\path\to\folder"`).

The data are **not** redistributed here. Partition sizes: 9,600 training, 4,301 validation and 6,196 test samples. Samples carry no date or location identifier, so spatial or temporal disjointness between partitions cannot be verified.

## Environment

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended. Inference times reported in the article were measured on an NVIDIA RTX 5050 GPU. Package versions: see `requirements.txt`.

## Reproducing the results

Run everything from the repository root. Each training script receives the seed as its only argument; the article uses seeds 1001, 2002 and 3003.

```bash
# 1. Train the four models with the three seeds (checkpoints are written to the working directory)
for s in 1001 2002 3003; do
  python -m models.train_unet $s
  python -m models.train_autoencoder $s      # MobileNetV2
  python -m models.train_convlstm $s
  python -m models.train_xlstm $s
done

# 2. Main table: validation-selected thresholds, per-sample metrics, mean ± SD over seeds, bootstrap CIs
python -m analysis.seeds_tuned_evaluation

# 3. Stratification by fire size (IoU and threshold-independent AP) and forest plot
python -m analysis.stratification_tuned
python -m analysis.stratification_ap
python -m analysis.forest_plot_tuned_en

# 4. Architecture table (parameters, FLOPs, inference time), persistence baseline, Figure 1
python -m analysis.architecture_table
python -m analysis.persistence_baseline
python -m analysis.make_figure1

# 5. Variable ablation (U-Net and xLSTM)
python -m analysis.variable_ablation
```

Outputs go to `outputs/` (per-sample CSV files in `outputs/predictions_per_fire/`).

Notes on the evaluation protocol:

- The CSV files written by the `models/train_*.py` scripts use the default threshold of 0.5 from `shared/config.py`. The numbers of the article come from `analysis/seeds_tuned_evaluation.py`, which uses the fixed thresholds chosen on the validation set: U-Net 0.45, MobileNetV2 0.25, ConvLSTM 0.30 and xLSTM 0.30.
- `analysis/variable_ablation.py` loads `best_unet_model.keras` and `best_xlstm_model.pt` (checkpoints without a seed suffix), and uses a threshold of 0.5.
- `analysis/temporal_demo.py` is an exploratory control experiment (ConvLSTM with T > 1 on a dataset without temporal continuity). It is not part of the article's results.

## Training configuration

All models use the loss 0.5·BCE + 0.5·Dice and the same inputs, normalisation and partitions. Hyperparameters were set by manual adjustment starting from the public code of the original dataset study; no grid or random search was run.

| Model | Optimiser | Batch | Max. epochs | Early stopping / schedule |
|---|---|---|---|---|
| MobileNetV2 | Adam, lr 1e-3, clipnorm 1.0 | 32 | 50 | patience 4 on validation Dice; lr × 0.7 after 2 epochs without improvement |
| U-Net | Adam, lr 1e-3, clipnorm 1.0 | 32 | 30 | same as above |
| ConvLSTM | Adam, lr 1e-3, clipnorm 1.0 | 32 | 50 | same as above |
| xLSTM | Adam, lr 1e-4 | 8 | 50 | patience 5 on validation loss; no lr schedule or gradient clipping |

The xLSTM was therefore **not** trained with the same hyperparameters as the other models.

## Known limitations

- Single-step evaluation only (T = 1), see above.
- Input normalisation uses the fixed per-variable ranges in `shared/config.py` (`FEATURE_RANGES`), inherited from the original code and clipped to [0, 1]. Some ranges are narrower than the observed data range, so part of the values of some channels (e.g. elevation, specific humidity) saturate at the limits.
- Partitions cannot be checked for spatial or temporal overlap.

## Citation

If you use this code, please cite the article and, until it is published, this repository (see `CITATION.cff`).

## License

See `LICENSE`.

## References

- Huot, F., Hu, R. L., Goyal, N., Sankar, T. P., Ihme, M., & Chen, Y.-F. (2022). Next day wildfire spread: A machine learning dataset to predict wildfire spreading from remote-sensing data. *IEEE Transactions on Geoscience and Remote Sensing, 60*, 1–13. https://doi.org/10.1109/TGRS.2022.3192974
- Hulsey, G. (2024). *Modified Next Day Wildfire Spread* [Data set]. Kaggle. https://www.kaggle.com/datasets/georgehulsey/modified-next-day-wildfire-spread
