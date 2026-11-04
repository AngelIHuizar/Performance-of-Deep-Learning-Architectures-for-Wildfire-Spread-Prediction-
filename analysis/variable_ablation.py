# -*- coding: utf-8 -*-
"""
analysis/variable_ablation.py

Protocolo de ablacion de variables (plan de extension, Semana 1-2):
para cada grupo (historial de incendio, meteorologia, vegetacion,
topografia, poblacion), se ENMASCARA ese grupo (se reemplaza por su
valor promedio calculado sobre el train set) y se mide cuanto cae el
rendimiento del modelo ya entrenado, sin reentrenar.

Se corre sobre U-Net (mejor modelo espacial) y xLSTM (mejor modelo
recurrente, T=1) -- carga los checkpoints ya guardados en la Semana 1,
no entrena de nuevo.

IMPORTANTE: esto mide la sensibilidad del modelo YA ENTRENADO a cada
grupo (enmascarar en evaluacion), no que el modelo aprenda sin esa
variable desde cero. Es la opcion explicitamente permitida por el plan
("reentrene o enmascare"), mas rapida que reentrenar 10 veces.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from shared.config import (
    BATCH_SIZE, SEED, THRESHOLD, ONLY_FIRE_PIXELS, PREDICTIONS_PER_FIRE_DIR, FEATURE_RANGES
)
from shared.partitions import get_train_val_test_files
from shared.variable_groups import VARIABLE_GROUPS, _validar_grupos
from shared.evaluate import evaluate_model_unified, make_keras_predict_fn, make_pytorch_predict_fn

tf.config.optimizer.set_jit(False)
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

INPUT_FEATURES = ['elevation', 'wind_75', 'wind_avg', 'tmp_75', 'tmp_day', 'avg_sph',
                  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'viirs_PrevFireMask']
OUTPUT_FEATURES = ['viirs_FireMask']

_validar_grupos(INPUT_FEATURES)

train_files, validation_files, test_files = get_train_val_test_files()


def normalize_feature(feature, feature_name):
    if feature_name == 'viirs_PrevFireMask':
        return feature * 5.0
    feature_range = FEATURE_RANGES.get(feature_name)
    if feature_range is not None:
        feature_min, feature_max = feature_range
        norm = (feature - feature_min) / (feature_max - feature_min + 1e-6)
        return tf.clip_by_value(norm, 0.0, 1.0)
    else:
        return (feature - tf.reduce_mean(feature)) / (tf.math.reduce_std(feature) + 1e-6)


def _parse_function(example_proto):
    feature_description = {fn: tf.io.FixedLenFeature([64, 64], tf.float32)
                            for fn in INPUT_FEATURES + OUTPUT_FEATURES}
    ex = tf.io.parse_single_example(example_proto, feature_description)
    inputs_list = []
    for fn in INPUT_FEATURES:
        feat = tf.expand_dims(ex[fn], axis=-1)
        inputs_list.append(normalize_feature(feat, fn))
    x = tf.concat(inputs_list, axis=-1)  # (64,64,12), normalizado, orden = INPUT_FEATURES
    y = tf.expand_dims(tf.clip_by_value(ex["viirs_FireMask"], 0.0, 1.0), axis=-1)
    return x, y


def files_to_numpy(files):
    ds = tf.data.Dataset.list_files(files, shuffle=False)
    ds = ds.interleave(lambda f: tf.data.TFRecordDataset(f),
                        num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    X_list, Y_list = [], []
    for x, y in ds.as_numpy_iterator():
        X_list.append(x)
        Y_list.append(y)
    return np.stack(X_list, axis=0), np.stack(Y_list, axis=0)


# ---------------------------------------------------------------------
# 1. Calcular la media POR CANAL (normalizado) sobre el TRAIN set --
#    valor de reemplazo para enmascarar cada grupo
# ---------------------------------------------------------------------
print("Cargando train para calcular medias de enmascaramiento...")
X_train, _ = files_to_numpy(train_files)
print(f"Total muestras cargadas en X_train: {X_train.shape[0]}")
channel_means = X_train.mean(axis=(0, 1, 2))  # (12,) -- una media por canal
channel_stds = X_train.std(axis=(0, 1, 2))    # (12,) -- DEBUG: para descartar bug real
print("Medias y desviaciones por canal (orden = INPUT_FEATURES):")
for name, m, s in zip(INPUT_FEATURES, channel_means, channel_stds):
    print(f"  {name}: media={m:.4f}  std={s:.4f}")

print("Cargando test set...")
X_test, Y_test = files_to_numpy(test_files)
print(f"Test: {X_test.shape[0]} muestras")


def mask_group(X, group_name):
    """Devuelve una copia de X con los canales del grupo reemplazados
    por su media del train set (misma media para todas las muestras)."""
    X_masked = X.copy()
    for feat_name in VARIABLE_GROUPS[group_name]:
        idx = INPUT_FEATURES.index(feat_name)
        X_masked[..., idx] = channel_means[idx]
    return X_masked


# ---------------------------------------------------------------------
# 2. Cargar checkpoints ya entrenados (sin reentrenar)
# ---------------------------------------------------------------------
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_pred = tf.clip_by_value(y_pred, smooth, 1 - smooth)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2. * intersection + smooth) / (union + smooth)


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return tf.reduce_mean(0.5 * bce + 0.5 * dice_loss(y_true, y_pred))


def focal_tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), smooth, 1.0 - smooth)
    y_true_f = tf.reshape(y_true, [-1]); y_pred_f = tf.reshape(y_pred, [-1])
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return tf.pow((1.0 - tversky), gamma)


class IoUScore(tf.keras.metrics.Metric):
    def __init__(self, name="IoU_score", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total_iou = self.add_weight(name="total_iou", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, dtype=tf.float32)
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        intersection = tf.reduce_sum(y_true * y_pred, axis=1)
        union = tf.reduce_sum(y_true + y_pred, axis=1) - intersection
        self.total_iou.assign_add(tf.reduce_mean(intersection / (union + tf.keras.backend.epsilon())))
        self.count.assign_add(1.0)

    def result(self):
        return self.total_iou / tf.maximum(self.count, 1.0)


class DiceCoefficient(tf.keras.metrics.Metric):
    def __init__(self, name="dice_coefficient", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total_dice = self.add_weight(name="total_dice", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, dtype=tf.float32)
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        intersection = tf.reduce_sum(y_true * y_pred, axis=1)
        self.total_dice.assign_add(tf.reduce_mean(
            (2. * intersection) / (tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1) + tf.keras.backend.epsilon())))
        self.count.assign_add(1.0)

    def result(self):
        return self.total_dice / tf.maximum(self.count, 1.0)


print("\nCargando checkpoint de U-Net...")
unet_model = tf.keras.models.load_model(
    "best_unet_model.keras",
    custom_objects={
        "focal_tversky_loss": focal_tversky_loss,
        "combined_loss": combined_loss,
        "IoUScore": IoUScore,
        "DiceCoefficient": DiceCoefficient,
    },
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Cargando checkpoint de xLSTM (dispositivo: {device})...")

# La clase debe coincidir exactamente con la de models/train_xlstm.py
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig
import torch.nn as nn


class FrameEncoderTokens(nn.Module):
    def __init__(self, in_ch=12, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, d, 3, padding=1), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MaskDecoder(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(d, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, z):
        return torch.sigmoid(self.net(z))


class FireXLSTM_Tokens(nn.Module):
    def __init__(self, T, in_ch=12, d=64, d_model=128, n_blocks=4):
        super().__init__()
        self.T = T
        self.d = d
        self.enc = FrameEncoderTokens(in_ch=in_ch, d=d)
        cfg = xLSTMBlockStackConfig(
            context_length=T, num_blocks=n_blocks, embedding_dim=d_model,
            mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)),
            slstm_at=[]
        )
        self.in_proj = nn.Linear(d, d_model)
        self.xlstm = xLSTMBlockStack(cfg)
        self.out_proj = nn.Linear(d_model, d)
        self.dec = MaskDecoder(d=d)

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        feats = torch.stack([self.enc(x[:, t]) for t in range(T)], dim=1)
        feats = feats.permute(0, 1, 3, 4, 2).contiguous()
        S = 16 * 16
        feats = feats.view(B, T, S, self.d)
        seq = feats.permute(0, 2, 1, 3).contiguous().view(B * S, T, self.d)
        seq = self.in_proj(seq)
        y = self.xlstm(seq)
        h = y[:, -1, :]
        tok = self.out_proj(h)
        grid = tok.view(B, S, self.d).view(B, 16, 16, self.d).permute(0, 3, 1, 2).contiguous()
        return self.dec(grid)


xlstm_model = FireXLSTM_Tokens(T=1, in_ch=12, d=64, d_model=128, n_blocks=4).to(device)
xlstm_model.load_state_dict(torch.load("best_xlstm_model.pt", map_location=device))
xlstm_model.eval()


# ---------------------------------------------------------------------
# 3. Evaluar baseline (sin enmascarar) y cada grupo enmascarado
# ---------------------------------------------------------------------
def eval_keras_per_sample(X, Y):
    """Devuelve un array con el IoU de CADA muestra individual (no el promedio),
    necesario para poder calcular el IC bootstrap de la caida, pareado por muestra."""
    preds = unet_model.predict(X, verbose=0, batch_size=BATCH_SIZE)
    preds_bin = (preds > THRESHOLD).astype("int32")
    y_bin = Y.astype("int32")
    tp = np.sum((y_bin == 1) & (preds_bin == 1), axis=(1, 2, 3))
    fp = np.sum((y_bin == 0) & (preds_bin == 1), axis=(1, 2, 3))
    fn = np.sum((y_bin == 1) & (preds_bin == 0), axis=(1, 2, 3))
    return tp / (tp + fp + fn + 1e-7)


def eval_xlstm_per_sample(X, Y):
    """Igual que arriba, pero para el modelo xLSTM en PyTorch."""
    X_seq = np.expand_dims(X, axis=1)
    ious = []
    with torch.no_grad():
        for i in range(0, len(X_seq), BATCH_SIZE):
            xb = torch.from_numpy(X_seq[i:i + BATCH_SIZE]).float().to(device)
            pred = xlstm_model(xb).cpu().numpy()
            pred_bin = (pred > THRESHOLD).astype("int32")
            yb = Y[i:i + BATCH_SIZE].transpose(0, 3, 1, 2).astype("int32")
            tp = np.sum((yb == 1) & (pred_bin == 1), axis=(1, 2, 3))
            fp = np.sum((yb == 0) & (pred_bin == 1), axis=(1, 2, 3))
            fn = np.sum((yb == 1) & (pred_bin == 0), axis=(1, 2, 3))
            ious.append(tp / (tp + fp + fn + 1e-7))
    return np.concatenate(ious)


def bootstrap_ci_diff(baseline_per_sample, masked_per_sample, n_boot=10_000, ci=0.95, seed=42):
    """IC bootstrap de la diferencia PAREADA (baseline - masked), muestra por
    muestra -- mismo criterio que el resto del articulo (bootstrap_stratification.py)."""
    rng = np.random.default_rng(seed)
    n = len(baseline_per_sample)
    diffs = baseline_per_sample - masked_per_sample
    boot_means = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    alpha = 1 - ci
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return diffs.mean(), lo, hi


print("\nEvaluando baseline (sin enmascarar)...")
baseline_unet_per_sample = eval_keras_per_sample(X_test, Y_test)
baseline_xlstm_per_sample = eval_xlstm_per_sample(X_test, Y_test)
baseline_unet = baseline_unet_per_sample.mean()
baseline_xlstm = baseline_xlstm_per_sample.mean()
print(f"IoU baseline U-Net: {baseline_unet:.4f}")
print(f"IoU baseline xLSTM: {baseline_xlstm:.4f}")

resultados = [
    {"grupo": "(baseline, sin enmascarar)", "modelo": "unet", "iou": baseline_unet,
     "caida": 0.0, "ci_low": 0.0, "ci_high": 0.0},
    {"grupo": "(baseline, sin enmascarar)", "modelo": "xlstm", "iou": baseline_xlstm,
     "caida": 0.0, "ci_low": 0.0, "ci_high": 0.0},
]

for grupo in VARIABLE_GROUPS:
    print(f"\nEnmascarando grupo: {grupo} ({VARIABLE_GROUPS[grupo]})")
    X_masked = mask_group(X_test, grupo)

    unet_masked_per_sample = eval_keras_per_sample(X_masked, Y_test)
    xlstm_masked_per_sample = eval_xlstm_per_sample(X_masked, Y_test)

    iou_unet = unet_masked_per_sample.mean()
    iou_xlstm = xlstm_masked_per_sample.mean()

    caida_unet, ci_lo_unet, ci_hi_unet = bootstrap_ci_diff(
        baseline_unet_per_sample, unet_masked_per_sample)
    caida_xlstm, ci_lo_xlstm, ci_hi_xlstm = bootstrap_ci_diff(
        baseline_xlstm_per_sample, xlstm_masked_per_sample)

    print(f"  U-Net : IoU={iou_unet:.4f}  caida={caida_unet:+.4f}  IC=[{ci_lo_unet:.4f}, {ci_hi_unet:.4f}]")
    print(f"  xLSTM : IoU={iou_xlstm:.4f}  caida={caida_xlstm:+.4f}  IC=[{ci_lo_xlstm:.4f}, {ci_hi_xlstm:.4f}]")

    resultados.append({"grupo": grupo, "modelo": "unet", "iou": iou_unet,
                        "caida": caida_unet, "ci_low": ci_lo_unet, "ci_high": ci_hi_unet})
    resultados.append({"grupo": grupo, "modelo": "xlstm", "iou": iou_xlstm,
                        "caida": caida_xlstm, "ci_low": ci_lo_xlstm, "ci_high": ci_hi_xlstm})

df_resultados = pd.DataFrame(resultados)
print("\n" + "=" * 60)
print("TABLA DE ABLACION COMPLETA (con IC bootstrap 95%)")
print("=" * 60)
print(df_resultados.round(4).to_string(index=False))

os.makedirs(PREDICTIONS_PER_FIRE_DIR, exist_ok=True)
out_path = os.path.join(PREDICTIONS_PER_FIRE_DIR, "variable_ablation_results.csv")
df_resultados.to_csv(out_path, index=False)
print(f"\nResultados guardados en: {out_path}")
