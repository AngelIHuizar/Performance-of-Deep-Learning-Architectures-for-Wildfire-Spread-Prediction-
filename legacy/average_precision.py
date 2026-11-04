# -*- coding: utf-8 -*-
"""
analysis/average_precision.py

Calcula Average Precision (AP) -- area bajo la curva precision-exhaustividad --
para los 4 modelos sobre el test set. A diferencia de IoU/F1 (que dependen
del umbral elegido), AP evalua el modelo sobre TODOS los umbrales posibles
a la vez, dando una comparacion de arquitectura pura, sin el factor de en
que punto de operacion se evalua cada una.

Metodologia: AP por MUESTRA (macro), consistente con el resto del proyecto.
Se excluyen del promedio las muestras sin ningun pixel de incendio real
(AP no esta definido si no hay clase positiva) -- se reporta cuantas se
excluyeron, para que sea auditable.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from shared.config import BATCH_SIZE, PREDICTIONS_PER_FIRE_DIR, FEATURE_RANGES
from shared.partitions import get_train_val_test_files

tf.config.optimizer.set_jit(False)
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

INPUT_FEATURES = ['elevation', 'wind_75', 'wind_avg', 'tmp_75', 'tmp_day', 'avg_sph',
                  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'viirs_PrevFireMask']
OUTPUT_FEATURES = ['viirs_FireMask']

_, _, test_files = get_train_val_test_files()


def normalize_feature(feature, feature_name):
    if feature_name == 'viirs_PrevFireMask':
        return feature * 5.0
    feature_range = FEATURE_RANGES.get(feature_name)
    if feature_range is not None:
        fmin, fmax = feature_range
        return tf.clip_by_value((feature - fmin) / (fmax - fmin + 1e-6), 0.0, 1.0)
    return (feature - tf.reduce_mean(feature)) / (tf.math.reduce_std(feature) + 1e-6)


def _parse_function(example_proto):
    fd = {fn: tf.io.FixedLenFeature([64, 64], tf.float32) for fn in INPUT_FEATURES + OUTPUT_FEATURES}
    ex = tf.io.parse_single_example(example_proto, fd)
    inputs_list = [normalize_feature(tf.expand_dims(ex[fn], -1), fn) for fn in INPUT_FEATURES]
    x = tf.concat(inputs_list, axis=-1)
    y = tf.expand_dims(tf.clip_by_value(ex["viirs_FireMask"], 0.0, 1.0), axis=-1)
    return x, y


def files_to_numpy(files):
    ds = tf.data.Dataset.list_files(files, shuffle=False)
    ds = ds.interleave(lambda f: tf.data.TFRecordDataset(f), num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    X, Y = [], []
    for x, y in ds.as_numpy_iterator():
        X.append(x); Y.append(y)
    return np.stack(X, 0), np.stack(Y, 0)


print("Cargando test set...")
X_test, Y_test = files_to_numpy(test_files)
print(f"Test: {X_test.shape[0]} muestras")


# ---------------------------------------------------------------------
# Custom objects y arquitecturas (identico a threshold_selection_and_ci.py)
# ---------------------------------------------------------------------
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_pred = tf.clip_by_value(y_pred, smooth, 1 - smooth)
    inter = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2. * inter + smooth) / (union + smooth)


def focal_tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), smooth, 1.0 - smooth)
    yt = tf.reshape(y_true, [-1]); yp = tf.reshape(y_pred, [-1])
    tp = tf.reduce_sum(yt * yp); fp = tf.reduce_sum((1 - yt) * yp); fn = tf.reduce_sum(yt * (1 - yp))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return tf.pow((1.0 - tversky), gamma)


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return tf.reduce_mean(0.5 * bce + 0.5 * dice_loss(y_true, y_pred))


class IoUScore(tf.keras.metrics.Metric):
    def __init__(self, name="IoU_score", threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.total_iou = self.add_weight(name="total_iou", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1]); y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        inter = tf.reduce_sum(y_true * y_pred, 1); union = tf.reduce_sum(y_true + y_pred, 1) - inter
        self.total_iou.assign_add(tf.reduce_mean(inter / (union + tf.keras.backend.epsilon())))
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
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1]); y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
        inter = tf.reduce_sum(y_true * y_pred, 1)
        self.total_dice.assign_add(tf.reduce_mean((2. * inter) / (tf.reduce_sum(y_true, 1) + tf.reduce_sum(y_pred, 1) + tf.keras.backend.epsilon())))
        self.count.assign_add(1.0)

    def result(self):
        return self.total_dice / tf.maximum(self.count, 1.0)


CUSTOM_OBJECTS = {
    "focal_tversky_loss": focal_tversky_loss,
    "combined_loss": combined_loss,
    "IoUScore": IoUScore,
    "DiceCoefficient": DiceCoefficient,
}

print("\nCargando checkpoints...")
unet_model = tf.keras.models.load_model("best_unet_model.keras", custom_objects=CUSTOM_OBJECTS)
mobilenet_model = tf.keras.models.load_model("best_mobilenetv2_model.keras", custom_objects=CUSTOM_OBJECTS)
convlstm_model = tf.keras.models.load_model("best_convlstm_model.keras", custom_objects=CUSTOM_OBJECTS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from xlstm import xLSTMBlockStack, xLSTMBlockStackConfig, mLSTMBlockConfig, mLSTMLayerConfig


class FrameEncoderTokens(nn.Module):
    def __init__(self, in_ch=12, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, d, 3, padding=1), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class MaskDecoder(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(d, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1))

    def forward(self, z):
        return torch.sigmoid(self.net(z))


class FireXLSTM_Tokens(nn.Module):
    def __init__(self, T, in_ch=12, d=64, d_model=128, n_blocks=4):
        super().__init__()
        self.T = T; self.d = d
        self.enc = FrameEncoderTokens(in_ch=in_ch, d=d)
        cfg = xLSTMBlockStackConfig(
            context_length=T, num_blocks=n_blocks, embedding_dim=d_model,
            mlstm_block=mLSTMBlockConfig(mlstm=mLSTMLayerConfig(conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4)),
            slstm_at=[])
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


def predict_keras_2d(model, X):
    return model.predict(X, verbose=0, batch_size=BATCH_SIZE)


def predict_keras_seq(model, X):
    X_seq = np.expand_dims(X, axis=1)
    return model.predict(X_seq, verbose=0, batch_size=BATCH_SIZE)


def predict_torch(model, X):
    X_seq = np.expand_dims(X, axis=1)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_seq), BATCH_SIZE):
            xb = torch.from_numpy(X_seq[i:i + BATCH_SIZE]).float().to(device)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def to_bhw1(pred):
    pred = np.asarray(pred)
    if pred.ndim == 5:
        pred = pred[:, 0]
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred.transpose(0, 2, 3, 1)
    return pred


print("\nCalculando probabilidades sobre test (una vez por modelo)...")
test_probs = {
    "unet": to_bhw1(predict_keras_2d(unet_model, X_test)),
    "mobilenetv2": to_bhw1(predict_keras_2d(mobilenet_model, X_test)),
    "convlstm": to_bhw1(predict_keras_seq(convlstm_model, X_test)),
    "xlstm": to_bhw1(predict_torch(xlstm_model, X_test)),
}


# ---------------------------------------------------------------------
# AP por muestra (macro), excluyendo muestras sin pixeles de incendio real
# ---------------------------------------------------------------------
def bootstrap_ci(values, n_boot=10_000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    alpha = 1 - ci
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return values.mean(), lo, hi


resultados = []
for name, probs in test_probs.items():
    ap_por_muestra = []
    excluidas = 0
    for i in range(len(Y_test)):
        y_true = Y_test[i].flatten()
        y_score = probs[i].flatten()
        if y_true.sum() == 0:
            excluidas += 1
            continue
        ap = average_precision_score(y_true, y_score)
        ap_por_muestra.append(ap)

    ap_arr = np.array(ap_por_muestra)
    mean_ap, lo, hi = bootstrap_ci(ap_arr)
    print(f"{name}: AP = {mean_ap:.4f} [{lo:.4f}, {hi:.4f}]  "
          f"(n={len(ap_arr)}, excluidas por sin-fuego={excluidas})")
    resultados.append({"modelo": name, "ap_mean": mean_ap, "ap_ci_low": lo, "ap_ci_high": hi,
                        "n_muestras": len(ap_arr), "excluidas_sin_fuego": excluidas})

df_ap = pd.DataFrame(resultados)
print("\n" + "=" * 60)
print("TABLA DE AVERAGE PRECISION (independiente de umbral)")
print("=" * 60)
print(df_ap.round(4).to_string(index=False))

os.makedirs(PREDICTIONS_PER_FIRE_DIR, exist_ok=True)
out_path = os.path.join(PREDICTIONS_PER_FIRE_DIR, "average_precision_results.csv")
df_ap.to_csv(out_path, index=False)
print(f"\nGuardado en: {out_path}")
