# -*- coding: utf-8 -*-
"""
analysis/seeds_tuned_evaluation.py

Evaluacion CONSISTENTE de los 4 modelos x 3 semillas (1001, 2002, 3003):

  * Por cada (modelo, semilla): se cargan los checkpoints, se elige el umbral
    en VALIDACION (maximizando IoU medio) y SOLO ENTONCES se evalua test con
    ese umbral. Tambien se calcula Average Precision (AP) por muestra.
  * Por modelo: media +/- desviacion estandar entre semillas, y el IC
    bootstrap del 95% (limite inferior y superior) para IoU, F1, Accuracy,
    Precision, Recall y AP, sobre la serie por muestra promediada entre
    semillas.

POR QUE EXISTE: los CSV `*_test_results_seed*.csv` que escriben los scripts
train_*.py se evaluan con THRESHOLD=0.5 de config.py, por lo que no coinciden
con los umbrales que reporta la Tabla 1 del articulo. Este script reemplaza
esa evaluacion.

Requiere checkpoints:  best_unet_model_seed{S}.keras, best_mobilenetv2_model_seed{S}.keras,
best_convlstm_model_seed{S}.keras, best_xlstm_model_seed{S}.pt  (S en 1001, 2002, 3003)
Uso:  python -m analysis.seeds_tuned_evaluation
"""


import os
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn

from shared.config import BATCH_SIZE, PREDICTIONS_PER_FIRE_DIR, FEATURE_RANGES
from shared.partitions import get_train_val_test_files

tf.config.optimizer.set_jit(False)
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

INPUT_FEATURES = ['elevation', 'wind_75', 'wind_avg', 'tmp_75', 'tmp_day', 'avg_sph',
                  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'viirs_PrevFireMask']
OUTPUT_FEATURES = ['viirs_FireMask']

train_files, validation_files, test_files = get_train_val_test_files()


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


print("Cargando validation set...")
X_val, Y_val = files_to_numpy(validation_files)
print("Cargando test set...")
X_test, Y_test = files_to_numpy(test_files)
print(f"Validation: {X_val.shape[0]} muestras | Test: {X_test.shape[0]} muestras")


# ---------------------------------------------------------------------
# Custom objects necesarios para cargar cada checkpoint de Keras
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



# ---------------------------------------------------------------------
# Evaluacion por (modelo, semilla)
# ---------------------------------------------------------------------
import gc
from sklearn.metrics import average_precision_score

SEEDS = [1001, 2002, 3003]
MODELOS = ["unet", "mobilenetv2", "convlstm", "xlstm"]
METRICAS = ["iou", "f1", "accuracy", "precision", "recall", "ap"]
THRESH_GRID = np.arange(0.10, 0.91, 0.05)

# Modo "fixed": umbral FIJO por modelo (elegido en validacion en la corrida original,
# los mismos que describe el Methods y la Tabla 1 del articulo). Uso:
#   python -m analysis.seeds_tuned_evaluation          (umbral fijo, por defecto)
#   python -m analysis.seeds_tuned_evaluation perseed  (umbral elegido por semilla)
import sys
FIXED_THRESHOLDS = {"unet": 0.45, "mobilenetv2": 0.25, "convlstm": 0.30, "xlstm": 0.30}
MODO_FIJO = not (len(sys.argv) > 1 and sys.argv[1] == "perseed")   # por defecto: umbral FIJO
print("=" * 60); print("MODO:", "umbral FIJO por modelo " + str(FIXED_THRESHOLDS) if MODO_FIJO else "umbral elegido por semilla"); print("=" * 60)


def to_bhw1(pred):
    pred = np.asarray(pred)
    if pred.ndim == 5:
        pred = pred[:, 0]
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred.transpose(0, 2, 3, 1)
    return pred


def predict_keras_2d(model, X):
    return model.predict(X, verbose=0, batch_size=BATCH_SIZE)


def predict_keras_seq(model, X):
    return model.predict(np.expand_dims(X, axis=1), verbose=0, batch_size=BATCH_SIZE)


def predict_torch(model, X):
    X_seq = np.expand_dims(X, axis=1)
    out = []
    with torch.no_grad():
        for i in range(0, len(X_seq), 8):
            xb = torch.from_numpy(X_seq[i:i + 8]).float().to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def iou_at_threshold(p, Y, thr):
    pb = (p > thr).astype("int32"); yb = Y.astype("int32")
    tp = np.sum((yb == 1) & (pb == 1), axis=(1, 2, 3))
    fp = np.sum((yb == 0) & (pb == 1), axis=(1, 2, 3))
    fn = np.sum((yb == 1) & (pb == 0), axis=(1, 2, 3))
    return (tp / (tp + fp + fn + 1e-7)).mean()


def per_sample_metrics(p, Y, thr):
    pb = (p > thr).astype("int32"); yb = Y.astype("int32")
    tp = np.sum((yb == 1) & (pb == 1), axis=(1, 2, 3))
    fp = np.sum((yb == 0) & (pb == 1), axis=(1, 2, 3))
    fn = np.sum((yb == 1) & (pb == 0), axis=(1, 2, 3))
    tn = np.sum((yb == 0) & (pb == 0), axis=(1, 2, 3))
    eps = 1e-7
    total = tp + fp + tn + fn
    prec = tp / (tp + fp + eps); rec = tp / (tp + fn + eps)
    return pd.DataFrame({
        "sample_id": np.arange(len(tp)), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "target_pixels": tp + fn, "n_pixels": total,
        "accuracy": (tp + tn) / (total + eps), "precision": prec, "recall": rec,
        "iou": tp / (tp + fp + fn + eps), "f1": 2 * prec * rec / (prec + rec + eps),
    })


def ap_per_sample(p, Y):
    """AP por muestra; NaN si la muestra no tiene pixeles de incendio reales."""
    out = np.full(len(Y), np.nan)
    for i in range(len(Y)):
        yt = Y[i].flatten()
        if yt.sum() > 0:
            out[i] = average_precision_score(yt, p[i].flatten())
    return out


def load_and_predict(model_name, seed):
    """Devuelve (probs_val, probs_test) o None si falta el checkpoint."""
    if model_name == "xlstm":
        path = f"best_xlstm_model_seed{seed}.pt"
        if not os.path.exists(path):
            return None
        m = FireXLSTM_Tokens(T=1, in_ch=12, d=64, d_model=128, n_blocks=4).to(device)
        m.load_state_dict(torch.load(path, map_location=device)); m.eval()
        pv, pt = predict_torch(m, X_val), predict_torch(m, X_test)
        del m; torch.cuda.empty_cache()
    else:
        path = f"best_{model_name}_model_seed{seed}.keras"
        if not os.path.exists(path):
            return None
        m = tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)
        fn = predict_keras_seq if model_name == "convlstm" else predict_keras_2d
        pv, pt = fn(m, X_val), fn(m, X_test)
        del m; tf.keras.backend.clear_session()
    gc.collect()
    return to_bhw1(pv), to_bhw1(pt)


os.makedirs(PREDICTIONS_PER_FIRE_DIR, exist_ok=True)
per_seed = {m: {} for m in MODELOS}      # per_seed[modelo][semilla] = DataFrame por muestra
filas_umbral = []
faltantes = []

for modelo in MODELOS:
    for seed in SEEDS:
        res = load_and_predict(modelo, seed)
        if res is None:
            faltantes.append((modelo, seed)); continue
        pv, pt = res
        ious = [iou_at_threshold(pv, Y_val, t) for t in THRESH_GRID]
        thr = round(float(THRESH_GRID[int(np.argmax(ious))]), 2)
        if MODO_FIJO:
            thr = FIXED_THRESHOLDS[modelo]
        df = per_sample_metrics(pt, Y_test, thr)
        df["ap"] = ap_per_sample(pt, Y_test)
        df.to_csv(os.path.join(PREDICTIONS_PER_FIRE_DIR, f"{modelo}_test_results_tuned_seed{seed}.csv"), index=False)
        per_seed[modelo][seed] = df
        filas_umbral.append({"modelo": modelo, "semilla": seed, "umbral": thr,
                             "iou_val_en_umbral": iou_at_threshold(pv, Y_val, thr), "iou_val_en_0.5": iou_at_threshold(pv, Y_val, 0.5)})
        print(f"{modelo} semilla {seed}: umbral={thr:.2f}  IoU test={df['iou'].mean():.4f}  AP={np.nanmean(df['ap']):.4f}")
        del pv, pt; gc.collect()

if faltantes:
    print("\n*** FALTAN CHECKPOINTS (se omiten): ", faltantes)

df_umbrales = pd.DataFrame(filas_umbral)
df_umbrales.to_csv(os.path.join(PREDICTIONS_PER_FIRE_DIR, "umbrales_por_modelo_y_semilla.csv"), index=False)
print("\nUMBRALES ELEGIDOS EN VALIDACION (por modelo y semilla)")
print(df_umbrales.round(4).to_string(index=False))

# ---------------------------------------------------------------------
# Agregacion: media +/- SD entre semillas e IC bootstrap (ambos limites)
# ---------------------------------------------------------------------
def bootstrap_ci_mean(values, n_boot=10_000, ci=0.95, seed=42):
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    a = 1 - ci
    lo, hi = np.percentile(means, [100 * a / 2, 100 * (1 - a / 2)])
    return lo, hi


def serie_promedio(modelo, metrica):
    return np.mean([per_seed[modelo][s][metrica].to_numpy() for s in per_seed[modelo]], axis=0)


filas = []
for modelo in MODELOS:
    if not per_seed[modelo]:
        continue
    seeds_ok = list(per_seed[modelo])
    umbrales = df_umbrales[df_umbrales.modelo == modelo]["umbral"].tolist()
    fila = {"modelo": modelo, "n_semillas": len(seeds_ok), "umbrales_por_semilla": str(umbrales)}
    for met in METRICAS:
        por_semilla = [np.nanmean(per_seed[modelo][s][met]) for s in seeds_ok]
        fila[f"{met}_mean"] = float(np.mean(por_semilla))
        fila[f"{met}_sd"] = float(np.std(por_semilla, ddof=1)) if len(por_semilla) > 1 else np.nan
        lo, hi = bootstrap_ci_mean(serie_promedio(modelo, met))
        fila[f"{met}_ci_low"], fila[f"{met}_ci_high"] = lo, hi
    filas.append(fila)

df_tabla = pd.DataFrame(filas)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print("\n" + "=" * 70)
print("TABLA PRINCIPAL: media, SD entre semillas e IC bootstrap 95% (inf, sup)")
print("=" * 70)
print(df_tabla.drop(columns=["umbrales_por_semilla"]).round(4).to_string(index=False))
df_tabla.to_csv(os.path.join(PREDICTIONS_PER_FIRE_DIR, "tabla_principal_3_semillas_tuned.csv"), index=False)

# Series por muestra promediadas entre semillas (insumo de la estratificacion)
for modelo in MODELOS:
    if not per_seed[modelo]:
        continue
    base = per_seed[modelo][list(per_seed[modelo])[0]][["sample_id", "target_pixels"]].copy()
    for met in METRICAS:
        base[met] = serie_promedio(modelo, met)
    base.to_csv(os.path.join(PREDICTIONS_PER_FIRE_DIR, f"{modelo}_test_results_tuned_mean.csv"), index=False)

# Diferencias pareadas (IoU y AP) entre modelos
PARES = [("xlstm", "unet"), ("xlstm", "convlstm"), ("convlstm", "unet"), ("mobilenetv2", "unet")]
filas_p = []
for a_, b_ in PARES:
    if not per_seed[a_] or not per_seed[b_]:
        continue
    for met in ["iou", "ap"]:
        d = serie_promedio(a_, met) - serie_promedio(b_, met)
        d = d[~np.isnan(d)]
        lo, hi = bootstrap_ci_mean(d)
        filas_p.append({"comparacion": f"{a_} vs {b_}", "metrica": met, "mean_diff": d.mean(),
                        "ci_low": lo, "ci_high": hi, "significativo": "SI" if (lo > 0 or hi < 0) else "no"})
df_p = pd.DataFrame(filas_p)
print("\nDIFERENCIAS PAREADAS (serie promediada entre semillas)")
print(df_p.round(4).to_string(index=False))
df_p.to_csv(os.path.join(PREDICTIONS_PER_FIRE_DIR, "diferencias_pareadas_tuned.csv"), index=False)
print("\nListo. Archivos en", PREDICTIONS_PER_FIRE_DIR)
