# -*- coding: utf-8 -*-
"""
models/train_xlstm.py

VERSION T=1 -- mismo motivo que train_convlstm.py: NDWS no tiene fecha ni
ubicacion por muestra, asi que el ventaneo T=6 original (via
interleave() + window()) fabricaba secuencias temporales que mezclaban
incendios no relacionados. Ver bitacora, Semana 1.

Aqui T no es solo un envoltorio externo -- es el context_length real de
los bloques mLSTM (FireXLSTM_Tokens). Con T=1, xLSTM sigue siendo la
arquitectura real (misma clase, mismos bloques mLSTM), pero procesa cada
muestra de forma independiente, sin memoria temporal fabricada -- mismo
criterio que se aplico a ConvLSTM, para que la comparacion entre los 4
modelos sea consistente.

Otros cambios respecto al script original:
1. Particion train/val/test desde shared.partitions (antes: heuristica
   propia por substring "train"/"val"/"test", que coincidia por
   casualidad con "eval" pero ya no hace falta mantenerla separada).
2. Normalizacion con los mismos valores de shared.config.FEATURE_RANGES
   que usan los otros 3 modelos (antes: valores distintos, ver bitacora).
3. Se elimina el paso de exportar a .npy y volver a cargar -- con T=1 se
   construyen los tensores directamente en memoria, mas simple y menos
   propenso a errores de sincronizacion entre pasos.
4. Evaluacion final unificada via shared.evaluate (antes: funcion propia
   compute_per_image_metrics_pytorch, no auditada -- ahora usa el mismo
   criterio que los otros 3 modelos: mismo threshold, mismo
   only_fire_pixels, mismo formato de salida).
5. Se quita la dependencia de Google Colab (drive.mount) -- no aplica
   en PC local.
"""

import os

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from xlstm import (
    xLSTMBlockStack, xLSTMBlockStackConfig,
    mLSTMBlockConfig, mLSTMLayerConfig
)

from shared.config import SEED, THRESHOLD, ONLY_FIRE_PIXELS, PREDICTIONS_PER_FIRE_DIR, FEATURE_RANGES
from shared.partitions import get_train_val_test_files
from shared.evaluate import evaluate_model_unified, make_pytorch_predict_fn

# ---------------------------------------------------------------------
# NUEVO: override de semilla via linea de comandos, para el analisis de
# estabilidad con 3 semillas. Uso: python -m models.train_xlstm 1001
# ---------------------------------------------------------------------
import sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else SEED
print(f"*** Semilla usada en esta corrida: {SEED} ***")
tf.random.set_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

# ---------------------------------------------------------------------
# 1. Particion de datos -- misma fuente que los demas modelos
# ---------------------------------------------------------------------
train_files, validation_files, test_files = get_train_val_test_files()

# ---------------------------------------------------------------------
# 2. Parsing de TFRecords (via TensorFlow, solo para leer/normalizar --
#    el modelo en si es PyTorch) -- mismos INPUT_FEATURES y misma
#    normalizacion por nombre que los otros 3 modelos
# ---------------------------------------------------------------------
INPUT_FEATURES = ['elevation', 'wind_75', 'wind_avg', 'tmp_75', 'tmp_day', 'avg_sph',
                  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'viirs_PrevFireMask']
OUTPUT_FEATURES = ['viirs_FireMask']


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
    x = tf.concat(inputs_list, axis=-1)  # (64,64,12)

    y = tf.expand_dims(tf.clip_by_value(ex["viirs_FireMask"], 0.0, 1.0), axis=-1)  # (64,64,1)
    return x, y


def files_to_numpy(files):
    """Lee todos los ejemplos de una lista de archivos y los devuelve como
    arrays numpy (N,64,64,12) y (N,64,64,1). Reemplaza el paso de
    exportar/reimportar .npy del script original -- con T=1 no hace
    falta ventanear, se puede construir todo directo en memoria."""
    ds = tf.data.Dataset.list_files(files, shuffle=False)
    ds = ds.interleave(lambda f: tf.data.TFRecordDataset(f),
                        num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)

    X_list, Y_list = [], []
    for x, y in ds.as_numpy_iterator():
        X_list.append(x)
        Y_list.append(y)
    X = np.stack(X_list, axis=0)  # (N,64,64,12)
    Y = np.stack(Y_list, axis=0)  # (N,64,64,1)
    return X, Y


print("Cargando datos de entrenamiento...")
X_train, Y_train = files_to_numpy(train_files)
print("Cargando datos de validacion...")
X_val, Y_val = files_to_numpy(validation_files)
print("Cargando datos de test...")
X_test, Y_test = files_to_numpy(test_files)

T = 1  # sin ventaneo -- cada ejemplo es su propia "secuencia" de 1 paso


def to_torch_seq(X, Y):
    # Agrega la dimension temporal T=1: (N,64,64,12) -> (N,1,64,64,12)
    X_seq = np.expand_dims(X, axis=1)
    X_t = torch.from_numpy(X_seq).float()
    # (N,64,64,1) -> (N,1,64,64) para que calce con la salida del modelo
    Y_t = torch.from_numpy(Y).float().permute(0, 3, 1, 2)
    return X_t, Y_t


X_train_t, Y_train_t = to_torch_seq(X_train, Y_train)
X_val_t, Y_val_t = to_torch_seq(X_val, Y_val)
X_test_t, Y_test_t = to_torch_seq(X_test, Y_test)

BATCH_SIZE = 8

train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=BATCH_SIZE,
                           shuffle=True, drop_last=True)
val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=BATCH_SIZE,
                         shuffle=False, drop_last=False)
test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t), batch_size=BATCH_SIZE,
                          shuffle=False, drop_last=False)

xb, yb = next(iter(train_loader))
print("Torch batch X:", xb.shape)  # (B,1,64,64,12)
print("Torch batch Y:", yb.shape)  # (B,1,64,64)


# ---------------------------------------------------------------------
# 3. Arquitectura xLSTM -- identica al original, solo T=1
# ---------------------------------------------------------------------
class FrameEncoderTokens(nn.Module):
    """Conv encoder por frame: (B,C,64,64) -> (B,d,16,16)"""
    def __init__(self, in_ch=12, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, d, 3, padding=1), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MaskDecoder(nn.Module):
    """(B,d,16,16) -> (B,1,64,64)"""
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
            context_length=T,
            num_blocks=n_blocks,
            embedding_dim=d_model,
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=4
                )
            ),
            slstm_at=[]
        )
        self.in_proj = nn.Linear(d, d_model)
        self.xlstm = xLSTMBlockStack(cfg)
        self.out_proj = nn.Linear(d_model, d)

        self.dec = MaskDecoder(d=d)

    def forward(self, x):
        # x: (B,T,64,64,12)
        B, T, H, W, C = x.shape
        assert T == self.T, f"T mismatch: got {T}, expected {self.T}"

        x = x.permute(0, 1, 4, 2, 3).contiguous()  # (B,T,C,H,W)

        feats = []
        for t in range(T):
            feats.append(self.enc(x[:, t]))
        feats = torch.stack(feats, dim=1)  # (B,T,d,16,16)

        feats = feats.permute(0, 1, 3, 4, 2).contiguous()
        S = 16 * 16
        feats = feats.view(B, T, S, self.d)

        seq = feats.permute(0, 2, 1, 3).contiguous().view(B * S, T, self.d)

        seq = self.in_proj(seq)
        y = self.xlstm(seq)
        h = y[:, -1, :]
        tok = self.out_proj(h)

        grid = tok.view(B, S, self.d).view(B, 16, 16, self.d).permute(0, 3, 1, 2).contiguous()

        return self.dec(grid)  # (B,1,64,64)


model = FireXLSTM_Tokens(T=T, in_ch=12, d=64, d_model=128, n_blocks=4).to(device)
print("Parametros:", sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------------
# 4. Loss -- identica al original (BCE + Dice)
# ---------------------------------------------------------------------
def dice_loss(y_pred, y_true, eps=1e-6):
    y_pred = y_pred.clamp(0, 1)
    num = 2 * (y_pred * y_true).sum(dim=(2, 3)) + eps
    den = (y_pred + y_true).sum(dim=(2, 3)) + eps
    return 1 - (num / den)


bce = nn.BCELoss()


def combined_loss(y_pred, y_true):
    return 0.5 * bce(y_pred, y_true) + 0.5 * dice_loss(y_pred, y_true).mean()


lr = 1e-4
epochs = 50  # bajado de 100 -- sin windowing, cada epoca ve menos "muestras infladas"
optimizer = optim.Adam(model.parameters(), lr=lr)


def evaluate_loss(loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = combined_loss(pred, yb)
            losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


# ---------------------------------------------------------------------
# 5. Entrenamiento, con early stopping simple (el original no tenia)
# ---------------------------------------------------------------------
best_val_loss = float("inf")
patience = 5
patience_counter = 0
best_state = None

train_losses_per_epoch = []
val_losses_per_epoch = []

for epoch in range(1, epochs + 1):
    model.train()
    train_losses = []
    for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
        xb, yb = xb.to(device), yb.to(device)

        pred = model(xb)
        loss = combined_loss(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    tr = float(np.mean(train_losses))
    va = evaluate_loss(val_loader)
    train_losses_per_epoch.append(tr)
    val_losses_per_epoch.append(va)
    print(f"Epoch {epoch}: train_loss={tr:.4f}  val_loss={va:.4f}")

    if va < best_val_loss:
        best_val_loss = va
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping en epoca {epoch}")
            break

if best_state is not None:
    model.load_state_dict(best_state)
    torch.save(best_state, f"best_xlstm_model_seed{SEED}.pt")
    print("Mejores pesos restaurados y guardados en best_xlstm_model.pt")

# ---------------------------------------------------------------------
# 6. EVALUACION FINAL -- unificada via shared.evaluate, igual criterio
#    que los otros 3 modelos
# ---------------------------------------------------------------------
print("Evaluando sobre el TEST set real (T=1, sin ventaneo sintetico)...")

predict_fn = make_pytorch_predict_fn(model, device)


def torch_loader_to_numpy_batches(loader):
    """Adapta el DataLoader de PyTorch (B,1,64,64,12)/(B,1,64,64) al
    formato numpy (inputs, labels) que espera evaluate_model_unified,
    con labels en formato (B,64,64,1) para que calce con la logica de
    conteo por pixel."""
    for xb, yb in loader:
        x_np = xb.numpy()  # (B,1,64,64,12)
        y_np = yb.numpy()  # (B,1,64,64)
        y_np = np.transpose(y_np, (0, 2, 3, 1))  # (B,64,64,1)
        yield x_np, y_np


results_df = evaluate_model_unified(
    predict_fn=predict_fn,
    eval_batches=torch_loader_to_numpy_batches(test_loader),
    sample_ids=None,
    threshold=THRESHOLD,
    only_fire_pixels=ONLY_FIRE_PIXELS,
)

print(f"Muestras evaluadas: {len(results_df)}")
print(f"IoU promedio (macro por muestra): {results_df['iou'].mean():.4f}")
print(f"F1 promedio: {results_df['f1'].mean():.4f}")
print(f"Accuracy promedio: {results_df['accuracy'].mean():.4f}")
print(f"Precision promedio: {results_df['precision'].mean():.4f}")
print(f"Recall promedio: {results_df['recall'].mean():.4f}")

os.makedirs(PREDICTIONS_PER_FIRE_DIR, exist_ok=True)
output_csv = os.path.join(PREDICTIONS_PER_FIRE_DIR, f"xlstm_test_results_seed{SEED}.csv")
results_df.to_csv(output_csv, index=False)
print(f"Resultados por muestra guardados en: {output_csv}")
