# -*- coding: utf-8 -*-
"""
analysis/architecture_table.py

Tabla de arquitecturas para el articulo (comentario R1-2): parametros,
FLOPs por muestra (paso hacia adelante) y tiempos de inferencia de cada
modelo, medidos sobre los checkpoints de la semilla 1001. Tambien guarda
el resumen de capas de cada modelo en outputs/architecture/*.txt.

FLOPs = 2 x MACs. En xLSTM, el contador de PyTorch solo cuenta convoluciones y
multiplicaciones de matrices, por lo que es una cota inferior (las compuertas
exponenciales y la normalizacion de mLSTM no se cuentan). Declararlo en el
articulo.

Uso:  python -m analysis.architecture_table
"""
import os, time, io, contextlib
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn

tf.config.optimizer.set_jit(False)
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

SEED_CKPT = 1001
OUT_DIR = os.path.join("outputs", "architecture")
os.makedirs(OUT_DIR, exist_ok=True)

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





def keras_flops(model, input_shape):
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    fn = tf.function(lambda x: model(x))
    cf = fn.get_concrete_function(tf.TensorSpec([1] + list(input_shape), tf.float32))
    _, graph_def = convert_variables_to_constants_v2_as_graph(cf)
    with tf.Graph().as_default() as g:
        tf.graph_util.import_graph_def(graph_def, name="")
        meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        info = tf.compat.v1.profiler.profile(graph=g, run_meta=meta, cmd="op", options=opts)
        return int(info.total_float_ops)


def time_fn(call, warmup=5, runs=30):
    for _ in range(warmup):
        call()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter(); call(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


rows = []
rng = np.random.default_rng(0)
X32 = rng.random((32, 64, 64, 12), dtype=np.float32)

for name in ["unet", "mobilenetv2", "convlstm"]:
    path = f"best_{name}_model_seed{SEED_CKPT}.keras"
    if not os.path.exists(path):
        print("FALTA", path); continue
    m = tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)
    seq = name == "convlstm"
    in_shape = (1, 64, 64, 12) if seq else (64, 64, 12)
    buf = io.StringIO()
    m.summary(print_fn=lambda s, **k: buf.write(s + "\n"))
    open(os.path.join(OUT_DIR, f"{name}_summary.txt"), "w", encoding="utf8").write(buf.getvalue())
    try:
        flops = keras_flops(m, in_shape)
    except Exception as e:
        print("FLOPs no disponibles para", name, e); flops = np.nan
    xb = np.expand_dims(X32, 1) if seq else X32
    t32 = time_fn(lambda: m.predict(xb, verbose=0, batch_size=32))
    rows.append({"modelo": name, "parametros": m.count_params(), "GFLOPs_por_muestra": flops / 1e9,
                 "ms_por_muestra_batch32": 1000 * t32 / 32})
    tf.keras.backend.clear_session()

path = f"best_xlstm_model_seed{SEED_CKPT}.pt"
if os.path.exists(path):
    m = FireXLSTM_Tokens(T=1, in_ch=12, d=64, d_model=128, n_blocks=4).to(device)
    m.load_state_dict(torch.load(path, map_location=device)); m.eval()
    open(os.path.join(OUT_DIR, "xlstm_summary.txt"), "w", encoding="utf8").write(str(m))
    x1 = torch.rand(1, 1, 64, 64, 12, device=device)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            m(x1)
        flops = fc.get_total_flops()
    except Exception as e:
        print("FLOPs no disponibles para xlstm", e); flops = np.nan
    xb = torch.rand(32, 1, 64, 64, 12, device=device)

    def call():
        with torch.no_grad():
            m(xb)
        if device.type == "cuda":
            torch.cuda.synchronize()
    t32 = time_fn(call)
    rows.append({"modelo": "xlstm", "parametros": sum(p.numel() for p in m.parameters()),
                 "GFLOPs_por_muestra": flops / 1e9, "ms_por_muestra_batch32": 1000 * t32 / 32})
else:
    print("FALTA", path)

df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
df.to_csv(os.path.join(OUT_DIR, "tabla_arquitecturas.csv"), index=False)
print("Guardado en", os.path.join(OUT_DIR, "tabla_arquitecturas.csv"))
print("(El tiempo de entrenamiento por epoca se toma de los logs de cada entrenamiento.)")
