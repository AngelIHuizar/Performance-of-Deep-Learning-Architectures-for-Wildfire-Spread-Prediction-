# -*- coding: utf-8 -*-
"""
analysis/temporal_demo.py

PRUEBA DE DEMOSTRACION (no forma parte del articulo): que pasa con ConvLSTM
cuando se le da T > 1 en NDWS, un dataset sin continuidad temporal verificable.

Tres modos, misma arquitectura, misma perdida, misma semilla:
  single     T=1                       (referencia; equivale al modelo del articulo)
  repeat     T=3, el MISMO dia repetido 3 veces (no agrega informacion nueva)
  neighbors  T=3, ventana de 3 muestras CONSECUTIVAS del archivo (lo que hacia el
             script original); la etiqueta es la de la ultima muestra. Los vecinos
             pueden ser incendios distintos: NDWS no trae fecha ni ubicacion.

Uso (desde la raiz del proyecto, con el entorno activado):
  python -m analysis.temporal_demo single 1 1001
  python -m analysis.temporal_demo repeat 3 1001
  python -m analysis.temporal_demo neighbors 3 1001
  python -m analysis.temporal_demo summary

SIN REENTRENAR (usa el ConvLSTM ya entrenado, best_convlstm_model_seed1001.keras, que
vio solo T=1; se evalua con T=3 reconstruyendo el modelo con tiempo variable):
  python -m analysis.temporal_demo eval single 1 1001
  python -m analysis.temporal_demo eval repeat 3 1001
  python -m analysis.temporal_demo eval neighbors 3 1001
  python -m analysis.temporal_demo summary
OJO: aqui el modelo NUNCA aprendio a usar historia (se entreno con T=1), asi que esta
prueba solo muestra que pasa si se le da historia; no si entrenar con T>1 ayudaria.
Cada corrida guarda outputs/temporal_demo/<modo>_T<T>_seed<S>.npz con IoU y AP
por muestra de TEST. 'summary' compara los modos con bootstrap pareado.

IoU se calcula con el umbral 0.30 (el de ConvLSTM, elegido en validacion en
el articulo) para todos los modos; AP no depende del umbral.
"""
import os
import sys
import glob
import gc

import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy

from shared.config import BATCH_SIZE, FEATURE_RANGES
from shared.partitions import get_train_val_test_files

OUT_DIR = os.path.join("outputs", "temporal_demo")
THRESHOLD = 0.30
INPUT_FEATURES = ['elevation', 'wind_75', 'wind_avg', 'tmp_75', 'tmp_day', 'avg_sph',
                  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'viirs_PrevFireMask']


# ----------------------------------------------------------------- datos
def normalize_feature(feature, name):
    if name == 'viirs_PrevFireMask':
        return feature * 5.0
    lo, hi = FEATURE_RANGES[name]
    return tf.clip_by_value((feature - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def parse(example_proto):
    desc = {n: tf.io.FixedLenFeature([64, 64], tf.float32)
            for n in INPUT_FEATURES + ['viirs_FireMask']}
    ex = tf.io.parse_single_example(example_proto, desc)
    x = tf.concat([normalize_feature(tf.expand_dims(ex[n], -1), n) for n in INPUT_FEATURES], axis=-1)
    y = tf.expand_dims(tf.clip_by_value(ex['viirs_FireMask'], 0.0, 1.0), -1)
    return x, y


def make_dataset(files, mode, T, shuffle, seed):
    # Flujo SECUENCIAL (sin interleave) para que 'vecino' signifique vecino del archivo
    ds = tf.data.TFRecordDataset(sorted(files)).map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    if mode == "single":
        ds = ds.map(lambda x, y: (x[None], y))
    elif mode == "repeat":
        ds = ds.map(lambda x, y: (tf.repeat(x[None], T, axis=0), y))
    elif mode == "neighbors":
        first = ds.take(1).repeat(T - 1)           # relleno para no perder las primeras muestras
        ds = first.concatenate(ds)
        ds = ds.window(T, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda wx, wy: tf.data.Dataset.zip((wx.batch(T), wy.batch(T))))
        ds = ds.map(lambda xs, ys: (xs, ys[-1]))
    else:
        raise ValueError(mode)
    if shuffle:
        ds = ds.shuffle(1000, seed=seed, reshuffle_each_iteration=True)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------- modelo
def build_model(T, dropout=0.2):
    inp = tf.keras.layers.Input(shape=(T, 64, 64, 12))
    x = tf.keras.layers.ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.ConvLSTM2D(32, (3, 3), padding="same", return_sequences=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    out = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(x)
    return tf.keras.Model(inp, out)


def dice_loss(y_true, y_pred, smooth=1e-6):
    y_pred = tf.clip_by_value(y_pred, smooth, 1 - smooth)
    inter = tf.reduce_sum(y_true * y_pred)
    return 1 - (2. * inter + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def combined_loss(y_true, y_pred):
    return tf.reduce_mean(0.5 * BinaryCrossentropy()(y_true, y_pred) + 0.5 * dice_loss(y_true, y_pred))


# ----------------------------------------------------------------- corrida
def evaluate(model, mode, T, seed, test_f, tag):
    test_ds = make_dataset(test_f, mode, T, shuffle=False, seed=seed)
    ious, aps = [], []
    for xb, yb in test_ds:
        pb = model.predict(xb, verbose=0)[..., 0]
        yb = yb.numpy()[..., 0]
        for p, y in zip(pb, yb):
            pred = p > THRESHOLD
            inter = np.logical_and(pred, y > 0.5).sum()
            union = np.logical_or(pred, y > 0.5).sum()
            ious.append(inter / (union + 1e-7))
            aps.append(average_precision_score(y.ravel() > 0.5, p.ravel()) if y.sum() > 0 else np.nan)
    ious, aps = np.array(ious), np.array(aps)
    print(f"[{tag}] Muestras: {len(ious)}  IoU(0.30)={ious.mean():.4f}  AP={np.nanmean(aps):.4f}")
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, f"{tag}_T{T}_seed{seed}.npz"), iou=ious, ap=aps)


def run_eval(mode, T, seed):
    """SIN reentrenar: carga el ConvLSTM ya entrenado con T=1 y lo evalua con T pasos."""
    tf.config.optimizer.set_jit(False)
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    _, _, test_f = get_train_val_test_files()
    ckpt = f"best_convlstm_model_seed{seed}.keras"
    print(f"*** EVAL sin reentrenar: {ckpt}  modo={mode} T={T} ***")
    base = tf.keras.models.load_model(ckpt, compile=False)
    flex = build_model(None)                    # mismo modelo, pero con tiempo variable
    flex.set_weights(base.get_weights())        # los pesos de ConvLSTM2D no dependen de T
    evaluate(flex, mode, T, seed, test_f, tag="eval" + mode)
    gc.collect()
    tf.keras.backend.clear_session()


def run(mode, T, seed):
    tf.config.optimizer.set_jit(False)
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    train_f, val_f, test_f = get_train_val_test_files()
    print(f"*** modo={mode} T={T} semilla={seed} ***")

    train_ds = make_dataset(train_f, mode, T, shuffle=True, seed=seed)
    val_ds = make_dataset(val_f, mode, T, shuffle=False, seed=seed)
    model = build_model(T)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.0), loss=combined_loss)
    model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=2, min_lr=1e-6, verbose=1)])

    evaluate(model, mode, T, seed, test_f, tag=mode)
    gc.collect()
    tf.keras.backend.clear_session()


def summary():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*.npz")))
    data = {os.path.basename(f)[:-4]: np.load(f) for f in files}
    base_key = next((k for k in data if k.startswith("evalsingle")), None) or next((k for k in data if k.startswith("single")), None)
    print(f"{'corrida':28s} {'n':>6s} {'IoU':>8s} {'AP':>8s}")
    for k, v in data.items():
        print(f"{k:28s} {len(v['iou']):6d} {v['iou'].mean():8.4f} {np.nanmean(v['ap']):8.4f}")
    if base_key is None:
        return
    rng = np.random.default_rng(42)
    print("\nDiferencias contra", base_key, "(bootstrap pareado 95%, 10,000 remuestreos)")
    for k, v in data.items():
        if k == base_key or len(v['iou']) != len(data[base_key]['iou']):
            continue
        for m in ("iou", "ap"):
            d = v[m] - data[base_key][m]
            d = d[~np.isnan(d)]
            idx = rng.integers(0, len(d), (10000, len(d)))
            bm = d[idx].mean(axis=1)
            lo, hi = np.percentile(bm, [2.5, 97.5])
            print(f"  {k:26s} d{m.upper():3s} = {d.mean():+.4f}  IC95% [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summary()
    elif sys.argv[1] == "eval":
        run_eval(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
