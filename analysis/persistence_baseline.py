# -*- coding: utf-8 -*-
"""
analysis/persistence_baseline.py

LINEA BASE DE PERSISTENCIA (sin modelo, sin entrenamiento): predice que el
fuego de manana es igual al de hoy, usando el canal viirs_PrevFireMask.
Variantes:
  persistencia      : mascara previa tal cual
  dilatacion_1      : mascara previa dilatada 1 pixel (vecindad 3x3)
  dilatacion_2      : mascara previa dilatada 2 pixeles (vecindad 5x5)

Sirve para mostrar cuanto aportan las arquitecturas respecto a copiar la
mascara previa (la ablacion mostro que ese canal domina).

Usa la MISMA funcion de evaluacion que los modelos (shared.evaluate), el
mismo conjunto de prueba y la misma definicion de IoU/F1 por muestra.

Uso (desde la raiz del proyecto, con el entorno activado):
    python -m analysis.persistence_baseline
Salida:
    outputs/predictions_per_fire/persistence_baseline_summary.csv
    outputs/predictions_per_fire/persistence_baseline_per_sample.csv
"""
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.ndimage import binary_dilation

from shared.config import FEATURE_RANGES, PREDICTIONS_PER_FIRE_DIR
from shared.partitions import get_train_val_test_files
from shared.evaluate import evaluate_model_unified

INPUT_FEATURES = ['elevation', 'wind_75', 'wind_avg', 'tmp_75', 'tmp_day', 'avg_sph',
                  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'viirs_PrevFireMask']
PREV_IDX = INPUT_FEATURES.index('viirs_PrevFireMask')
BATCH = 256
STRATA = [("1-25", 1, 25), ("26-100", 26, 100), ("101-500", 101, 500), (">500", 501, 10**9)]

# Valores de la Tabla 2 del articulo (3 semillas, umbral fijo) para comparar
REFERENCIA = {"U-Net": (0.2681, 0.3987), "MobileNetV2": (0.2522, 0.3803),
              "ConvLSTM": (0.2662, 0.3971), "xLSTM": (0.2703, 0.4053)}


def parse(example_proto):
    desc = {n: tf.io.FixedLenFeature([64, 64], tf.float32)
            for n in INPUT_FEATURES + ['viirs_FireMask']}
    ex = tf.io.parse_single_example(example_proto, desc)
    chans = []
    for n in INPUT_FEATURES:
        f = tf.expand_dims(ex[n], -1)
        if n == 'viirs_PrevFireMask':
            f = f * 5.0                      # igual que en los modelos
        else:
            lo, hi = FEATURE_RANGES[n]
            f = tf.clip_by_value((f - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        chans.append(f)
    x = tf.concat(chans, axis=-1)
    y = tf.expand_dims(tf.clip_by_value(ex['viirs_FireMask'], 0.0, 1.0), -1)
    return x, y


def batches(files):
    # MISMO orden que seeds_tuned_evaluation.py (list_files sin shuffle + interleave
    # determinista), para poder emparejar muestra a muestra con los modelos.
    ds = tf.data.Dataset.list_files(files, shuffle=False)
    ds = ds.interleave(lambda f: tf.data.TFRecordDataset(f),
                       num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.map(parse).batch(BATCH)
    for xb, yb in ds:
        yield xb.numpy(), yb.numpy()


def make_predict_fn(dilation):
    def predict_fn(inputs):
        prev = inputs[..., PREV_IDX] / 5.0 > 0.5             # mascara previa binaria
        out = np.zeros_like(prev, dtype="float32")
        for i in range(prev.shape[0]):
            m = prev[i]
            if dilation > 0:
                m = binary_dilation(m, structure=np.ones((3, 3), bool), iterations=dilation)
            out[i] = m
        return out[..., None]
    return predict_fn


def comparar_con_modelos(por_muestra, n_boot=10000):
    """Diferencia pareada (modelo - persistencia) en IoU por muestra, con IC95% bootstrap.
    Solo si el orden de muestras coincide (se verifica con target_pixels)."""
    rng = np.random.default_rng(42)
    print("\n=== Modelos vs persistencia (IoU por muestra, bootstrap pareado 95%) ===")
    for df in por_muestra:
        nombre = df["variante"].iloc[0]
        for modelo in ["unet", "mobilenetv2", "convlstm", "xlstm"]:
            ruta = os.path.join(PREDICTIONS_PER_FIRE_DIR, f"{modelo}_test_results_tuned_mean.csv")
            if not os.path.exists(ruta):
                continue
            m = pd.read_csv(ruta)
            if len(m) != len(df) or not np.array_equal(m["target_pixels"].to_numpy(), df["target_pixels"].to_numpy()):
                print(f"  {modelo}: el orden de muestras NO coincide; no se puede emparejar")
                continue
            d = m["iou"].to_numpy() - df["iou"].to_numpy()
            bm = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
            lo, hi = np.percentile(bm, [2.5, 97.5])
            print(f"  {modelo:12s} - {nombre:13s}: {d.mean():+.4f}  IC95% [{lo:+.4f}, {hi:+.4f}]")
            if nombre == "persistencia":
                for etiqueta, a, b in STRATA:
                    sel = (m["target_pixels"] >= a) & (m["target_pixels"] <= b)
                    print(f"      IoU {modelo} en {etiqueta}: {m.loc[sel, 'iou'].mean():.4f}")


def main():
    _, _, test_files = get_train_val_test_files()
    variantes = [("persistencia", 0), ("dilatacion_1", 1), ("dilatacion_2", 2)]
    resumen, por_muestra = [], []
    for nombre, k in variantes:
        df = evaluate_model_unified(
            predict_fn=make_predict_fn(k), eval_batches=batches(test_files),
            sample_ids=None, threshold=0.5, only_fire_pixels=False)
        df["variante"] = nombre
        por_muestra.append(df)
        fila = {"variante": nombre, "n": len(df), "iou": df["iou"].mean(), "f1": df["f1"].mean(),
                "precision": df["precision"].mean(), "recall": df["recall"].mean()}
        for etiqueta, lo, hi in STRATA:
            sub = df[(df["target_pixels"] >= lo) & (df["target_pixels"] <= hi)]
            fila[f"iou_{etiqueta}"] = sub["iou"].mean()
        resumen.append(fila)

    res = pd.DataFrame(resumen)
    print("\n=== LINEA BASE DE PERSISTENCIA (conjunto de prueba) ===")
    print(res.round(4).to_string(index=False))
    print("\n=== Referencia: modelos del articulo (IoU, F1; media de 3 semillas) ===")
    for m, (iou, f1) in REFERENCIA.items():
        print(f"  {m:12s} IoU={iou:.4f}  F1={f1:.4f}")

    comparar_con_modelos(por_muestra)

    os.makedirs(PREDICTIONS_PER_FIRE_DIR, exist_ok=True)
    res.to_csv(os.path.join(PREDICTIONS_PER_FIRE_DIR, "persistence_baseline_summary.csv"), index=False)
    pd.concat(por_muestra).to_csv(
        os.path.join(PREDICTIONS_PER_FIRE_DIR, "persistence_baseline_per_sample.csv"), index=False)
    print("\nGuardado en", PREDICTIONS_PER_FIRE_DIR)


if __name__ == "__main__":
    main()
