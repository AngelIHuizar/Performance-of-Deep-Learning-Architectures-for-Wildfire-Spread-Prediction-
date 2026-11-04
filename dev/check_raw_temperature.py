"""
analysis/check_raw_temperature.py

Diagnostico rapido: revisa los valores CRUDOS (sin normalizar) de
tmp_75 y tmp_day sobre unas cuantas muestras del train set, para
confirmar si el rango asumido en FEATURE_RANGES (280, 320) -- en Kelvin --
realmente corresponde a como vienen los datos, o si estan en otra unidad
(Celsius, o alguna escala distinta).
"""

import tensorflow as tf

from shared.partitions import get_train_val_test_files

train_files, _, _ = get_train_val_test_files()

FEATURES_A_REVISAR = ["tmp_75", "tmp_day", "erc", "NDVI"]

feature_description = {fn: tf.io.FixedLenFeature([64, 64], tf.float32) for fn in FEATURES_A_REVISAR}

ds = tf.data.Dataset.list_files(train_files, shuffle=False)
ds = ds.interleave(lambda f: tf.data.TFRecordDataset(f), num_parallel_calls=tf.data.AUTOTUNE)
ds = ds.map(lambda ex: tf.io.parse_single_example(ex, feature_description))

mins = {fn: float("inf") for fn in FEATURES_A_REVISAR}
maxs = {fn: float("-inf") for fn in FEATURES_A_REVISAR}
sums = {fn: 0.0 for fn in FEATURES_A_REVISAR}
counts = {fn: 0 for fn in FEATURES_A_REVISAR}

n_muestras = 0
for parsed in ds.as_numpy_iterator():
    for fn in FEATURES_A_REVISAR:
        v = parsed[fn]
        mins[fn] = min(mins[fn], v.min())
        maxs[fn] = max(maxs[fn], v.max())
        sums[fn] += v.sum()
        counts[fn] += v.size
    n_muestras += 1
    if n_muestras >= 200:  # con 200 muestras es suficiente para el diagnostico
        break

print(f"Diagnostico sobre {n_muestras} muestras del train set:\n")
for fn in FEATURES_A_REVISAR:
    print(f"{fn}: min={mins[fn]:.4f}  max={maxs[fn]:.4f}  mean={sums[fn] / counts[fn]:.4f}")
