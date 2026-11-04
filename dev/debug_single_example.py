"""
analysis/debug_single_example.py

Aisla el bug de las medias identicas probando con UNA SOLA muestra:
si ya ahi tmp_75/tmp_day/NDVI/erc salen identicos despues de normalizar
y concatenar, el bug esta en normalize_feature/_parse_function. Si salen
distintos aqui pero identicos al promediar sobre muchas muestras, el bug
esta en como se agrega/apila (stack) sobre el dataset completo.
"""

import numpy as np
import tensorflow as tf

from shared.config import FEATURE_RANGES
from shared.partitions import get_train_val_test_files

train_files, _, _ = get_train_val_test_files()

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
        normed = normalize_feature(feat, fn)
        inputs_list.append(normed)
        # DEBUG: imprimir la media de CADA canal individualmente, antes del concat
        print(f"  [antes de concat] {fn}: media={tf.reduce_mean(normed).numpy():.4f}")
    x = tf.concat(inputs_list, axis=-1)
    y = tf.expand_dims(tf.clip_by_value(ex["viirs_FireMask"], 0.0, 1.0), axis=-1)
    return x, y


print("=" * 60)
print("UNA SOLA MUESTRA -- valores antes de tf.concat")
print("=" * 60)

raw_ds = tf.data.TFRecordDataset([train_files[0]])
one_example = next(iter(raw_ds))
x, y = _parse_function(one_example)

print("\n" + "=" * 60)
print("DESPUES de tf.concat -- leyendo cada canal desde el tensor final x")
print("=" * 60)
x_np = x.numpy()  # (64,64,12)
for i, fn in enumerate(INPUT_FEATURES):
    print(f"  x[..., {i}] ({fn}): media={x_np[..., i].mean():.4f}")
