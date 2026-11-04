# -*- coding: utf-8 -*-
"""
models/train_convlstm.py

VERSION T=1 -- se elimino el ventaneo temporal sintetico de T=6 dias.

Por que: NDWS no incluye fecha ni ubicacion por muestra -- los ejemplos
son eventos aislados, sin relacion temporal verificable entre si. El
ventaneo de 6 dias del script original juntaba ejemplos consecutivos del
archivo (potencialmente de incendios distintos) y los presentaba a
ConvLSTM como si fueran la evolucion real de un solo incendio. Eso fue
confirmado como invalido para este dataset (ver bitacora, Semana 1).

Con T=1, cada ejemplo se procesa individualmente -- ConvLSTM sigue siendo
una arquitectura recurrente, pero sin inventar una relacion temporal que
el dataset no puede sustentar. Usa la misma informacion temporal que ya
tienen U-Net/MobileNetV2 via el canal viirs_PrevFireMask. Esto se
documenta como limitacion explicita en el articulo: no se pudo evaluar
si el modelado multi-dia ayuda, porque NDWS no lo permite con este
benchmark (ver "Puntos de decision" del plan de extension original).

Otros cambios respecto al script original (iguales a los demas modelos):
1. Particion train/val/test desde shared.partitions.
2. Normalizacion por nombre desde shared.config.FEATURE_RANGES.
3. Checkpoint renombrado a "best_convlstm_model.keras".
4. Evaluacion final unificada via shared.evaluate, sobre test real.
5. Fix de GPU (set_jit(False) + memory growth) para la RTX 5050.
"""

import os
import gc

import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from shared.config import BATCH_SIZE, SEED, THRESHOLD, ONLY_FIRE_PIXELS, PREDICTIONS_PER_FIRE_DIR, FEATURE_RANGES
from shared.partitions import get_train_val_test_files
from shared.evaluate import (
    evaluate_model_unified, make_keras_predict_fn, keras_dataset_to_batches
)

# ---------------------------------------------------------------------
# NUEVO: override de semilla via linea de comandos, para el analisis de
# estabilidad con 3 semillas. Uso: python -m models.train_convlstm 1001
# ---------------------------------------------------------------------
import sys
import numpy as np
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else SEED
print(f"*** Semilla usada en esta corrida: {SEED} ***")
tf.random.set_seed(SEED)
np.random.seed(SEED)

# FIX de GPU -- mismo ajuste que en los otros modelos
tf.config.optimizer.set_jit(False)
gpus_detectadas = tf.config.list_physical_devices('GPU')
for gpu in gpus_detectadas:
    tf.config.experimental.set_memory_growth(gpu, True)

# ---------------------------------------------------------------------
# 1. Particion de datos
# ---------------------------------------------------------------------
train_files, validation_files, test_files = get_train_val_test_files()

# ---------------------------------------------------------------------
# 2. Parsing de TFRecords -- identico a los otros modelos
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
    feature_description = {}
    for feature_name in INPUT_FEATURES + OUTPUT_FEATURES:
        feature_description[feature_name] = tf.io.FixedLenFeature([64, 64], tf.float32)

    parsed_example = tf.io.parse_single_example(example_proto, feature_description)

    inputs_list = []
    for feature_name in INPUT_FEATURES:
        feature = tf.expand_dims(parsed_example[feature_name], axis=-1)
        normalized_feature = normalize_feature(feature, feature_name)
        inputs_list.append(normalized_feature)
    inputs = tf.concat(inputs_list, axis=-1)  # [64, 64, 12]

    # AGREGA la dimension temporal T=1 -- sin ventaneo, sin mezclar ejemplos
    inputs_seq = tf.expand_dims(inputs, axis=0)  # [1, 64, 64, 12]

    label = tf.expand_dims(tf.clip_by_value(parsed_example['viirs_FireMask'], 0.0, 1.0), axis=-1)
    return inputs_seq, label


def get_dataset(file_list, batch_size: int, shuffle: bool = True, seed: int = SEED) -> tf.data.Dataset:
    dataset = tf.data.Dataset.list_files(file_list, shuffle=False)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ---------------------------------------------------------------------
# 3. Datasets de entrenamiento y validacion
# ---------------------------------------------------------------------
T = 1  # sin ventaneo -- cada ejemplo es su propia "secuencia" de 1 paso

train_dataset = (get_dataset(train_files, batch_size=BATCH_SIZE)
                  .shuffle(1000)
                  .cache()
                  .repeat()
                  .prefetch(tf.data.AUTOTUNE))

validation_dataset = (get_dataset(validation_files, batch_size=BATCH_SIZE)
                       .cache()
                       .repeat()
                       .prefetch(tf.data.AUTOTUNE))

num_samples = sum(1 for f in train_files for _ in tf.data.TFRecordDataset(f))
STEPS_PER_EPOCH = num_samples // BATCH_SIZE
print(f"Total muestras de entrenamiento: {num_samples}")
print(f"Steps per epoch: {STEPS_PER_EPOCH}")


# ---------------------------------------------------------------------
# 4. Metricas custom -- identicas a los otros modelos
# ---------------------------------------------------------------------
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
        iou = intersection / (union + tf.keras.backend.epsilon())
        self.total_iou.assign_add(tf.reduce_mean(iou))
        self.count.assign_add(1.0)

    def result(self):
        return self.total_iou / tf.maximum(self.count, 1.0)

    def reset_states(self):
        self.total_iou.assign(0.0)
        self.count.assign(0.0)


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
        dice = (2. * intersection) / (
            tf.reduce_sum(y_true, axis=1) + tf.reduce_sum(y_pred, axis=1) + tf.keras.backend.epsilon())
        self.total_dice.assign_add(tf.reduce_mean(dice))
        self.count.assign_add(1.0)

    def result(self):
        return self.total_dice / tf.maximum(self.count, 1.0)

    def reset_states(self):
        self.total_dice.assign(0.0)
        self.count.assign(0.0)


# ---------------------------------------------------------------------
# 5. Arquitectura ConvLSTM -- igual al original, solo input_shape con T=1
# ---------------------------------------------------------------------
def build_convlstm_model(input_shape=(T, 64, 64, 12), dropout=0.2):
    inp = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True)(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.ConvLSTM2D(32, (3, 3), padding="same", return_sequences=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    out = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same")(x)

    return tf.keras.Model(inp, out, name="ConvLSTM_T1_NextDayFire")


def dice_loss(y_true, y_pred, smooth=1e-6):
    y_pred = tf.clip_by_value(y_pred, smooth, 1 - smooth)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2. * intersection + smooth) / (union + smooth)


def combined_loss(y_true, y_pred):
    bce = BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return tf.reduce_mean(0.5 * bce + 0.5 * dice)


# ---------------------------------------------------------------------
# 6. Entrenamiento
# ---------------------------------------------------------------------
convlstm_model = build_convlstm_model()
convlstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    loss=combined_loss,
    metrics=[IoUScore(), DiceCoefficient(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()],
)
convlstm_model.summary()

EPOCHS = 50
early_stopping = EarlyStopping(monitor='val_dice_coefficient', patience=4,
                                restore_best_weights=True, verbose=1, mode="max")
reduce_lr = ReduceLROnPlateau(monitor='val_dice_coefficient', factor=0.7,
                               patience=2, min_lr=1e-6, verbose=1, mode='max')
checkpoint = ModelCheckpoint(f"best_convlstm_model_seed{SEED}.keras", monitor="val_dice_coefficient",
                              save_best_only=True, verbose=1, mode='max')

history = convlstm_model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_dataset,
    validation_steps=STEPS_PER_EPOCH // 4,
    callbacks=[early_stopping, reduce_lr, checkpoint],
)

# ---------------------------------------------------------------------
# 7. EVALUACION FINAL -- sobre test real
# ---------------------------------------------------------------------
print("Evaluando sobre el TEST set real (T=1, sin ventaneo sintetico)...")

test_dataset_eval = get_dataset(test_files, batch_size=BATCH_SIZE, shuffle=False, seed=SEED)

predict_fn = make_keras_predict_fn(convlstm_model)
results_df = evaluate_model_unified(
    predict_fn=predict_fn,
    eval_batches=keras_dataset_to_batches(test_dataset_eval),
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
output_csv = os.path.join(PREDICTIONS_PER_FIRE_DIR, f"convlstm_test_results_seed{SEED}.csv")
results_df.to_csv(output_csv, index=False)
print(f"Resultados por muestra guardados en: {output_csv}")

gc.collect()
tf.keras.backend.clear_session()
