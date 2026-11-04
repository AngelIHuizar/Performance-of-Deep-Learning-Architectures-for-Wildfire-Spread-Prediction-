# -*- coding: utf-8 -*-
"""
models/train_autoencoder.py

NOTA DE NOMBRES: este script entrena el modelo que el manuscrito llama
"MobileNetV2" -- la variable se llama `autoencoder_model` en el codigo
original porque la arquitectura es un encoder-decoder (down_stack/up_stack)
con MobileNetV2 como backbone congelable. Es el mismo modelo, dos nombres.

Cambios respecto al original:
1. Particion train/val/test desde shared.partitions (una sola fuente).
2. Normalizacion por nombre desde shared.config.FEATURE_RANGES (mismo bug
   corregido que en U-Net -- este script tenia la lista desalineada).
3. BUG CORREGIDO: la evaluacion final no fijaba shuffle=False en el
   test_dataset. Esto no afecta la cifra agregada, pero rompe la
   posibilidad de comparar resultados por muestra entre modelos (Semana 2).
4. Checkpoint renombrado a "best_mobilenetv2_model.keras" -- el original
   se llamaba igual que el de U-Net ("best_unet_model.keras") y uno
   sobrescribia al otro si se entrenaban en la misma carpeta.
5. Evaluacion final unificada via shared.evaluate, mismo criterio que
   los demas modelos.
"""

import os
import gc
from typing import Text

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from shared.config import (
    BATCH_SIZE, SEED, THRESHOLD, ONLY_FIRE_PIXELS, PREDICTIONS_PER_FIRE_DIR,
    FEATURE_RANGES,
)
from shared.partitions import get_train_val_test_files
from shared.evaluate import (
    evaluate_model_unified, make_keras_predict_fn, keras_dataset_to_batches
)

# ---------------------------------------------------------------------
# Semilla via linea de comandos (analisis de estabilidad con 3 semillas).
# Uso: python -m models.train_autoencoder 1001
# Sin argumento usa la SEED de shared/config.py.
# ---------------------------------------------------------------------
import sys
import numpy as np
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else SEED
print(f"*** Semilla usada en esta corrida: {SEED} ***")
tf.random.set_seed(SEED)
np.random.seed(SEED)

# FIX: mismo ajuste que en train_unet.py -- la RTX 5050 (compute capability
# 12.0a) es demasiado nueva para los kernels precompilados de TensorFlow.
tf.config.optimizer.set_jit(False)

gpus_detectadas = tf.config.list_physical_devices('GPU')
for gpu in gpus_detectadas:
    tf.config.experimental.set_memory_growth(gpu, True)

# ---------------------------------------------------------------------
# 1. Particion de datos -- misma fuente que los demas modelos
# ---------------------------------------------------------------------
train_files, validation_files, test_files = get_train_val_test_files()

# ---------------------------------------------------------------------
# 2. Parsing de TFRecords -- identico a train_unet.py
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
    inputs = tf.concat(inputs_list, axis=-1)

    label = tf.clip_by_value(parsed_example['viirs_FireMask'], 0.0, 1.0)
    return inputs, label


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
# 3. Datasets de entrenamiento y validacion (para fit)
# ---------------------------------------------------------------------
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
# 4. Metricas custom -- identicas a train_unet.py
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
# 5. Arquitectura: MobileNetV2 backbone + decoder (upsample estilo pix2pix)
#    igual al original, pero con upsample() implementado aqui mismo en
#    vez de importarlo de tensorflow_examples (paquete que solo se
#    instala desde un commit de GitHub, no es reproducible de forma
#    confiable para el articulo).
# ---------------------------------------------------------------------
def upsample(filters, size):
    """Equivalente a tensorflow_examples.models.pix2pix.pix2pix.upsample(),
    sin apply_dropout (el script original tampoco lo usaba)."""
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(
        filters, size, strides=2, padding='same',
        kernel_initializer=initializer, use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())
    return result


def build_CNN_AE_model() -> Model:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=[64, 64, 12],
        include_top=False,
        weights=None
    )

    layer_names = [
        'block_1_expand_relu',   # (64,64)
        'block_3_expand_relu',   # (32,32)
        'block_6_expand_relu',   # (16,16)
        'block_13_expand_relu',  # (8,8)
        'block_16_project',      # (4,4)
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    down_stack.trainable = True

    up_stack = [
        upsample(512, 3),
        upsample(256, 3),
        upsample(128, 3),
        upsample(64, 3),
    ]

    inputs = Input(shape=[64, 64, 12])
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


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
autoencoder_model = build_CNN_AE_model()
autoencoder_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    loss=combined_loss,
    metrics=[IoUScore(), DiceCoefficient(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()],
)
autoencoder_model.summary()

EPOCHS = 50
early_stopping = EarlyStopping(monitor='val_dice_coefficient', patience=4,
                                restore_best_weights=True, verbose=1, mode="max")
reduce_lr = ReduceLROnPlateau(monitor='val_dice_coefficient', factor=0.7,
                               patience=2, min_lr=1e-6, verbose=1, mode='max')
# CORREGIDO: nombre de checkpoint propio (antes colisionaba con el de U-Net)
checkpoint = ModelCheckpoint(f"best_mobilenetv2_model_seed{SEED}.keras", monitor="val_dice_coefficient",
                              save_best_only=True, verbose=1, mode='max')

history = autoencoder_model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_dataset,
    validation_steps=STEPS_PER_EPOCH // 4,
    callbacks=[reduce_lr, checkpoint, early_stopping],
)

# ---------------------------------------------------------------------
# 7. EVALUACION FINAL -- CORREGIDA: shuffle=False explicito
# ---------------------------------------------------------------------
print("Evaluando sobre el TEST set real (shuffle=False)...")

test_dataset_eval = get_dataset(test_files, batch_size=BATCH_SIZE, shuffle=False, seed=SEED)

predict_fn = make_keras_predict_fn(autoencoder_model)
results_df = evaluate_model_unified(
    predict_fn=predict_fn,
    eval_batches=keras_dataset_to_batches(test_dataset_eval),
    sample_ids=None,  # TODO: mismo pendiente que en train_unet.py
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
# Nombrado "mobilenetv2" para que coincida con la terminologia del manuscrito
output_csv = os.path.join(PREDICTIONS_PER_FIRE_DIR, f"mobilenetv2_test_results_seed{SEED}.csv")
results_df.to_csv(output_csv, index=False)
print(f"Resultados por muestra guardados en: {output_csv}")

gc.collect()
tf.keras.backend.clear_session()
