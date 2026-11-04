# -*- coding: utf-8 -*-
"""
models/train_unet.py

Adaptado desde el notebook original de Colab. Cambios respecto al original:

1. La particion train/val/test ya NO se calcula aqui -- viene de
   shared.partitions.get_train_val_test_files(), que falla ruidosamente
   si algo no cuadra (en vez del riesgo de heuristicas divergentes entre
   scripts que encontramos en la auditoria).

2. BUG CORREGIDO: el script original evaluaba el modelo final sobre
   `fixed_viz_dataset` (construido desde validation_files) pero el print
   decia "Evaluating on test data...". Ahora se evalua explicitamente
   sobre un dataset construido desde test_files.

3. La evaluacion final usa shared.evaluate.evaluate_model_unified(), la
   misma funcion que van a usar Autoencoder, ConvLSTM y xLSTM -- para que
   los 4 modelos sean comparables con el mismo criterio.

4. Los resultados por muestra se guardan en un CSV, insumo directo para
   el script de bootstrap de la Semana 2.

TODO PENDIENTE (no resuelto por este script): confirmar si cada muestra
del dataset equivale a un incendio unico, o si un incendio puede tener
varias muestras. Ver nota en shared/evaluate.py.
"""

import os
import gc
from typing import Text

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, Input,
    BatchNormalization, Activation, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
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
# NUEVO: override de semilla via linea de comandos, para el analisis de
# estabilidad con 3 semillas. Uso: python -m models.train_unet 1001
# Si no se pasa argumento, usa la SEED de shared/config.py por defecto.
# ---------------------------------------------------------------------
import sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else SEED
print(f"*** Semilla usada en esta corrida: {SEED} ***")
tf.random.set_seed(SEED)
np.random.seed(SEED)

# FIX: la RTX 5050 (arquitectura Blackwell, compute capability 12.0a) es
# demasiado nueva -- el compilador XLA falla al generar el kernel para
# Conv2DTranspose backward. Desactivar JIT evita que XLA intente compilar
# estas operaciones y usa el camino estandar de cuDNN en su lugar.
tf.config.optimizer.set_jit(False)

# Tambien evita que TensorFlow reserve toda la VRAM de una sola vez --
# esto reduce la probabilidad de que el autotuner falle por falta de
# memoria disponible durante la busqueda de algoritmos.
gpus_detectadas = tf.config.list_physical_devices('GPU')
for gpu in gpus_detectadas:
    tf.config.experimental.set_memory_growth(gpu, True)

# ---------------------------------------------------------------------
# 1. Particion de datos -- UNA sola fuente de verdad (shared/partitions.py)
# ---------------------------------------------------------------------
train_files, validation_files, test_files = get_train_val_test_files()

# ---------------------------------------------------------------------
# 2. Parsing de TFRecords -- IDENTICO al script original, sin cambios
# ---------------------------------------------------------------------
INPUT_FEATURES = ['elevation', 'wind_75', 'wind_avg', 'tmp_75', 'tmp_day', 'avg_sph',
                  'pr', 'pdsi', 'NDVI', 'population', 'erc', 'viirs_PrevFireMask']
OUTPUT_FEATURES = ['viirs_FireMask']

# FEATURE_RANGES ahora vive en shared/config.py como diccionario por nombre
# (una sola fuente de verdad, usada por los 4 modelos -- ver nota ahi).


def normalize_feature(feature, feature_name):
    if feature_name == 'viirs_PrevFireMask':
        return feature * 5.0
    # CORREGIDO: se indexa por NOMBRE (diccionario), no por posicion en una
    # lista -- este es justo el patron que causaba la desalineacion original.
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
# 3. Datasets de entrenamiento y validacion (para fit) -- igual al original
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
# 4. Metricas custom de entrenamiento -- igual al original (para monitoreo
#    durante el fit, NO son la evaluacion final reportada en la tabla)
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
# 5. Arquitectura U-Net -- igual al original, sin cambios
# ---------------------------------------------------------------------
def build_unet(input_shape=(64, 64, 12), dropout_rate=0.3, l2_reg=1e-4):
    inputs = Input(shape=input_shape)

    def conv_block(x, filters, dropout=False):
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (3, 3), padding='same', kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if dropout:
            x = Dropout(dropout_rate)(x)
        return x

    c1 = conv_block(inputs, 64, dropout=True); p1 = MaxPooling2D((2, 2))(c1)
    c2 = conv_block(p1, 128, dropout=True); p2 = MaxPooling2D((2, 2))(c2)
    c3 = conv_block(p2, 256, dropout=True); p3 = MaxPooling2D((2, 2))(c3)
    c4 = conv_block(p3, 512, dropout=True); p4 = MaxPooling2D((2, 2))(c4)
    c5 = conv_block(p4, 1024, dropout=True)

    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = Concatenate(axis=-1)([u6, c4]); c6 = conv_block(u6, 512, dropout=True)
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = Concatenate(axis=-1)([u7, c3]); c7 = conv_block(u7, 256, dropout=True)
    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = Concatenate(axis=-1)([u8, c2]); c8 = conv_block(u8, 128, dropout=True)
    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = Concatenate(axis=-1)([u9, c1]); c9 = conv_block(u9, 64, dropout=True)

    outputs = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(c9)
    return Model(inputs, outputs)


def dice_loss(y_true, y_pred, smooth=1e-6):
    y_pred = tf.clip_by_value(y_pred, smooth, 1 - smooth)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2. * intersection + smooth) / (union + smooth)


def combined_loss(y_true, y_pred):
    # CORREGIDO: el script original subido usaba focal_tversky_loss (ver abajo),
    # pero esa version resulto ser de una prueba anterior. La configuracion real
    # usada para el articulo de MCPR es esta perdida combinada BCE+Dice al 50/50,
    # igual que MobileNetV2, ConvLSTM y xLSTM -- necesaria para poder afirmar
    # honestamente "misma configuracion experimental para los 4 modelos".
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return tf.reduce_mean(0.5 * bce + 0.5 * dice_loss(y_true, y_pred))


def focal_tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6):
    # Se conserva la definicion por si se usa en algun analisis exploratorio,
    # pero YA NO es la perdida usada para entrenar (ver combined_loss arriba).
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), smooth, 1.0 - smooth)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return tf.pow((1.0 - tversky), gamma)


# ---------------------------------------------------------------------
# 6. Entrenamiento -- igual al original
# ---------------------------------------------------------------------
unet_model = build_unet()
unet_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
    loss=combined_loss,
    metrics=[IoUScore(), DiceCoefficient(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()],
)
unet_model.summary()

EPOCHS = 30
early_stopping = EarlyStopping(monitor='val_dice_coefficient', patience=4,
                                restore_best_weights=True, verbose=1, mode="max")
reduce_lr = ReduceLROnPlateau(monitor='val_dice_coefficient', factor=0.7,
                               patience=2, min_lr=1e-6, verbose=1, mode='max')
checkpoint = ModelCheckpoint(f"best_unet_model_seed{SEED}.keras", monitor="val_dice_coefficient",
                              save_best_only=True, verbose=1, mode='max')

history = unet_model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_dataset,
    validation_steps=STEPS_PER_EPOCH // 4,
    callbacks=[reduce_lr, checkpoint, early_stopping],
)

# ---------------------------------------------------------------------
# 7. EVALUACION FINAL -- CORREGIDA: ahora sobre test_files, no validation
# ---------------------------------------------------------------------
print("Evaluando sobre el TEST set real (no validation)...")

test_dataset_eval = get_dataset(test_files, batch_size=BATCH_SIZE, shuffle=False, seed=SEED)

predict_fn = make_keras_predict_fn(unet_model)
results_df = evaluate_model_unified(
    predict_fn=predict_fn,
    eval_batches=keras_dataset_to_batches(test_dataset_eval),
    sample_ids=None,  # TODO: reemplazar por fire_id real si se confirma que existe
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
output_csv = os.path.join(PREDICTIONS_PER_FIRE_DIR, f"unet_test_results_seed{SEED}.csv")
results_df.to_csv(output_csv, index=False)
print(f"Resultados por muestra guardados en: {output_csv}")

gc.collect()
tf.keras.backend.clear_session()
