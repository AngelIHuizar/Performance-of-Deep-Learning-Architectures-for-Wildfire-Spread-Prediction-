"""
shared/config.py

UNA SOLA fuente de verdad para rutas, hiperparametros de evaluacion y semillas.
Los 4 scripts de entrenamiento (U-Net, Autoencoder, ConvLSTM, xLSTM) importan
este archivo en vez de redefinir estos valores cada uno por su lado.

Por que esto importa: la auditoria encontro que el mismo valor (threshold,
dataset_path, only_fire_pixels) estaba escrito 4 veces en 4 archivos
distintos. Bastaba con que UNO se desactualizara para invalidar la
comparacion entre modelos sin que nadie lo notara.
"""

import os

# ---------------------------------------------------------------------
# Dataset path. Set the environment variable WILDFIRE_DATASET_PATH to the
# folder with the *.tfrecord files (see README, "Data"). Default: ./data/ndws
# ---------------------------------------------------------------------
DATASET_PATH = os.environ.get(
    "WILDFIRE_DATASET_PATH",
    os.path.join("data", "ndws"),
)

# ---------------------------------------------------------------------
# Hiperparametros de evaluacion -- FIJOS y compartidos por los 4 modelos
# ---------------------------------------------------------------------
BATCH_SIZE = 32
SEED = 2026

# Umbral de binarizacion. Debe fijarse mirando SOLO el set de validacion,
# nunca el de test. Documentar aqui la justificacion cuando se decida.
THRESHOLD = 0.5

# Si True, la evaluacion descarta muestras sin ningun pixel de incendio
# en la mascara real (ver auditoria: esto cambia el significado de las
# metricas). DEBE ser el mismo valor para los 4 modelos en la tabla final.
ONLY_FIRE_PIXELS = False  # <-- decidir explicitamente, no dejar en default

# ---------------------------------------------------------------------
# Rangos de normalizacion por variable -- UNA SOLA fuente de verdad
# ---------------------------------------------------------------------
# La auditoria encontro que U-Net/Autoencoder/ConvLSTM usaban una LISTA
# indexada por posicion que quedo desalineada con INPUT_FEATURES (6 de 11
# variables terminaban saturadas a una constante). El script de xLSTM ya
# habia corregido esto usando un diccionario por nombre, pero con valores
# numericos DISTINTOS a los que se necesitan aca. Este diccionario es la
# version unica que deben usar los 4 modelos -- si se ajusta un valor,
# se ajusta para los 4 a la vez.
#
# TODO: estos valores salen de los comentarios del script original
# (TITLES_WITH_UNITS). Idealmente reemplazar por los rangos reales
# documentados en el paper de NDWS (Huot et al. 2022) o calculados
# empiricamente sobre el propio dataset, en vez de heredar valores
# puestos a mano en un notebook.
FEATURE_RANGES = {
    "elevation": (1750.0, 2500.0),
    "wind_75": (0.5, 6.0),
    "wind_avg": (0.5, 5.0),
    "tmp_75": (-10.0, 45.0),      # CORREGIDO: estaba en (280,320) asumiendo Kelvin,
                                    # los datos reales vienen en Celsius (rango real: -8 a 38)
    "tmp_day": (-10.0, 45.0),     # CORREGIDO: mismo problema que tmp_75
    "avg_sph": (0.005, 0.015),
    "pr": (0.0, 1.0),
    "pdsi": (-5.0, 5.0),
    "NDVI": (-2000.0, 10000.0),  # CORREGIDO: convencion MODIS/VIIRS de entero
                                    # escalado (x10000), no fraccion 0-1 directa
                                    # (rango real observado: -1976 a 9756)
    "population": (0.0, 1000.0),
    "erc": (0.0, 100.0),         # verificado: rango real (-5.4 a 106) es razonable
    "viirs_PrevFireMask": None,  # binaria, caso especial
}

# ---------------------------------------------------------------------
# Convencion de nombres de archivo del dataset (NDWS)
# ---------------------------------------------------------------------
# Los archivos originales usan estos substrings. OJO: "eval" contiene "val"
# como substring, esto ya causo confusion entre scripts -- por eso ahora
# hay una unica funcion (partitions.py) que resuelve esto una sola vez.
TRAIN_SUBSTR = "train"
VAL_SUBSTR = "eval"     # NO uses "val" a secas -- ver partitions.py
TEST_SUBSTR = "test"

# ---------------------------------------------------------------------
# Rutas de salida
# ---------------------------------------------------------------------
OUTPUT_DIR = "outputs"
PREDICTIONS_PER_FIRE_DIR = os.path.join(OUTPUT_DIR, "predictions_per_fire")
