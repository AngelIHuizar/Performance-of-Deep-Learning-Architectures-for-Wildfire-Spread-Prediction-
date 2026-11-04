"""
analysis/inspect_all_tfrecord_fields.py

Hasta ahora solo hemos leido los 13 campos conocidos (12 INPUT_FEATURES +
viirs_FireMask). Este script inspecciona el TFRecord SIN asumir un esquema
fijo, para ver si existe algun campo adicional (id de incendio, fecha,
latitud/longitud, etc.) que hayamos estado ignorando.
"""

import tensorflow as tf
from shared.partitions import get_train_val_test_files

train_files, _, _ = get_train_val_test_files()

# Leer un solo ejemplo crudo, sin especificar feature_description, para
# extraer el esquema completo directamente del protobuf.
raw_dataset = tf.data.TFRecordDataset([train_files[0]])
raw_record = next(iter(raw_dataset))

example = tf.train.Example()
example.ParseFromString(raw_record.numpy())

print("=" * 60)
print("TODOS los campos encontrados en el TFRecord (esquema completo):")
print("=" * 60)

known_fields = {'elevation', 'wind_75', 'wind_avg', 'tmp_75', 'tmp_day',
                 'avg_sph', 'pr', 'pdsi', 'NDVI', 'population', 'erc',
                 'viirs_PrevFireMask', 'viirs_FireMask'}

all_fields = sorted(example.features.feature.keys())
for field in all_fields:
    feature = example.features.feature[field]
    kind = feature.WhichOneof('kind')
    if kind == 'float_list':
        n_values = len(feature.float_list.value)
        sample_vals = list(feature.float_list.value[:3])
    elif kind == 'bytes_list':
        n_values = len(feature.bytes_list.value)
        sample_vals = [v[:50] for v in feature.bytes_list.value[:1]]
    elif kind == 'int64_list':
        n_values = len(feature.int64_list.value)
        sample_vals = list(feature.int64_list.value[:3])
    else:
        n_values = 0
        sample_vals = None

    marca = "  (ya conocido)" if field in known_fields else "  *** CAMPO NUEVO, NO USADO ANTES ***"
    print(f"{field}{marca}")
    print(f"    tipo={kind}  n_valores={n_values}  muestra={sample_vals}")

print("\n" + "=" * 60)
campos_nuevos = set(all_fields) - known_fields
if campos_nuevos:
    print(f"Se encontraron {len(campos_nuevos)} campos NO usados hasta ahora:")
    for c in campos_nuevos:
        print(f"  - {c}")
    print("\nRevisa si alguno de estos podria servir como identificador de")
    print("incendio/evento o fecha (ej. nombres tipo 'id', 'date', 'lat', 'lon',")
    print("'fire_id', 'time', etc.)")
else:
    print("No se encontraron campos adicionales -- el esquema conocido")
    print("(13 campos) es todo lo que existe en este TFRecord. NDWS NO provee")
    print("un identificador de incendio/evento ni fecha por muestra.")
