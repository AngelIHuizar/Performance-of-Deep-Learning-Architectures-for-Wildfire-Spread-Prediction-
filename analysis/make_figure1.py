# -*- coding: utf-8 -*-
"""
analysis/make_figure1.py

Nueva Figura 1 (R2-6): muestras DISTINTAS a las de MCPR, con letra grande
(R1-11). Elige del conjunto de PRUEBA una muestra de un estrato de tamano de
incendio y dibuja sus 12 canales de entrada (valores originales, con unidades)
mas la mascara de fuego del dia siguiente.

Uso:
    python -m analysis.make_figure1                  # estrato 101-500, semilla 7
    python -m analysis.make_figure1 101-500 11       # estrato y semilla de seleccion
    python -m analysis.make_figure1 ">500" 3

Genera outputs/figures/figure1_<estrato>_seed<semilla>.png y .pdf, e imprime el
indice de la muestra elegida. Cambia la semilla para obtener otra muestra.
IMPORTANTE: no se puede saber desde aqui que muestras uso MCPR; compara
visualmente con su Figura 1 y cambia la semilla si coincide.
"""
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from shared.partitions import get_train_val_test_files

STRATUM = sys.argv[1] if len(sys.argv) > 1 else "101-500"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
LIMITS = {"1-25": (1, 25), "26-100": (26, 100), "101-500": (101, 500), ">500": (501, 10**9)}
lo, hi = LIMITS[STRATUM]

INPUTS = ['elevation', 'wind_75', 'wind_avg', 'tmp_75', 'tmp_day', 'avg_sph',
          'pr', 'pdsi', 'NDVI', 'population', 'erc', 'viirs_PrevFireMask']
TARGET = 'viirs_FireMask'
# (titulo, cmap). Ajustar los titulos de wind_75/tmp_75 cuando se aclare el
# significado de "75" (comentario R1-4 del revisor).
TITLES = {
    'elevation': ("Elevation (m)", "terrain"),
    'wind_75': ("Wind speed, wind_75 (m/s)", "viridis"),
    'wind_avg': ("Mean wind speed (m/s)", "viridis"),
    'tmp_75': ("Temperature, tmp_75 (°C)", "magma"),
    'tmp_day': ("Daytime temperature (°C)", "magma"),
    'avg_sph': ("Specific humidity (kg/kg)", "Blues"),
    'pr': ("Precipitation (mm)", "Blues"),
    'pdsi': ("PDSI (unitless)", "RdBu"),
    'NDVI': ("NDVI (scaled ×10⁴)", "YlGn"),
    'population': ("Population density (people/km²)", "Purples"),
    'erc': ("Energy Release Component", "YlOrRd"),
    'viirs_PrevFireMask': ("Fire mask at day t", None),
}

_, _, test_files = get_train_val_test_files()


def parse(proto):
    spec = {f: tf.io.FixedLenFeature([64, 64], tf.float32) for f in INPUTS + [TARGET]}
    return tf.io.parse_single_example(proto, spec)


def dataset():
    ds = tf.data.Dataset.list_files(test_files, shuffle=False)
    ds = ds.interleave(lambda f: tf.data.TFRecordDataset(f), num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    return ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)


# Paso 1: indices de muestras del estrato (por pixeles de fuego del dia t+1)
cands = [i for i, ex in enumerate(dataset().as_numpy_iterator())
         if lo <= int(np.clip(ex[TARGET], 0, 1).sum()) <= hi and ex['viirs_PrevFireMask'].sum() > 0]
rng = np.random.default_rng(SEED)
idx = int(rng.choice(cands))
print(f"Estrato {STRATUM}: {len(cands)} candidatas; muestra elegida = indice {idx} (semilla {SEED})")

# Paso 2: extraer la muestra elegida
sample = next(ex for i, ex in enumerate(dataset().as_numpy_iterator()) if i == idx)

plt.rcParams.update({"font.size": 15, "font.family": "serif"})
fig, axes = plt.subplots(3, 5, figsize=(21, 13.5))
mask_cmap = ListedColormap(["#1a1a1a", "#ff8c00"])

for ax, name in zip(axes.flat, INPUTS + [TARGET]):
    if name in INPUTS:
        title, cmap = TITLES[name]
    else:
        title, cmap = "Fire mask at day t+1 (target)", None
    data = sample[name]
    if cmap is None:
        im = ax.imshow(np.clip(data, 0, 1), cmap=mask_cmap, vmin=0, vmax=1)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, ticks=[0, 1])
        cb.ax.set_yticklabels(["no fire", "fire"])
    else:
        const = float(np.ptp(data)) == 0.0
        if const:   # canal constante en esta muestra (p. ej. precipitacion = 0)
            v = float(data.flat[0])
            im = ax.imshow(data, cmap=cmap, vmin=v - 0.5, vmax=v + 0.5)
        else:
            im = ax.imshow(data, cmap=cmap)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        if const:
            ax.text(0.5, 0.5, f"constant\n({v:g})", transform=ax.transAxes, ha="center",
                    va="center", fontsize=16, color="black",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
    cb.ax.tick_params(labelsize=13)
    ax.set_title(title, fontsize=15, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
# 13 paneles de datos en una cuadrilla de 15: los dos sobrantes se ocultan
for ax in list(axes.flat)[13:]:
    ax.axis("off")
axes.flat[13].text(0.0, 0.5, "Each panel: 64 km × 64 km\n(1 km per pixel)\nTest-set sample",
                   fontsize=16, va="center")
fig.tight_layout()

os.makedirs("outputs/figures", exist_ok=True)
stem = f"outputs/figures/figure1_{STRATUM.replace('>', 'gt')}_seed{SEED}"
fig.savefig(stem + ".png", dpi=250, bbox_inches="tight")
fig.savefig(stem + ".pdf", bbox_inches="tight")
print("Figura guardada en", stem + ".png / .pdf")
