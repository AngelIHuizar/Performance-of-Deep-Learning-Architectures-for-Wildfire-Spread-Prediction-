# -*- coding: utf-8 -*-
"""
analysis/aggregate_seeds.py

Agrega los resultados de las 3 semillas para U-Net, ConvLSTM y xLSTM:
1. Por modelo y metrica: media y desviacion estandar ENTRE LAS 3 SEMILLAS
   (cada semilla aporta un numero -- el promedio de esa corrida sobre el
   test set completo).
2. Diferencia pareada entre modelos con IC bootstrap del 95%: se promedia
   el IoU/F1 por muestra a traves de las 3 semillas (dando una serie por
   muestra mas estable), y sobre esa serie promediada se aplica el mismo
   bootstrap pareado usado en el resto del articulo.

Requiere que ya hayas corrido, para cada modelo, las 3 semillas:
    python -m models.train_unet 1001 / 2002 / 3003
    python -m models.train_convlstm 1001 / 2002 / 3003
    python -m models.train_xlstm 1001 / 2002 / 3003
"""

import os
import numpy as np
import pandas as pd

PRED_DIR = "outputs/predictions_per_fire"
SEEDS = [1001, 2002, 3003]
MODELOS = ["unet", "convlstm", "xlstm"]
METRICAS = ["iou", "f1", "accuracy", "precision", "recall"]

# ---------------------------------------------------------------------
# 1. Cargar los 9 CSVs
# ---------------------------------------------------------------------
dfs = {}  # dfs[modelo][semilla] = DataFrame
faltantes = []
for modelo in MODELOS:
    dfs[modelo] = {}
    for seed in SEEDS:
        path = os.path.join(PRED_DIR, f"{modelo}_test_results_seed{seed}.csv")
        if not os.path.exists(path):
            faltantes.append(path)
            continue
        dfs[modelo][seed] = pd.read_csv(path)

if faltantes:
    print("*** ADVERTENCIA: faltan estos archivos (corridas sin terminar) ***")
    for f in faltantes:
        print(f"  - {f}")
    print("El script continua con lo que SI esta disponible.\n")

# ---------------------------------------------------------------------
# 2. Verificar alineamiento (target_pixels debe coincidir entre semillas
#    del MISMO modelo, ya que la evaluacion siempre usa shuffle=False)
# ---------------------------------------------------------------------
print("Verificando alineamiento de muestras entre semillas...")
for modelo in MODELOS:
    seeds_disponibles = list(dfs[modelo].keys())
    if len(seeds_disponibles) < 2:
        continue
    ref = dfs[modelo][seeds_disponibles[0]]["target_pixels"].to_numpy()
    for seed in seeds_disponibles[1:]:
        actual = dfs[modelo][seed]["target_pixels"].to_numpy()
        if len(actual) != len(ref) or not np.allclose(actual, ref):
            print(f"  *** ALERTA: {modelo} semilla {seed} no coincide en target_pixels "
                  f"con la semilla {seeds_disponibles[0]} -- revisar antes de confiar en esto ***")
        else:
            print(f"  OK: {modelo} semilla {seed} alineada correctamente")

# ---------------------------------------------------------------------
# 3. Tabla de estabilidad: media +/- desviacion ENTRE SEMILLAS, por modelo
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("TABLA DE ESTABILIDAD (media +/- desviacion estandar entre 3 semillas)")
print("=" * 70)

filas_estabilidad = []
for modelo in MODELOS:
    seeds_disponibles = list(dfs[modelo].keys())
    fila = {"modelo": modelo, "n_semillas": len(seeds_disponibles)}
    for metrica in METRICAS:
        valores_por_semilla = [dfs[modelo][s][metrica].mean() for s in seeds_disponibles]
        fila[f"{metrica}_mean"] = np.mean(valores_por_semilla) if valores_por_semilla else np.nan
        fila[f"{metrica}_std"] = np.std(valores_por_semilla, ddof=1) if len(valores_por_semilla) > 1 else np.nan
    filas_estabilidad.append(fila)

df_estabilidad = pd.DataFrame(filas_estabilidad)
pd.set_option("display.width", 200)
print(df_estabilidad.round(4).to_string(index=False))

df_estabilidad.to_csv(os.path.join(PRED_DIR, "tabla_estabilidad_3_semillas.csv"), index=False)

# ---------------------------------------------------------------------
# 4. Serie por muestra promediada entre semillas (para el IC pareado)
# ---------------------------------------------------------------------
def serie_promediada(modelo, metrica):
    seeds_disponibles = list(dfs[modelo].keys())
    arrs = [dfs[modelo][s][metrica].to_numpy() for s in seeds_disponibles]
    return np.mean(arrs, axis=0)


def bootstrap_ci_diff(a, b, n_boot=10_000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = a - b
    boot_means = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    alpha = 1 - ci
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return diffs.mean(), lo, hi


# ---------------------------------------------------------------------
# 5. Diferencias pareadas entre modelos (IoU), usando la serie promediada
#    entre semillas -- mismo criterio de bootstrap que el resto del articulo
# ---------------------------------------------------------------------
PARES = [("xlstm", "unet"), ("xlstm", "convlstm"), ("convlstm", "unet")]

print("\n" + "=" * 70)
print("DIFERENCIAS PAREADAS ENTRE MODELOS (IoU, promediado entre semillas)")
print("=" * 70)

resultados_pareados = []
modelos_completos = [m for m in MODELOS if len(dfs[m]) == len(SEEDS)]
for model_a, model_b in PARES:
    if model_a not in modelos_completos or model_b not in modelos_completos:
        print(f"{model_a} vs {model_b}: SALTADO (faltan semillas de alguno de los dos)")
        continue
    serie_a = serie_promediada(model_a, "iou")
    serie_b = serie_promediada(model_b, "iou")
    mean_diff, lo, hi = bootstrap_ci_diff(serie_a, serie_b)
    sig = "SI" if (lo > 0 or hi < 0) else "no"
    print(f"{model_a} vs {model_b}: diff={mean_diff:+.4f}  IC=[{lo:.4f}, {hi:.4f}]  significativo={sig}")
    resultados_pareados.append({
        "comparacion": f"{model_a} vs {model_b}", "mean_diff": mean_diff,
        "ci_low": lo, "ci_high": hi, "significativo": sig,
    })

df_pareado = pd.DataFrame(resultados_pareados)
df_pareado.to_csv(os.path.join(PRED_DIR, "diferencias_pareadas_3_semillas.csv"), index=False)

print(f"\nArchivos guardados en {PRED_DIR}/:")
print("  - tabla_estabilidad_3_semillas.csv")
print("  - diferencias_pareadas_3_semillas.csv")
