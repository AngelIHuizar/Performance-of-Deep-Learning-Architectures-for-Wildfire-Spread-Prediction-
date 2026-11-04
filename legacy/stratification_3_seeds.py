# -*- coding: utf-8 -*-
"""
analysis/stratification_3_seeds.py

Repite el analisis de estratificacion por tamano de incendio (como
bootstrap_stratification.py), pero usando la serie de IoU por muestra
PROMEDIADA ENTRE LAS 3 SEMILLAS de cada modelo (U-Net, ConvLSTM, xLSTM),
en vez de una sola corrida. Esto responde directamente al punto 3 del plan
del asesor: "estratificacion por tamano con resultados completos [de las
3 semillas]".

Requiere que ya hayas corrido las 9 combinaciones (3 modelos x 3 semillas).
"""

import os
import numpy as np
import pandas as pd

PRED_DIR = "outputs/predictions_per_fire"
SEEDS = [1001, 2002, 3003]
MODELOS = ["unet", "convlstm", "xlstm"]

# ---------------------------------------------------------------------
# 1. Cargar los 9 CSVs
# ---------------------------------------------------------------------
dfs = {}
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
    print("*** Faltan estos archivos -- el script se detiene, corre las 9 combinaciones primero ***")
    for f in faltantes:
        print(f"  - {f}")
    raise SystemExit(1)

# ---------------------------------------------------------------------
# 2. Verificar alineamiento: target_pixels debe coincidir entre TODAS las
#    9 corridas (mismo test set, mismo orden, shuffle=False siempre)
# ---------------------------------------------------------------------
print("Verificando alineamiento completo (9 corridas)...")
ref = dfs["unet"][SEEDS[0]]["target_pixels"].to_numpy()
ok = True
for modelo in MODELOS:
    for seed in SEEDS:
        actual = dfs[modelo][seed]["target_pixels"].to_numpy()
        if len(actual) != len(ref) or not np.allclose(actual, ref):
            print(f"  *** ALERTA: {modelo} semilla {seed} no coincide con la referencia ***")
            ok = False
if ok:
    print("OK: las 9 corridas estan perfectamente alineadas por muestra.\n")
else:
    print("Hay desalineamientos -- revisar antes de confiar en este analisis.\n")

target_pixels = ref  # ya verificado que es igual en las 9 corridas

# ---------------------------------------------------------------------
# 3. Serie de IoU por muestra, promediada entre las 3 semillas, por modelo
# ---------------------------------------------------------------------
iou_promediado = {}
for modelo in MODELOS:
    arrs = [dfs[modelo][s]["iou"].to_numpy() for s in SEEDS]
    iou_promediado[modelo] = np.mean(arrs, axis=0)

# ---------------------------------------------------------------------
# 4. Estratos (mismos limites que en el analisis original)
# ---------------------------------------------------------------------
BINS = [0, 25, 100, 500, np.inf]
LABELS = ["1-25", "26-100", "101-500", ">500"]
size_stratum = pd.cut(target_pixels, bins=BINS, labels=LABELS, right=True)

print("Distribucion de incendios por estrato de tamano:")
print(pd.Series(size_stratum).value_counts().sort_index())


def bootstrap_ci_diff(a, b, n_boot=10_000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = a - b
    boot_means = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    alpha = 1 - ci
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return diffs.mean(), lo, hi


PARES = [("xlstm", "unet"), ("xlstm", "convlstm"), ("convlstm", "unet")]

resultados = []
for model_a, model_b in PARES:
    print(f"\n{'=' * 60}")
    print(f"Comparacion: {model_a} vs {model_b} (IoU, promedio de 3 semillas)")
    print(f"{'=' * 60}")
    for estrato in LABELS:
        mask = (size_stratum == estrato)
        a = iou_promediado[model_a][mask]
        b = iou_promediado[model_b][mask]
        n = mask.sum()
        mean_diff, lo, hi = bootstrap_ci_diff(a, b)
        sig = "SI" if (lo > 0 or hi < 0) else "no"
        print(f"  {estrato}: n={n}  diff={mean_diff:+.4f}  IC=[{lo:.4f}, {hi:.4f}]  significativo={sig}")
        resultados.append({
            "comparacion": f"{model_a} vs {model_b}", "stratum": estrato, "n": n,
            "mean_diff": mean_diff, "ci_low": lo, "ci_high": hi, "significativo": sig,
        })

df_resultados = pd.DataFrame(resultados)
out_path = os.path.join(PRED_DIR, "estratificacion_3_semillas.csv")
df_resultados.to_csv(out_path, index=False)
print(f"\nResultados guardados en: {out_path}")
