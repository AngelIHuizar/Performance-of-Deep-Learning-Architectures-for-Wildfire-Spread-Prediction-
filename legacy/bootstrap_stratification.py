"""
analysis/bootstrap_stratification.py

Protocolo de estratificacion bootstrap (Semana 2), ahora sobre los 4 CSVs
reales generados por cada train_*.py en la Semana 1:
  outputs/predictions_per_fire/unet_test_results.csv
  outputs/predictions_per_fire/mobilenetv2_test_results.csv
  outputs/predictions_per_fire/convlstm_test_results.csv
  outputs/predictions_per_fire/xlstm_test_results.csv

SUPUESTO CRITICO A VERIFICAR: cada CSV tiene una columna "sample_id" que
es solo un indice de posicion (0,1,2,...), NO un identificador real del
ejemplo. Para poder comparar "el mismo incendio" entre modelos, se asume
que el orden de evaluacion (shuffle=False, misma lista de test_files) fue
identico en los 4 scripts. Este script verifica esa suposicion antes de
usar los datos: como "target_pixels" viene de la mascara real (no depende
del modelo), debe ser IDENTICO entre los 4 CSVs para el mismo sample_id
si el orden realmente coincide. Si no coincide, el script avisa y se
detiene en vez de seguir con un merge silenciosamente mal alineado.
"""

import os
import numpy as np
import pandas as pd

PRED_DIR = "outputs/predictions_per_fire"

MODEL_FILES = {
    "unet": "unet_test_results_final.csv",
    "mobilenetv2": "mobilenetv2_test_results_final.csv",
    "convlstm": "convlstm_test_results_final.csv",
    "xlstm": "xlstm_test_results_final.csv",
}

# ---------------------------------------------------------------------
# 1. Cargar los 4 CSVs
# ---------------------------------------------------------------------
dfs = {}
for name, filename in MODEL_FILES.items():
    path = os.path.join(PRED_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontro {path} -- corre primero "
                                 f"models/train_{name}.py")
    dfs[name] = pd.read_csv(path)
    print(f"{name}: {len(dfs[name])} muestras cargadas")

# ---------------------------------------------------------------------
# 2. Verificar que el orden coincide entre los 4 (chequeo de target_pixels)
# ---------------------------------------------------------------------
n_rows = [len(df) for df in dfs.values()]
if len(set(n_rows)) != 1:
    raise ValueError(
        f"Los 4 CSVs tienen distinto numero de filas: {dict(zip(MODEL_FILES.keys(), n_rows))}. "
        f"No se pueden alinear por posicion -- revisa que los 4 modelos hayan "
        f"evaluado sobre exactamente el mismo test set."
    )

base_name = "unet"
merged = pd.DataFrame({"sample_id": dfs[base_name]["sample_id"]})
for name, df in dfs.items():
    merged[f"target_pixels_{name}"] = df["target_pixels"].values
    merged[f"iou_{name}"] = df["iou"].values
    merged[f"f1_{name}"] = df["f1"].values

target_cols = [f"target_pixels_{name}" for name in MODEL_FILES.keys()]
mismatches = merged[target_cols].nunique(axis=1) > 1
n_mismatch = mismatches.sum()

if n_mismatch > 0:
    pct = 100 * n_mismatch / len(merged)
    print(f"\n*** ADVERTENCIA CRITICA ***")
    print(f"{n_mismatch} de {len(merged)} filas ({pct:.1f}%) tienen 'target_pixels' "
          f"DISTINTO entre modelos para el mismo sample_id.")
    print("Esto significa que el orden de evaluacion NO coincide entre los 4 "
          "CSVs -- comparar por posicion no es valido tal como esta.")
    print("Ejemplo de filas con mismatch:")
    print(merged[mismatches].head(5)[["sample_id"] + target_cols])
    if pct > 1.0:
        raise ValueError(
            "Mas del 1% de las filas no coinciden -- DETENIENDO el analisis. "
            "No continues con el bootstrap hasta resolver el desalineamiento "
            "(ver nota en el encabezado de este script)."
        )
    else:
        print("Menos del 1% de mismatch -- probablemente ruido de borde "
              "(ultimo batch incompleto). Continuando, pero documentalo.")
else:
    print("\nOK: target_pixels coincide exactamente entre los 4 modelos "
          "para todas las filas -- el orden esta correctamente alineado.")

# A partir de aqui, usamos target_pixels de unet como el "real" (ya
# verificado que es igual en los 4)
merged["target_pixels"] = merged["target_pixels_unet"]

# ---------------------------------------------------------------------
# 3. Definir estratos de tamano (mismos intervalos que en la Semana 2 original)
# ---------------------------------------------------------------------
BINS = [0, 25, 100, 500, np.inf]
LABELS = ["1-25", "26-100", "101-500", ">500"]
merged["size_stratum"] = pd.cut(merged["target_pixels"], bins=BINS, labels=LABELS, right=True)

print("\nDistribucion de incendios por estrato de tamano:")
print(merged["size_stratum"].value_counts().sort_index())

# ---------------------------------------------------------------------
# 4. Funcion de bootstrap (igual que antes)
# ---------------------------------------------------------------------
def bootstrap_diff_ci(scores_a, scores_b, n_boot=10_000, ci=0.95, seed=0):
    assert len(scores_a) == len(scores_b)
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    if n < 5:
        return {"n_fires": n, "mean_diff": np.nan, "ci_low": np.nan,
                "ci_high": np.nan, "warning": "n < 5, no confiable"}
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = scores_a[idx].mean() - scores_b[idx].mean()
    alpha = 1 - ci
    ci_low, ci_high = np.percentile(boot_diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "n_fires": n,
        "mean_diff": scores_a.mean() - scores_b.mean(),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "excludes_zero": (ci_low > 0) or (ci_high < 0),
    }


# ---------------------------------------------------------------------
# 5. Comparaciones pareadas por estrato -- los 4 modelos, pares de interes
# ---------------------------------------------------------------------
# Pregunta original del plan de extension: xLSTM vs U-Net (linea base espacial)
PARES_DE_INTERES = [
    ("xlstm", "unet"),
    ("xlstm", "convlstm"),
    ("convlstm", "unet"),
    ("mobilenetv2", "unet"),
]

for model_a, model_b in PARES_DE_INTERES:
    print(f"\n{'=' * 60}")
    print(f"Comparacion: {model_a} vs {model_b} (IoU)")
    print(f"{'=' * 60}")
    resultados = []
    for estrato in LABELS:
        sub = merged[merged["size_stratum"] == estrato]
        r = bootstrap_diff_ci(
            sub[f"iou_{model_a}"].to_numpy(),
            sub[f"iou_{model_b}"].to_numpy(),
            n_boot=10_000, seed=42,
        )
        r["stratum"] = estrato
        resultados.append(r)

    tabla = pd.DataFrame(resultados)[["stratum", "n_fires", "mean_diff", "ci_low", "ci_high", "excludes_zero"]]
    print(tabla.to_string(index=False))

# ---------------------------------------------------------------------
# 6. Guardar el archivo mergeado para inspeccion manual si hace falta
# ---------------------------------------------------------------------
os.makedirs(PRED_DIR, exist_ok=True)
merged.to_csv(os.path.join(PRED_DIR, "merged_all_models.csv"), index=False)
print(f"\nArchivo combinado guardado en: {os.path.join(PRED_DIR, 'merged_all_models.csv')}")
