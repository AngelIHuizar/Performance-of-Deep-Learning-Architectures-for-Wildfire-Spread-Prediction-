# -*- coding: utf-8 -*-
"""
analysis/stratification_tuned.py

Estratificacion por tamano de incendio con la evaluacion CONSISTENTE
(umbral elegido en validacion por modelo y semilla; ver seeds_tuned_evaluation.py).
Usa las series por muestra promediadas entre las 3 semillas
(`<modelo>_test_results_tuned_mean.csv`) de los 4 modelos, incluido MobileNetV2.

Salida: outputs/predictions_per_fire/estratificacion_tuned.csv
Uso:    python -m analysis.stratification_tuned
"""
import os
import numpy as np
import pandas as pd

PRED_DIR = "outputs/predictions_per_fire"
MODELOS = ["unet", "mobilenetv2", "convlstm", "xlstm"]
BINS = [0, 25, 100, 500, np.inf]
LABELS = ["1-25", "26-100", "101-500", ">500"]

dfs = {m: pd.read_csv(os.path.join(PRED_DIR, f"{m}_test_results_tuned_mean.csv")) for m in MODELOS}
ref = dfs["unet"]["target_pixels"].to_numpy()
for m in MODELOS:
    assert len(dfs[m]) == len(ref) and np.allclose(dfs[m]["target_pixels"], ref), f"{m} desalineado"
print("OK: los 4 modelos estan alineados por muestra.")

size_stratum = pd.cut(ref, bins=BINS, labels=LABELS, right=True)
print("\nIncendios por estrato:\n", pd.Series(size_stratum).value_counts().sort_index())


def boot_diff(a, b, n_boot=10_000, seed=42):
    rng = np.random.default_rng(seed)
    d = a - b
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return d.mean(), lo, hi


PARES = [("xlstm", "unet"), ("xlstm", "convlstm"), ("convlstm", "unet"), ("mobilenetv2", "unet")]
rows = []
for a_, b_ in PARES:
    print(f"\n=== {a_} vs {b_} (IoU, promedio de 3 semillas) ===")
    for est in LABELS:
        mk = np.asarray(size_stratum == est)
        md, lo, hi = boot_diff(dfs[a_]["iou"].to_numpy()[mk], dfs[b_]["iou"].to_numpy()[mk])
        sig = "SI" if (lo > 0 or hi < 0) else "no"
        print(f"  {est}: n={mk.sum()}  diff={md:+.4f}  IC=[{lo:+.4f}, {hi:+.4f}]  significativo={sig}")
        rows.append({"comparacion": f"{a_} vs {b_}", "stratum": est, "n": int(mk.sum()),
                     "mean_diff": md, "ci_low": lo, "ci_high": hi, "significativo": sig})

pd.DataFrame(rows).to_csv(os.path.join(PRED_DIR, "estratificacion_tuned.csv"), index=False)
print("\nGuardado en", os.path.join(PRED_DIR, "estratificacion_tuned.csv"))
