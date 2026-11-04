# -*- coding: utf-8 -*-
"""
analysis/stratification_ap.py

Estratificacion por tamano con una metrica INDEPENDIENTE DEL UMBRAL: la
Average Precision (AP) por muestra, promediada entre las 3 semillas.
Sirve para comprobar si las diferencias por estrato de stratification_tuned.py
(que dependen del umbral fijo de cada modelo) se mantienen.

Requiere: <modelo>_test_results_tuned_mean.csv (seeds_tuned_evaluation.py).
Uso:      python -m analysis.stratification_ap
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
strat = pd.cut(ref, bins=BINS, labels=LABELS, right=True)


def boot(a, b, n_boot=10_000, seed=42):
    rng = np.random.default_rng(seed)
    d = a - b
    n = len(d)
    m = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(m, [2.5, 97.5])
    return d.mean(), lo, hi


print("AP medio por estrato y modelo (promedio de 3 semillas):")
tab = {m: [np.nanmean(dfs[m]["ap"].to_numpy()[np.asarray(strat == e)]) for e in LABELS] for m in MODELOS}
print(pd.DataFrame(tab, index=LABELS).round(4).to_string())

PARES = [("xlstm", "unet"), ("xlstm", "convlstm"), ("convlstm", "unet"), ("mobilenetv2", "unet")]
rows = []
for a_, b_ in PARES:
    print(f"\n=== {a_} vs {b_} (AP) ===")
    for e in LABELS:
        mk = np.asarray(strat == e)
        a = dfs[a_]["ap"].to_numpy()[mk]; b = dfs[b_]["ap"].to_numpy()[mk]
        ok = ~(np.isnan(a) | np.isnan(b))
        md, lo, hi = boot(a[ok], b[ok])
        sig = "SI" if (lo > 0 or hi < 0) else "no"
        print(f"  {e}: n={ok.sum()}  diff={md:+.4f}  IC=[{lo:+.4f}, {hi:+.4f}]  significativo={sig}")
        rows.append({"comparacion": f"{a_} vs {b_}", "stratum": e, "n": int(ok.sum()),
                     "mean_diff": md, "ci_low": lo, "ci_high": hi, "significativo": sig})
pd.DataFrame(rows).to_csv(os.path.join(PRED_DIR, "estratificacion_ap.csv"), index=False)
print("\nGuardado en", os.path.join(PRED_DIR, "estratificacion_ap.csv"))
