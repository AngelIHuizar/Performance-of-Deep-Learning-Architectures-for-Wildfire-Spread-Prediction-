"""
analysis/forest_plot.py

Genera el grafico de bosque (forest plot) de las comparaciones bootstrap
por estrato de tamano de incendio, usando el archivo ya guardado por
bootstrap_stratification.py (outputs/predictions_per_fire/merged_all_models.csv).

Guarda una figura en outputs/figures/forest_plot_iou.png y .pdf (para
usar directo en el manuscrito).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRED_DIR = "outputs/predictions_per_fire"
FIG_DIR = "outputs/figures"
MERGED_PATH = os.path.join(PRED_DIR, "merged_all_models.csv")

if not os.path.exists(MERGED_PATH):
    raise FileNotFoundError(
        f"No se encontro {MERGED_PATH} -- corre primero "
        f"analysis/bootstrap_stratification.py"
    )

merged = pd.read_csv(MERGED_PATH)

BINS = [0, 25, 100, 500, np.inf]
LABELS = ["1-25", "26-100", "101-500", ">500"]
if "size_stratum" not in merged.columns:
    merged["size_stratum"] = pd.cut(merged["target_pixels"], bins=BINS, labels=LABELS, right=True)

PARES_DE_INTERES = [
    ("xlstm", "unet"),
    ("xlstm", "convlstm"),
    ("convlstm", "unet"),
    ("mobilenetv2", "unet"),
]

NOMBRES_BONITOS = {
    "unet": "U-Net",
    "mobilenetv2": "MobileNetV2",
    "convlstm": "ConvLSTM (T=1)",
    "xlstm": "xLSTM (T=1)",
}


def bootstrap_diff_ci(scores_a, scores_b, n_boot=10_000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(scores_a)
    if n < 5:
        return np.nan, np.nan, np.nan
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = scores_a[idx].mean() - scores_b[idx].mean()
    alpha = 1 - ci
    ci_low, ci_high = np.percentile(boot_diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean_diff = scores_a.mean() - scores_b.mean()
    return mean_diff, ci_low, ci_high


# ---------------------------------------------------------------------
# Calcular resultados para las 4 comparaciones x 4 estratos
# ---------------------------------------------------------------------
resultados = []
for model_a, model_b in PARES_DE_INTERES:
    for estrato in LABELS:
        sub = merged[merged["size_stratum"] == estrato]
        mean_diff, ci_low, ci_high = bootstrap_diff_ci(
            sub[f"iou_{model_a}"].to_numpy(),
            sub[f"iou_{model_b}"].to_numpy(),
        )
        resultados.append({
            "comparacion": f"{NOMBRES_BONITOS[model_a]} vs {NOMBRES_BONITOS[model_b]}",
            "stratum": estrato,
            "mean_diff": mean_diff,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": len(sub),
        })

df_res = pd.DataFrame(resultados)

# ---------------------------------------------------------------------
# Grafico: un panel por comparacion, estratos en el eje Y
# ---------------------------------------------------------------------
comparaciones = [f"{NOMBRES_BONITOS[a]} vs {NOMBRES_BONITOS[b]}" for a, b in PARES_DE_INTERES]
n_comp = len(comparaciones)

fig, axes = plt.subplots(1, n_comp, figsize=(4.2 * n_comp, 4.5), sharey=True)
if n_comp == 1:
    axes = [axes]

y_positions = np.arange(len(LABELS))

for ax, comp in zip(axes, comparaciones):
    sub = df_res[df_res["comparacion"] == comp].set_index("stratum").loc[LABELS]

    colors = ["#2ca02c" if excl else "#888888"
              for excl in ((sub["ci_low"] > 0) | (sub["ci_high"] < 0))]

    ax.errorbar(
        sub["mean_diff"], y_positions,
        xerr=[sub["mean_diff"] - sub["ci_low"], sub["ci_high"] - sub["mean_diff"]],
        fmt="o", capsize=4, color="black", ecolor="black", markersize=0,
    )
    for i, (y, x, c) in enumerate(zip(y_positions, sub["mean_diff"], colors)):
        ax.scatter(x, y, color=c, s=70, zorder=3, edgecolor="black", linewidth=0.5)

    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(LABELS)
    ax.set_title(comp, fontsize=10)
    ax.set_xlabel("Diferencia de IoU\n(A - B)")
    ax.grid(axis="x", alpha=0.3)

axes[0].set_ylabel("Estrato de tamaño de incendio (píxeles)")

fig.suptitle(
    "Diferencia de IoU por estrato de tamaño de incendio (bootstrap, IC 95%)\n"
    "Verde = intervalo excluye cero (diferencia significativa) | Gris = no significativa",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.90])

os.makedirs(FIG_DIR, exist_ok=True)
png_path = os.path.join(FIG_DIR, "forest_plot_iou.png")
pdf_path = os.path.join(FIG_DIR, "forest_plot_iou.pdf")
fig.savefig(png_path, dpi=200, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

print(f"Figura guardada en: {png_path}")
print(f"Figura guardada en: {pdf_path} (vectorial, para el manuscrito)")

df_res.to_csv(os.path.join(PRED_DIR, "forest_plot_data.csv"), index=False)
print(f"Datos de la figura guardados en: {os.path.join(PRED_DIR, 'forest_plot_data.csv')}")
