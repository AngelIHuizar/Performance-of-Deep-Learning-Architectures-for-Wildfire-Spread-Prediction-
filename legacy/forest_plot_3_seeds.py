"""
analysis/forest_plot_3_seeds.py

Genera el forest plot final usando estratificacion_3_semillas.csv (ya
calculado por stratification_3_seeds.py) -- no vuelve a correr el
bootstrap, solo grafica.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRED_DIR = "outputs/predictions_per_fire"
FIG_DIR = "outputs/figures"

df_res = pd.read_csv(os.path.join(PRED_DIR, "estratificacion_3_semillas.csv"))

NOMBRES_BONITOS = {
    "xlstm vs unet": "xLSTM (T=1) vs U-Net\n(promedio 3 semillas)",
    "xlstm vs convlstm": "xLSTM (T=1) vs ConvLSTM (T=1)\n(promedio 3 semillas)",
    "convlstm vs unet": "ConvLSTM (T=1) vs U-Net\n(promedio 3 semillas)",
}
LABELS = ["1-25", "26-100", "101-500", ">500"]
comparaciones = list(NOMBRES_BONITOS.keys())
y_positions = np.arange(len(LABELS))

fig, axes = plt.subplots(1, len(comparaciones), figsize=(5 * len(comparaciones), 4.5), sharey=True)

for ax, comp in zip(axes, comparaciones):
    sub = df_res[df_res["comparacion"] == comp].set_index("stratum").loc[LABELS]
    colors = ["#2ca02c" if s == "SI" else "#888888" for s in sub["significativo"]]

    ax.errorbar(
        sub["mean_diff"], y_positions,
        xerr=[sub["mean_diff"] - sub["ci_low"], sub["ci_high"] - sub["mean_diff"]],
        fmt="o", capsize=4, color="black", ecolor="black", markersize=0,
    )
    for y, x, c in zip(y_positions, sub["mean_diff"], colors):
        ax.scatter(x, y, color=c, s=70, zorder=3, edgecolor="black", linewidth=0.5)

    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(LABELS)
    ax.set_title(NOMBRES_BONITOS[comp], fontsize=10)
    ax.set_xlabel("Diferencia de IoU\n(A - B)")
    ax.grid(axis="x", alpha=0.3)

axes[0].set_ylabel("Estrato de tamaño de incendio (píxeles)")
fig.suptitle(
    "Diferencia de IoU por estrato (bootstrap, IC 95%, promedio de 3 semillas)\n"
    "Verde = significativo | Gris = no significativo",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.90])

os.makedirs(FIG_DIR, exist_ok=True)
png_path = os.path.join(FIG_DIR, "forest_plot_iou_3_seeds.png")
pdf_path = os.path.join(FIG_DIR, "forest_plot_iou_3_seeds.pdf")
fig.savefig(png_path, dpi=200, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"Figura guardada en: {png_path}")
print(f"Figura guardada en: {pdf_path}")
