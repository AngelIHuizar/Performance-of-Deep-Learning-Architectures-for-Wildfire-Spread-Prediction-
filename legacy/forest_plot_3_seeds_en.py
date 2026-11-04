"""
analysis/forest_plot_3_seeds_en.py

Version en ingles del forest plot final (3 semillas), usando los datos ya
calculados por stratification_3_seeds.py (estratificacion_3_semillas.csv).
No vuelve a correr el bootstrap, solo grafica con etiquetas en ingles.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRED_DIR = "outputs/predictions_per_fire"
FIG_DIR = "outputs/figures"

df_res = pd.read_csv(os.path.join(PRED_DIR, "estratificacion_3_semillas.csv"))

NICE_NAMES = {
    "xlstm vs unet": "xLSTM (T=1) vs U-Net\n(3-seed average)",
    "xlstm vs convlstm": "xLSTM (T=1) vs ConvLSTM (T=1)\n(3-seed average)",
    "convlstm vs unet": "ConvLSTM (T=1) vs U-Net\n(3-seed average)",
}
LABELS = ["1-25", "26-100", "101-500", ">500"]
comparisons = list(NICE_NAMES.keys())
y_positions = np.arange(len(LABELS))

fig, axes = plt.subplots(1, len(comparisons), figsize=(5 * len(comparisons), 4.5), sharey=True)

for ax, comp in zip(axes, comparisons):
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
    ax.set_title(NICE_NAMES[comp], fontsize=10)
    ax.set_xlabel("IoU difference\n(A - B)")
    ax.grid(axis="x", alpha=0.3)

axes[0].set_ylabel("Fire-size stratum (pixels)")
fig.suptitle(
    "IoU difference by stratum (bootstrap, 95% CI, 3-seed average)\n"
    "Green = significant | Gray = not significant",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.90])

os.makedirs(FIG_DIR, exist_ok=True)
png_path = os.path.join(FIG_DIR, "forest_plot_iou_3_seeds_EN.png")
pdf_path = os.path.join(FIG_DIR, "forest_plot_iou_3_seeds_EN.pdf")
fig.savefig(png_path, dpi=200, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"Figure saved to: {png_path}")
print(f"Figure saved to: {pdf_path}")
