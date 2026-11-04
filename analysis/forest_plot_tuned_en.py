# -*- coding: utf-8 -*-
"""
analysis/forest_plot_tuned_en.py

Forest plot (English, large fonts for the journal) of the paired IoU
differences by fire-size stratum, from estratificacion_tuned.csv.
Uso: python -m analysis.forest_plot_tuned_en
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRED_DIR = "outputs/predictions_per_fire"
FIG_DIR = "outputs/figures"
df = pd.read_csv(os.path.join(PRED_DIR, "estratificacion_tuned.csv"))

NAMES = {
    "xlstm vs unet": "xLSTM vs U-Net",
    "xlstm vs convlstm": "xLSTM vs ConvLSTM",
    "convlstm vs unet": "ConvLSTM vs U-Net",
    "mobilenetv2 vs unet": "MobileNetV2 vs U-Net",
}
LABELS = ["1-25", "26-100", "101-500", ">500"]
y = np.arange(len(LABELS))
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "font.family": "serif"})

fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharey=True)
for ax, (key, title) in zip(axes, NAMES.items()):
    s = df[df["comparacion"] == key].set_index("stratum").loc[LABELS]
    ax.hlines(y, s["ci_low"], s["ci_high"], color="black", linewidth=1.8)
    ax.scatter(s["mean_diff"], y, s=90, zorder=3, edgecolor="black", linewidth=0.7,
               color=["#2ca02c" if v == "SI" else "#9a9a9a" for v in s["significativo"]])
    ax.axvline(0, color="red", linestyle="--", linewidth=1.3)
    ax.set_yticks(y); ax.set_yticklabels(LABELS)
    ax.set_title(title)
    ax.set_xlabel("IoU difference (A − B)")
    ax.grid(axis="x", alpha=0.3)
axes[0].set_ylabel("Fire-size stratum (pixels)")
fig.suptitle("Paired IoU difference by fire size (95% bootstrap CI, mean of 3 seeds). "
             "Green: CI excludes zero; grey: it does not.", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
os.makedirs(FIG_DIR, exist_ok=True)
fig.savefig(os.path.join(FIG_DIR, "forest_plot_iou_tuned_EN.png"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(FIG_DIR, "forest_plot_iou_tuned_EN.pdf"), bbox_inches="tight")
print("Figure saved in", FIG_DIR)
