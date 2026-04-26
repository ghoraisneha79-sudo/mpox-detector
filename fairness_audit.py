"""
=============================================================================
  Explainable and Fair Monkeypox Detection Using an Optimized VGG19 Model
  -------------------------------------------------------------------------
  Component 3 of 3 — Fairness & Robustness Audit

  Run AFTER mpox_vgg19_main.py so that:
    outputs/best_vgg19.keras     exists
    outputs/test_filepaths.npy   exists
    outputs/y_true.npy           exists
    outputs/y_pred.npy           exists
    outputs/y_prob.npy           exists

  Subgroups created from image properties (no demographic labels needed):
    • Brightness  — mean pixel intensity  (low vs high)
    • Contrast    — std of pixel intensity (low vs high)
    • Saturation  — HSV saturation channel mean (low vs high)
    • Class       — Monkeypox vs Others  (always available)
=============================================================================
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
from sklearn.calibration import calibration_curve

# ── settings ─────────────────────────────────────────────────────────────────
OUTPUT_DIR   = "outputs"
FAIRNESS_DIR = os.path.join(OUTPUT_DIR, "fairness")
os.makedirs(FAIRNESS_DIR, exist_ok=True)

CLASS_NAMES = ["Monkeypox", "Others"]

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD SAVED PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════
filepaths = np.load(os.path.join(OUTPUT_DIR, "test_filepaths.npy"),
                    allow_pickle=True)
y_true    = np.load(os.path.join(OUTPUT_DIR, "y_true.npy"))
y_pred    = np.load(os.path.join(OUTPUT_DIR, "y_pred.npy"))
y_prob    = np.load(os.path.join(OUTPUT_DIR, "y_prob.npy"))

print(f"Loaded {len(filepaths)} test samples.\n")


# ═══════════════════════════════════════════════════════════════════════════
# 2. EXTRACT IMAGE-LEVEL FEATURES
# ═══════════════════════════════════════════════════════════════════════════
def extract_image_stats(path):
    """Return (brightness, contrast, saturation) for a single image."""
    img = cv2.imread(path)
    if img is None:
        return np.nan, np.nan, np.nan
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
    hsv      = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness  = float(gray.mean())
    contrast    = float(gray.std())
    saturation  = float(hsv[:, :, 1].mean() / 255.0)
    return brightness, contrast, saturation


print("Extracting image statistics …")
stats = [extract_image_stats(p) for p in filepaths]
brightness  = np.array([s[0] for s in stats])
contrast    = np.array([s[1] for s in stats])
saturation  = np.array([s[2] for s in stats])
print(f"  Brightness  — mean={brightness.mean():.3f}  std={brightness.std():.3f}")
print(f"  Contrast    — mean={contrast.mean():.3f}  std={contrast.std():.3f}")
print(f"  Saturation  — mean={saturation.mean():.3f}  std={saturation.std():.3f}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 3. DEFINE SUBGROUPS  (median split for each continuous feature)
# ═══════════════════════════════════════════════════════════════════════════
def binary_split(values, name):
    """Return (labels_array, low_label, high_label) split at median."""
    median = np.median(values)
    labels = np.where(values < median, f"Low {name}", f"High {name}")
    return labels, f"Low {name}", f"High {name}"


brightness_group, *_ = binary_split(brightness, "Brightness")
contrast_group,   *_ = binary_split(contrast,   "Contrast")
saturation_group, *_ = binary_split(saturation, "Saturation")
class_group           = np.array([CLASS_NAMES[t] for t in y_true])


# ═══════════════════════════════════════════════════════════════════════════
# 4. METRIC COMPUTATION PER SUBGROUP
# ═══════════════════════════════════════════════════════════════════════════
def subgroup_metrics(group_labels, y_true, y_pred, y_prob):
    """Return a DataFrame with per-subgroup metrics."""
    rows = []
    for grp in np.unique(group_labels):
        mask = group_labels == grp
        if mask.sum() < 3:
            continue
        yt, yp, ypr = y_true[mask], y_pred[mask], y_prob[mask]
        row = {
            "Subgroup":   grp,
            "N":          int(mask.sum()),
            "Accuracy":   round(accuracy_score(yt, yp), 4),
            "Precision":  round(precision_score(yt, yp, zero_division=0), 4),
            "Recall":     round(recall_score(yt, yp, zero_division=0), 4),
            "F1":         round(f1_score(yt, yp, zero_division=0), 4),
        }
        try:
            row["ROC-AUC"] = round(roc_auc_score(yt, ypr), 4)
        except ValueError:
            row["ROC-AUC"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


subgroup_definitions = {
    "Brightness":  brightness_group,
    "Contrast":    contrast_group,
    "Saturation":  saturation_group,
    "Class":       class_group,
}

all_tables = {}
print("=" * 60)
print("FAIRNESS AUDIT — SUBGROUP METRICS")
print("=" * 60)
for feature_name, group_arr in subgroup_definitions.items():
    df = subgroup_metrics(group_arr, y_true, y_pred, y_prob)
    all_tables[feature_name] = df
    print(f"\n── {feature_name} ──")
    print(df.to_string(index=False))

# Save combined CSV
combined = pd.concat(
    [df.assign(Feature=name) for name, df in all_tables.items()],
    ignore_index=True,
)[["Feature", "Subgroup", "N", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
csv_path = os.path.join(FAIRNESS_DIR, "subgroup_metrics.csv")
combined.to_csv(csv_path, index=False)
print(f"\n[✓] Saved subgroup_metrics.csv")


# ═══════════════════════════════════════════════════════════════════════════
# 5. FAIRNESS BAR CHART
# ═══════════════════════════════════════════════════════════════════════════
def plot_fairness_bars(all_tables):
    metrics = ["Accuracy", "F1", "ROC-AUC"]
    n_features = len(all_tables)
    fig, axes = plt.subplots(1, n_features,
                             figsize=(6 * n_features, 6), sharey=False)
    if n_features == 1:
        axes = [axes]

    palette = sns.color_palette("Set2", 4)

    for ax, (feature_name, df) in zip(axes, all_tables.items()):
        x = np.arange(len(df))
        width = 0.22
        for i, metric in enumerate(metrics):
            vals = df[metric].fillna(0).values
            bars = ax.bar(x + i * width, vals, width,
                          label=metric, color=palette[i])
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7,
                )
        ax.set_xticks(x + width)
        ax.set_xticklabels(df["Subgroup"].tolist(), rotation=20, ha="right",
                           fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"Subgroup Performance\nby {feature_name}", fontsize=11)
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Fairness & Robustness Audit — VGG19 Mpox Classifier",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(FAIRNESS_DIR, "fairness_bar_chart.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved fairness_bar_chart.png")


plot_fairness_bars(all_tables)


# ═══════════════════════════════════════════════════════════════════════════
# 6. BRIGHTNESS / CONTRAST SCATTER  (diagnostic plot)
# ═══════════════════════════════════════════════════════════════════════════
def plot_scatter_diagnostic():
    correct = (y_true == y_pred).astype(int)
    colors  = ["#e74c3c" if c == 0 else "#2ecc71" for c in correct]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(brightness, y_prob, c=colors, alpha=0.7, edgecolors="k",
                    linewidths=0.3, s=50)
    axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Image Brightness (mean pixel)")
    axes[0].set_ylabel("Predicted Probability (Mpox)")
    axes[0].set_title("Model Confidence vs Brightness")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(contrast, y_prob, c=colors, alpha=0.7, edgecolors="k",
                    linewidths=0.3, s=50)
    axes[1].axhline(0.5, color="grey", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Image Contrast (pixel std)")
    axes[1].set_ylabel("Predicted Probability (Mpox)")
    axes[1].set_title("Model Confidence vs Contrast")
    axes[1].grid(alpha=0.3)

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor="#2ecc71", label="Correct"),
                    Patch(facecolor="#e74c3c", label="Incorrect")]
    fig.legend(handles=legend_elems, loc="upper right", fontsize=10)
    plt.suptitle("Image Quality vs Prediction Confidence",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FAIRNESS_DIR, "scatter_brightness_contrast.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved scatter_brightness_contrast.png")


plot_scatter_diagnostic()


# ═══════════════════════════════════════════════════════════════════════════
# 7. CALIBRATION CURVE
# ═══════════════════════════════════════════════════════════════════════════
def plot_calibration():
    prob_true, prob_pred = calibration_curve(y_true, y_prob,
                                             n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, "s-", color="#4C72B0",
            label="VGG19 (Mpox)", linewidth=2, markersize=7)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives", fontsize=12)
    ax.set_title("Calibration Curve — Mpox Classifier", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FAIRNESS_DIR, "calibration_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved calibration_curve.png")


plot_calibration()


# ═══════════════════════════════════════════════════════════════════════════
# 8. DISPARITY SUMMARY  (max gap across subgroups)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DISPARITY SUMMARY (max metric gap across subgroups)")
print("=" * 60)
for feature_name, df in all_tables.items():
    for metric in ["Accuracy", "F1", "ROC-AUC"]:
        vals = df[metric].dropna().values
        if len(vals) >= 2:
            gap = vals.max() - vals.min()
            print(f"  {feature_name:<12} {metric:<10} gap = {gap:.4f}")

print(f"\n[✓] All fairness outputs saved to: {FAIRNESS_DIR}")
print("[✓] Pipeline complete!\n")
print("Paper-ready outputs summary:")
print(f"  outputs/training_curves.png")
print(f"  outputs/confusion_matrix.png")
print(f"  outputs/roc_curve.png")
print(f"  outputs/gradcam/paper_gradcam_grid.png")
print(f"  outputs/fairness/fairness_bar_chart.png")
print(f"  outputs/fairness/scatter_brightness_contrast.png")
print(f"  outputs/fairness/calibration_curve.png")
print(f"  outputs/fairness/subgroup_metrics.csv")
