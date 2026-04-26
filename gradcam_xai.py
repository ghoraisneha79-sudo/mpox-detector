"""
=============================================================================
  Explainable and Fair Monkeypox Detection Using an Optimized VGG19 Model
  -------------------------------------------------------------------------
  Component 2 of 3 — Grad-CAM Explainability (XAI)

  Run AFTER mpox_vgg19_main.py so that:
    outputs/best_vgg19.keras     exists
    outputs/test_filepaths.npy   exists
    outputs/y_true.npy           exists
    outputs/y_pred.npy           exists
=============================================================================
"""

import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

# ── settings ─────────────────────────────────────────────────────────────────
IMG_SIZE   = 228
OUTPUT_DIR = "outputs"
GRADCAM_DIR = os.path.join(OUTPUT_DIR, "gradcam")
os.makedirs(GRADCAM_DIR, exist_ok=True)

CLASS_NAMES = ["Monkeypox", "Others"]   # index 0 = Mpox, 1 = Others


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD MODEL & PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════
model      = load_model(os.path.join(OUTPUT_DIR, "best_vgg19.keras"))
filepaths  = np.load(os.path.join(OUTPUT_DIR, "test_filepaths.npy"),
                     allow_pickle=True)
y_true     = np.load(os.path.join(OUTPUT_DIR, "y_true.npy"))
y_pred     = np.load(os.path.join(OUTPUT_DIR, "y_pred.npy"))

# Target the last convolutional layer of VGG19 (block5_conv4)
# Identify the last conv layer inside the base model
base_model = model.get_layer("vgg19")
LAST_CONV_LAYER = "block5_conv4"       # always present in VGG19


# ═══════════════════════════════════════════════════════════════════════════
# 2. GRAD-CAM CORE
# ═══════════════════════════════════════════════════════════════════════════
def load_preprocess(img_path):
    """Load a single image and return (4-D tensor, 3-D numpy array)."""
    img = keras_image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = keras_image.img_to_array(img) / 255.0
    return tf.expand_dims(arr, 0), arr          # (1,H,W,3), (H,W,3)


def make_gradcam_heatmap(img_tensor, model, base_model_name,
                          last_conv_layer_name, pred_index=None):
    """
    Compute Grad-CAM heatmap.
    Returns a (H, W) float32 array normalised to [0, 1].
    """
    # Build a sub-model: inputs → [last conv output, final logits]
    base = model.get_layer(base_model_name)
    conv_layer = base.get_layer(last_conv_layer_name)

    # Grad-model: original input → conv feature maps AND predictions
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(base_model_name).get_layer(
                last_conv_layer_name
            ).output,
            model.output,
        ],
    )

    with tf.GradientTape() as tape:
        img_tensor = tf.cast(img_tensor, tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)

    conv_outputs = conv_outputs[0]                          # (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()                                   # (H, W)


def overlay_heatmap(original_arr, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """Superimpose Grad-CAM heatmap onto the original image."""
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    colored         = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_rgb     = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB) / 255.0
    original_rgb    = np.clip(original_arr, 0, 1)
    superimposed    = alpha * colored_rgb + (1 - alpha) * original_rgb
    return np.clip(superimposed, 0, 1)


# ═══════════════════════════════════════════════════════════════════════════
# 3. SELECT EXAMPLES FOR VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════
def get_sample_indices(y_true, y_pred, category, n=10):
    """
    category: 'tp' | 'tn' | 'fp' | 'fn'
    Returns up to n indices.
    """
    if category == "tp":   mask = (y_true == 0) & (y_pred == 0)   # Mpox correct
    elif category == "tn": mask = (y_true == 1) & (y_pred == 1)   # Others correct
    elif category == "fp": mask = (y_true == 1) & (y_pred == 0)   # Others → Mpox wrong
    elif category == "fn": mask = (y_true == 0) & (y_pred == 1)   # Mpox → Others wrong
    indices = np.where(mask)[0]
    np.random.shuffle(indices)
    return indices[:n]


# ═══════════════════════════════════════════════════════════════════════════
# 4. BATCH VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════
def visualize_batch(indices, title, filename, pred_class_index=0):
    """Save a grid of [original | heatmap overlay] for each index."""
    n = len(indices)
    if n == 0:
        print(f"[!] No samples for: {title}")
        return

    fig = plt.figure(figsize=(14, 3.5 * n))
    gs  = gridspec.GridSpec(n, 2, figure=fig, hspace=0.4, wspace=0.1)

    for row, idx in enumerate(indices):
        path = filepaths[idx]
        img_tensor, img_arr = load_preprocess(path)

        heatmap = make_gradcam_heatmap(
            img_tensor, model, "vgg19", LAST_CONV_LAYER,
            pred_index=pred_class_index,
        )
        overlay = overlay_heatmap(img_arr, heatmap)

        ax_orig = fig.add_subplot(gs[row, 0])
        ax_orig.imshow(img_arr)
        ax_orig.set_title(
            f"Original\nTrue: {CLASS_NAMES[y_true[idx]]}  "
            f"Pred: {CLASS_NAMES[y_pred[idx]]}",
            fontsize=9,
        )
        ax_orig.axis("off")

        ax_cam = fig.add_subplot(gs[row, 1])
        ax_cam.imshow(overlay)
        ax_cam.set_title("Grad-CAM Activation", fontsize=9)
        ax_cam.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.005)
    out_path = os.path.join(GRADCAM_DIR, filename)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved {out_path}  ({n} samples)")


# ═══════════════════════════════════════════════════════════════════════════
# 5. SINGLE SUMMARY FIGURE (paper-ready 2×5 grid)
# ═══════════════════════════════════════════════════════════════════════════
def paper_figure(indices_mpox, indices_others, n_each=5):
    """
    Top row: n_each correct Mpox examples with Grad-CAM.
    Bottom row: n_each correct Others examples with Grad-CAM.
    """
    fig, axes = plt.subplots(2, n_each * 2, figsize=(n_each * 5, 9))
    fig.suptitle(
        "Grad-CAM Visualizations — Correct Classifications\n"
        "(Left: original  |  Right: Grad-CAM overlay)",
        fontsize=13, fontweight="bold",
    )

    def fill_row(row_axes, indices, label, pred_idx):
        for col, idx in enumerate(indices[:n_each]):
            path = filepaths[idx]
            img_tensor, img_arr = load_preprocess(path)
            heatmap = make_gradcam_heatmap(
                img_tensor, model, "vgg19", LAST_CONV_LAYER, pred_idx
            )
            overlay = overlay_heatmap(img_arr, heatmap)

            ax_o = row_axes[col * 2]
            ax_c = row_axes[col * 2 + 1]
            ax_o.imshow(img_arr); ax_o.axis("off")
            ax_c.imshow(overlay); ax_c.axis("off")
            if col == 0:
                ax_o.set_ylabel(label, fontsize=11, fontweight="bold",
                                rotation=0, labelpad=60, va="center")

    fill_row(axes[0], indices_mpox,  "Monkeypox", pred_idx=0)
    fill_row(axes[1], indices_others, "Others",    pred_idx=1)

    plt.tight_layout()
    out_path = os.path.join(GRADCAM_DIR, "paper_gradcam_grid.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved paper-ready Grad-CAM grid: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Grad-CAM XAI — Generating Visualizations")
    print("=" * 60 + "\n")

    np.random.seed(42)

    tp_idx = get_sample_indices(y_true, y_pred, "tp", n=10)  # correct Mpox
    tn_idx = get_sample_indices(y_true, y_pred, "tn", n=10)  # correct Others
    fn_idx = get_sample_indices(y_true, y_pred, "fn", n=5)   # missed Mpox
    fp_idx = get_sample_indices(y_true, y_pred, "fp", n=5)   # false alarms

    print(f"TP (correct Mpox)   : {len(tp_idx)} samples")
    print(f"TN (correct Others) : {len(tn_idx)} samples")
    print(f"FN (missed Mpox)    : {len(fn_idx)} samples")
    print(f"FP (false alarms)   : {len(fp_idx)} samples\n")

    visualize_batch(tp_idx, "Correct Mpox Detections — Grad-CAM",
                    "gradcam_true_positive.png", pred_class_index=0)
    visualize_batch(tn_idx, "Correct Others Classifications — Grad-CAM",
                    "gradcam_true_negative.png", pred_class_index=1)
    visualize_batch(fn_idx, "Missed Mpox Cases (False Negatives) — Grad-CAM",
                    "gradcam_false_negative.png", pred_class_index=0)
    visualize_batch(fp_idx, "False Alarms (False Positives) — Grad-CAM",
                    "gradcam_false_positive.png", pred_class_index=0)

    paper_figure(tp_idx, tn_idx, n_each=5)

    print(f"\n[✓] All Grad-CAM outputs saved to: {GRADCAM_DIR}")
    print("[✓] Run fairness_audit.py for subgroup analysis.\n")
