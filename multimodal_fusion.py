"""
=============================================================================
  Explainable and Fair Monkeypox Detection Using an Optimized VGG19 Model
  -------------------------------------------------------------------------
  Component 4 (Optional) — Light Multimodal Fusion
  
  Adds 4 simulated/real symptom features on top of VGG19 image embeddings:
    fever            (0 / 1)
    lymph_swelling   (0 / 1)
    lesion_count     (integer, normalised)
    rash_duration    (days, normalised)

  If you have a real symptom CSV (one row per image, filename as key),
  set  SYMPTOM_CSV = "symptoms.csv"  and the script will read it.
  Otherwise it synthesises plausible values so the pipeline still runs.

  Architecture:
    VGG19 features (512-d)  ──┐
                               ├─► Fusion MLP ─► sigmoid output
    Symptom MLP   (8-d)    ──┘

  Run AFTER mpox_vgg19_main.py (needs best_vgg19.keras and the data/ folder).
=============================================================================
"""

import os, csv, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, auc,
)

# ── settings ─────────────────────────────────────────────────────────────────
IMG_SIZE     = 228
BATCH_SIZE   = 32
EPOCHS       = 20
LR           = 5e-5
PATIENCE     = 5
DATA_DIR     = "data"
OUTPUT_DIR   = "outputs"
FUSION_DIR   = os.path.join(OUTPUT_DIR, "fusion")
SYMPTOM_CSV  = None          # set to "symptoms.csv" if you have real data
SEED         = 42

os.makedirs(FUSION_DIR, exist_ok=True)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAMES = ["Monkeypox", "Others"]


# ═══════════════════════════════════════════════════════════════════════════
# 1.  SYMPTOM TABLE  (real or synthesised)
# ═══════════════════════════════════════════════════════════════════════════
def load_or_synthesise_symptoms(filepaths, y_true, csv_path=None, seed=42):
    """
    Returns a (N, 4) float32 array:
      col 0 – fever            [0/1]
      col 1 – lymph_swelling   [0/1]
      col 2 – lesion_count     [0–1 normalised from 0–50]
      col 3 – rash_duration    [0–1 normalised from 0–21 days]

    If csv_path exists, read from it (expects columns:
      filename, fever, lymph_swelling, lesion_count, rash_duration).
    Otherwise synthesise plausible values correlated with the true label.
    """
    if csv_path and os.path.isfile(csv_path):
        df = pd.read_csv(csv_path).set_index("filename")
        out = []
        for fp in filepaths:
            key = os.path.basename(fp)
            row = df.loc[key] if key in df.index else pd.Series(
                {"fever": 0, "lymph_swelling": 0,
                 "lesion_count": 0, "rash_duration": 0}
            )
            out.append([
                float(row["fever"]),
                float(row["lymph_swelling"]),
                float(row["lesion_count"]) / 50.0,
                float(row["rash_duration"]) / 21.0,
            ])
        return np.array(out, dtype=np.float32)

    # ── synthesise ──────────────────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    N   = len(filepaths)
    feats = np.zeros((N, 4), dtype=np.float32)
    for i, label in enumerate(y_true):
        if label == 0:   # Monkeypox — higher symptom burden
            feats[i, 0] = float(rng.binomial(1, 0.80))   # fever  (80%)
            feats[i, 1] = float(rng.binomial(1, 0.70))   # lymph  (70%)
            feats[i, 2] = float(rng.integers(5, 40)) / 50.0
            feats[i, 3] = float(rng.integers(3, 18)) / 21.0
        else:            # Others — lower symptom burden
            feats[i, 0] = float(rng.binomial(1, 0.20))
            feats[i, 1] = float(rng.binomial(1, 0.15))
            feats[i, 2] = float(rng.integers(0, 10)) / 50.0
            feats[i, 3] = float(rng.integers(0, 5))  / 21.0
    print("[i] Using SYNTHESISED symptom features (no CSV found).")
    return feats


# ═══════════════════════════════════════════════════════════════════════════
# 2.  EXTRACT VGG19 EMBEDDINGS  (frozen base)
# ═══════════════════════════════════════════════════════════════════════════
def build_feature_extractor():
    """Returns model that outputs GAP features from frozen VGG19."""
    base = VGG19(weights="imagenet", include_top=False,
                 input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    return models.Model(inp, x, name="VGG19_extractor")


def extract_embeddings(extractor, data_dir, split):
    """
    Returns (embeddings, y_true, filepaths) for a split folder.
    """
    gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        os.path.join(data_dir, split),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=16,
        class_mode="binary",
        shuffle=False,
    )
    embs = extractor.predict(gen, verbose=1)
    return embs, gen.classes, np.array(gen.filepaths)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  FUSION MLP MODEL
# ═══════════════════════════════════════════════════════════════════════════
def build_fusion_model(img_feat_dim=512, sym_feat_dim=4):
    """
    Dual-branch MLP:
      Image branch:   img_feat_dim → 256 → 128
      Symptom branch: sym_feat_dim → 16  → 8
      Concat         → 64 → 1 (sigmoid)
    """
    # image branch
    img_in  = tf.keras.Input(shape=(img_feat_dim,), name="image_features")
    x_img   = layers.Dense(256, activation="relu")(img_in)
    x_img   = layers.BatchNormalization()(x_img)
    x_img   = layers.Dropout(0.3)(x_img)
    x_img   = layers.Dense(128, activation="relu")(x_img)

    # symptom branch
    sym_in  = tf.keras.Input(shape=(sym_feat_dim,), name="symptom_features")
    x_sym   = layers.Dense(16, activation="relu")(sym_in)
    x_sym   = layers.Dense(8,  activation="relu")(x_sym)

    # fusion
    merged  = layers.Concatenate()([x_img, x_sym])
    x       = layers.Dense(64, activation="relu")(merged)
    x       = layers.Dropout(0.2)(x)
    out     = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=[img_in, sym_in], outputs=out,
                         name="Fusion_VGG19_Symptom")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 4.  TRAIN & EVALUATE FUSION MODEL
# ═══════════════════════════════════════════════════════════════════════════
def train_fusion(model, X_img_tr, X_sym_tr, y_tr,
                 X_img_va, X_sym_va, y_va):
    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(
            os.path.join(FUSION_DIR, "best_fusion.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=3, min_lr=1e-7, verbose=1),
    ]
    history = model.fit(
        [X_img_tr, X_sym_tr], y_tr,
        validation_data=([X_img_va, X_sym_va], y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_list,
        verbose=1,
    )
    return history


def evaluate_fusion(model, X_img, X_sym, y_true, split_name="Test"):
    y_prob = model.predict([X_img, X_sym], verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    print(f"\n{'='*60}")
    print(f"FUSION MODEL — {split_name} Evaluation")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    try:
        roc = roc_auc_score(y_true, y_prob)
        print(f"ROC-AUC: {roc:.4f}")
    except ValueError:
        roc = float("nan")
    print(f"{'='*60}\n")
    return y_true, y_pred, y_prob


# ═══════════════════════════════════════════════════════════════════════════
# 5.  COMPARISON PLOT  (image-only VGG19 vs fusion)
# ═══════════════════════════════════════════════════════════════════════════
def comparison_bar(image_only_metrics, fusion_metrics):
    """
    Both dicts: {"Accuracy": float, "F1": float, "ROC-AUC": float}
    """
    metrics = list(image_only_metrics.keys())
    x = np.arange(len(metrics))
    w = 0.32

    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - w/2, [image_only_metrics[m] for m in metrics],
                w, label="VGG19 (Image only)", color="#4C72B0")
    b2 = ax.bar(x + w/2, [fusion_metrics[m] for m in metrics],
                w, label="VGG19 + Symptoms (Fusion)", color="#55A868")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Image-Only VGG19 vs Multimodal Fusion",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(FUSION_DIR, "image_vs_fusion_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[✓] Saved comparison chart: {out}")


# ═══════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Multimodal Fusion — VGG19 + Symptom Features")
    print("=" * 60 + "\n")

    # --- Step A: Extract VGG19 embeddings for all splits
    extractor = build_feature_extractor()

    print("Extracting train embeddings …")
    X_img_tr, y_tr, fp_tr = extract_embeddings(extractor, DATA_DIR, "train")
    print("Extracting val embeddings …")
    X_img_va, y_va, fp_va = extract_embeddings(extractor, DATA_DIR, "val")
    print("Extracting test embeddings …")
    X_img_te, y_te, fp_te = extract_embeddings(extractor, DATA_DIR, "test")

    # --- Step B: Load or synthesise symptom features
    X_sym_tr = load_or_synthesise_symptoms(fp_tr, y_tr, SYMPTOM_CSV, SEED)
    X_sym_va = load_or_synthesise_symptoms(fp_va, y_va, SYMPTOM_CSV, SEED)
    X_sym_te = load_or_synthesise_symptoms(fp_te, y_te, SYMPTOM_CSV, SEED)

    print(f"\nImage feat shape : {X_img_tr.shape}")
    print(f"Symptom feat shape: {X_sym_tr.shape}\n")

    # --- Step C: Build and train fusion model
    fusion_model = build_fusion_model(
        img_feat_dim=X_img_tr.shape[1],
        sym_feat_dim=X_sym_tr.shape[1],
    )
    fusion_model.summary()

    history = train_fusion(
        fusion_model,
        X_img_tr, X_sym_tr, y_tr,
        X_img_va, X_sym_va, y_va,
    )

    # --- Step D: Evaluate on test set
    y_true_f, y_pred_f, y_prob_f = evaluate_fusion(
        fusion_model, X_img_te, X_sym_te, y_te
    )

    # --- Step E: Load baseline VGG19 image-only results for comparison
    y_true_b = np.load(os.path.join(OUTPUT_DIR, "y_true.npy"))
    y_pred_b = np.load(os.path.join(OUTPUT_DIR, "y_pred.npy"))
    y_prob_b = np.load(os.path.join(OUTPUT_DIR, "y_prob.npy"))

    from sklearn.metrics import accuracy_score, f1_score
    image_only = {
        "Accuracy": round(accuracy_score(y_true_b, y_pred_b), 4),
        "F1":       round(f1_score(y_true_b, y_pred_b, zero_division=0), 4),
        "ROC-AUC":  round(roc_auc_score(y_true_b, y_prob_b), 4),
    }
    fusion_res = {
        "Accuracy": round(accuracy_score(y_true_f, y_pred_f), 4),
        "F1":       round(f1_score(y_true_f, y_pred_f, zero_division=0), 4),
        "ROC-AUC":  round(roc_auc_score(y_true_f, y_prob_f), 4),
    }

    print(f"Image-only  → {image_only}")
    print(f"Fusion      → {fusion_res}")

    comparison_bar(image_only, fusion_res)

    print(f"\n[✓] Fusion outputs saved to: {FUSION_DIR}")
    print("[✓] Multimodal fusion pipeline complete.\n")
