"""
=============================================================================
  Explainable and Fair Monkeypox Detection Using an Optimized VGG19 Model
  -------------------------------------------------------------------------
  Component 1 of 3 — Model Training & Evaluation
  
  Dataset expected layout:
    data/
      train/
        Monkeypox/      <- mpox images
        Others/         <- non-mpox images
      val/
        Monkeypox/
        Others/
      test/
        Monkeypox/
        Others/

  If you have a flat folder (all images together), run dataset_splitter.py first.
=============================================================================
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend — safe on any machine
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc,
    precision_score, recall_score, f1_score
)

# ── reproducibility ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── hyper-parameters (match your draft) ─────────────────────────────────────
IMG_SIZE    = 228          # paper uses 228×228
BATCH_SIZE  = 50
EPOCHS      = 30
LR          = 1e-4
PATIENCE    = 5            # early-stopping patience
DROPOUT     = 0.3

DATA_DIR    = "data"       # root folder containing train/ val/ test/
OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ["Monkeypox", "Others"]   # alphabetical = Keras default order


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA GENERATORS
# ═══════════════════════════════════════════════════════════════════════════
def build_generators():
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        zoom_range=0.15,
        rotation_range=15,
        brightness_range=[0.8, 1.2],
        width_shift_range=0.10,
        height_shift_range=0.10,
        fill_mode="nearest",
    )
    val_test_aug = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_aug.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=SEED,
    )
    val_gen = val_test_aug.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False,
    )
    test_gen = val_test_aug.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=1,
        class_mode="binary",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


# ═══════════════════════════════════════════════════════════════════════════
# 2. MODEL ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════
def build_model():
    base = VGG19(weights="imagenet",
                 include_top=False,
                 input_shape=(IMG_SIZE, IMG_SIZE, 3))

    # Freeze all layers first; we fine-tune the last block
    base.trainable = False
    for layer in base.layers[-4:]:      # last conv block (block5_conv1..4)
        layer.trainable = True

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="VGG19_Mpox")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
# 3. TRAINING
# ═══════════════════════════════════════════════════════════════════════════
def train(model, train_gen, val_gen):
    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "best_vgg19.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1
        ),
        callbacks.CSVLogger(os.path.join(OUTPUT_DIR, "training_log.csv")),
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=cb_list,
        verbose=1,
    )
    return history


# ═══════════════════════════════════════════════════════════════════════════
# 4. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
def evaluate(model, test_gen):
    """Returns y_true, y_pred (binary), y_prob (sigmoid scores)."""
    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=1).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = test_gen.classes
    return y_true, y_pred, y_prob


def print_metrics(y_true, y_pred, y_prob):
    report = classification_report(y_true, y_pred,
                                   target_names=CLASS_NAMES)
    roc = roc_auc_score(y_true, y_prob)
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    print(f"ROC-AUC : {roc:.4f}")
    print("=" * 60 + "\n")
    return roc


# ═══════════════════════════════════════════════════════════════════════════
# 5. PLOTS
# ═══════════════════════════════════════════════════════════════════════════
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in zip(
        axes,
        [("accuracy", "val_accuracy"), ("loss", "val_loss")],
        ["Accuracy", "Loss"],
    ):
        ax.plot(history.history[metric[0]], label=f"Train {title}", linewidth=2)
        ax.plot(history.history[metric[1]], label=f"Val {title}",
                linewidth=2, linestyle="--")
        ax.set_title(f"Training vs Validation {title}", fontsize=13)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close()
    print(f"[✓] Saved training_curves.png")


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix — VGG19 Mpox Classifier", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print(f"[✓] Saved confusion_matrix.png")


def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2,
            label=f"VGG19 (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Mpox vs Others", fontsize=13)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print(f"[✓] Saved roc_curve.png")


# ═══════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Mpox VGG19 Classifier — Training Pipeline")
    print("=" * 60)

    train_gen, val_gen, test_gen = build_generators()
    print(f"\nClass indices : {train_gen.class_indices}")
    print(f"Train samples : {train_gen.samples}")
    print(f"Val   samples : {val_gen.samples}")
    print(f"Test  samples : {test_gen.samples}\n")

    model = build_model()
    model.summary()

    history = train(model, train_gen, val_gen)

    # --- load best weights (already restored by EarlyStopping)
    model.load_weights(os.path.join(OUTPUT_DIR, "best_vgg19.keras"))

    y_true, y_pred, y_prob = evaluate(model, test_gen)
    roc_auc = print_metrics(y_true, y_pred, y_prob)

    plot_history(history)
    plot_confusion_matrix(y_true, y_pred)
    plot_roc(y_true, y_prob)

    # Save raw predictions for fairness script
    np.save(os.path.join(OUTPUT_DIR, "y_true.npy"), y_true)
    np.save(os.path.join(OUTPUT_DIR, "y_pred.npy"), y_pred)
    np.save(os.path.join(OUTPUT_DIR, "y_prob.npy"), y_prob)
    np.save(os.path.join(OUTPUT_DIR, "test_filepaths.npy"),
            np.array(test_gen.filepaths))

    print("\n[✓] All outputs saved to:", OUTPUT_DIR)
    print("[✓] Run gradcam_xai.py next for explainability visualizations.")
    print("[✓] Run fairness_audit.py next for subgroup analysis.\n")
