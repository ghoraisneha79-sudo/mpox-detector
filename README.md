# Explainable & Fair Mpox Detection — VGG19 Pipeline

Complete 7-script pipeline for the paper:  
**"Explainable and Fair Monkeypox Detection Using an Optimized VGG19 Model"**

---

## File overview

| Script | Purpose | Day |
|---|---|---|
| `dataset_splitter.py` | 70/15/15 train/val/test split | Before Day 1 |
| `mpox_vgg19_main.py` | Train VGG19, save metrics & checkpoints | Day 1 |
| `gradcam_xai.py` | Grad-CAM heatmaps (XAI) | Day 2 AM |
| `fairness_audit.py` | Subgroup fairness analysis | Day 2 AM |
| `benchmark_table.py` | Literature comparison table & bar chart | Day 2 PM |
| `multimodal_fusion.py` | *(Optional)* VGG19 + symptom features | Day 2 PM |
| `run_all.py` | One-click master launcher | Any time |

---

## Folder layout expected BEFORE running

```
pox1/
├── raw_images/
│   ├── Monkeypox/    ← 102 mpox images
│   └── Others/       ← 126 non-mpox images
├── dataset_splitter.py
├── mpox_vgg19_main.py
├── gradcam_xai.py
├── fairness_audit.py
├── benchmark_table.py
├── multimodal_fusion.py
├── run_all.py
└── requirements.txt
```

---

## Quickstart — run the entire pipeline at once

```bash
pip install -r requirements.txt
python run_all.py
```

Edit the flags at the top of `run_all.py` to skip steps you've already done.

---

## Step-by-step

### Step 0 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 1 — Split dataset *(run once)*
```bash
python dataset_splitter.py
```
Creates `data/train/`, `data/val/`, `data/test/` at **70 / 15 / 15**.

### Step 2 — Train VGG19 & evaluate *(Day 1)*
```bash
python mpox_vgg19_main.py
```
**Outputs in `outputs/`:**

| File | Description |
|---|---|
| `best_vgg19.keras` | Best model checkpoint |
| `training_curves.png` | Accuracy & loss curves |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `roc_curve.png` | ROC-AUC curve |
| `training_log.csv` | Per-epoch metrics |
| `y_true/pred/prob.npy` | Saved predictions |

### Step 3 — Grad-CAM XAI *(Day 2, morning)*
```bash
python gradcam_xai.py
```
**Outputs in `outputs/gradcam/`:**

| File | Description |
|---|---|
| `paper_gradcam_grid.png` | 2×5 paper-ready Grad-CAM figure |
| `gradcam_true_positive.png` | Correct Mpox + heatmaps |
| `gradcam_true_negative.png` | Correct Others + heatmaps |
| `gradcam_false_negative.png` | Missed Mpox (FN) cases |
| `gradcam_false_positive.png` | False alarm (FP) cases |

### Step 4 — Fairness audit *(Day 2, morning)*
```bash
python fairness_audit.py
```
**Outputs in `outputs/fairness/`:**

| File | Description |
|---|---|
| `subgroup_metrics.csv` | Per-subgroup Acc/F1/AUC table |
| `fairness_bar_chart.png` | Grouped bar chart |
| `scatter_brightness_contrast.png` | Confidence vs image quality |
| `calibration_curve.png` | Model calibration plot |

### Step 5 — Benchmark comparison *(Day 2, afternoon)*
```bash
python benchmark_table.py
```
> **Edit `YOUR_ACC`, `YOUR_F1`, `YOUR_AUC` at the top of this file first!**

**Outputs in `outputs/`:**

| File | Description |
|---|---|
| `benchmark_comparison.csv` | Tabular literature comparison |
| `benchmark_table_figure.png` | Formatted table image for paper |
| `benchmark_accuracy_bar.png` | Horizontal bar comparison chart |

### Step 6 — Multimodal fusion *(Optional, Day 2)*
```bash
python multimodal_fusion.py
```
Fuses VGG19 image embeddings with 4 symptom features (fever, lymph node swelling,
lesion count, rash duration). Synthesises features if no symptom CSV is available.

**Outputs in `outputs/fusion/`:**

| File | Description |
|---|---|
| `best_fusion.keras` | Best fusion model |
| `image_vs_fusion_comparison.png` | Accuracy/F1/AUC side-by-side bar |

---

## Key hyperparameters

| Parameter | Value |
|---|---|
| Image size | 228 × 228 |
| Batch size | 50 |
| Max epochs | 30 |
| Early stopping patience | 5 |
| Learning rate | 0.0001 |
| Dropout | 0.3 |
| Optimizer | Adam |
| Base model | VGG19 (ImageNet) |
| Fine-tuned layers | block5_conv1–4 |

---

## Novelty claim (copy-paste into paper)

> Unlike prior mpox image classifiers that mainly report overall accuracy on
> small public datasets, our framework combines an optimized VGG19 backbone
> with explainable AI (Grad-CAM) and a fairness-oriented subgroup audit to
> improve interpretability and practical trustworthiness.

---

## Paper section → output mapping

| Paper Section | Output File(s) |
|---|---|
| 4. Proposed Model | `training_curves.png` |
| 5. Explainability | `paper_gradcam_grid.png`, `gradcam_*.png` |
| 6. Fairness Audit | `fairness_bar_chart.png`, `subgroup_metrics.csv` |
| 7. Results & Comparison | `confusion_matrix.png`, `roc_curve.png`, `benchmark_table_figure.png` |
