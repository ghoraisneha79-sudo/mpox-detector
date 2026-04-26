"""
=============================================================================
  run_all.py — Master Pipeline Launcher
  -------------------------------------------------------------------------
  Runs every component in order. Edit the FLAGS below to skip steps
  you've already completed or want to skip.
=============================================================================
"""

import subprocess
import sys
import os
import time

# ── control flags ─────────────────────────────────────────────────────────
RUN_SPLIT      = True    # set False if data/train/ already exists
RUN_TRAIN      = True    # Step 1 — VGG19 training
RUN_GRADCAM    = True    # Step 2 — Grad-CAM XAI
RUN_FAIRNESS   = True    # Step 3 — Fairness audit
RUN_FUSION     = False   # Step 4 — Multimodal fusion (optional, slower)
RUN_BENCHMARK  = True    # Step 5 — Benchmark table
# ──────────────────────────────────────────────────────────────────────────

PYTHON = sys.executable


def run(script, label):
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}")
    print(f"{'='*60}\n")
    t0 = time.time()
    result = subprocess.run([PYTHON, script], check=False)
    elapsed = time.time() - t0
    status = "✓  DONE" if result.returncode == 0 else "✗  FAILED"
    print(f"\n[{status}] {label}  ({elapsed:.0f}s)\n")
    return result.returncode == 0


# Quick sanity check
if not os.path.isdir("raw_images") and not os.path.isdir("data"):
    print("\n[!] ERROR: Neither 'raw_images/' nor 'data/' folder found.")
    print("    • Place your MSLD images in  raw_images/Monkeypox/  and  raw_images/Others/")
    print("    • Or create  data/train/ data/val/ data/test/  manually.\n")
    sys.exit(1)

results = {}

if RUN_SPLIT and not os.path.isdir(os.path.join("data", "train")):
    results["dataset_split"] = run("dataset_splitter.py", "Dataset Split (70/15/15)")
elif RUN_SPLIT:
    print("[i] data/train/ already exists — skipping dataset split.")

if RUN_TRAIN:
    results["train"] = run("mpox_vgg19_main.py", "VGG19 Training & Evaluation")

if RUN_GRADCAM:
    results["gradcam"] = run("gradcam_xai.py", "Grad-CAM XAI Visualizations")

if RUN_FAIRNESS:
    results["fairness"] = run("fairness_audit.py", "Fairness & Robustness Audit")

if RUN_FUSION:
    results["fusion"] = run("multimodal_fusion.py", "Multimodal Fusion Model")

if RUN_BENCHMARK:
    results["benchmark"] = run("benchmark_table.py", "Benchmark Comparison Table")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PIPELINE SUMMARY")
print("=" * 60)
for step, ok in results.items():
    icon = "✓" if ok else "✗"
    print(f"  [{icon}] {step}")

print("\nPaper-ready outputs are in:")
print("  outputs/                  ← training curves, confusion matrix, ROC")
print("  outputs/gradcam/          ← Grad-CAM heatmaps")
print("  outputs/fairness/         ← subgroup metrics & charts")
if RUN_FUSION:
    print("  outputs/fusion/           ← multimodal fusion results")
print("  outputs/benchmark_*.png   ← comparison table & bar chart")
print()
