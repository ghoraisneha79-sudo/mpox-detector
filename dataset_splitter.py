"""
=============================================================================
  dataset_splitter.py — One-time utility
  -------------------------------------------------------------------------
  Use this ONLY if your images are in a flat directory like:
    raw_images/
      Monkeypox/   (all mpox images)
      Others/      (all non-mpox images)

  This script creates the train/val/test split at 70 / 15 / 15 ratio and
  copies (does NOT move) images into:
    data/train/  data/val/  data/test/

  Run once before mpox_vgg19_main.py.
=============================================================================
"""

import os
import shutil
import random

RAW_DIR    = "raw_images"      # <-- change to your source folder
DATA_DIR   = "data"
SPLIT      = (0.70, 0.15, 0.15)   # train / val / test
SEED       = 42
CLASSES    = ["Monkeypox", "Others"]

random.seed(SEED)

for cls in CLASSES:
    src = os.path.join(RAW_DIR, cls)
    all_files = [
        f for f in os.listdir(src)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ]
    random.shuffle(all_files)

    n = len(all_files)
    n_train = int(n * SPLIT[0])
    n_val   = int(n * SPLIT[1])
    splits  = {
        "train": all_files[:n_train],
        "val":   all_files[n_train:n_train + n_val],
        "test":  all_files[n_train + n_val:],
    }

    for split_name, files in splits.items():
        dst_dir = os.path.join(DATA_DIR, split_name, cls)
        os.makedirs(dst_dir, exist_ok=True)
        for fname in files:
            shutil.copy2(os.path.join(src, fname),
                         os.path.join(dst_dir, fname))
        print(f"  {cls:<12} {split_name:<6} → {len(files)} images → {dst_dir}")

print("\n[✓] Dataset split complete. Now run mpox_vgg19_main.py")
