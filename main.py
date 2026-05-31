import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing  import load_image, extract_patches, get_grid_shape
from src.hog_features   import compute_hog_features, compute_hog_single, get_feature_dim

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_PATH = "images/forest_path.jpg"
PATCH_SIZE  = 16
STRIDE      = 16
OUTPUT_DIR  = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Forest Path Segmentation — HOG Texture")
    print("=" * 50)

    # ── Step 2: Load image + extract patches ──────────────────────────────
    img_bgr, img_gray = load_image(IMAGE_PATH, max_width=600)
    h, w = img_gray.shape
    print(f"\n[Step 2] Image loaded  : {w} x {h} px")

    patches, positions = extract_patches(img_gray, PATCH_SIZE, STRIDE)
    n_rows, n_cols     = get_grid_shape(img_gray, PATCH_SIZE, STRIDE)
    print(f"[Step 2] Patch size    : {PATCH_SIZE} x {PATCH_SIZE} px")
    print(f"[Step 2] Grid          : {n_rows} rows x {n_cols} cols")
    print(f"[Step 2] Total patches : {len(patches)}")

    # Draw + save patch grid
    img_preview = img_bgr.copy()
    for (row, col) in positions:
        cv2.rectangle(
            img_preview,
            (col, row),
            (col + PATCH_SIZE, row + PATCH_SIZE),
            color=(0, 200, 0),
            thickness=1
        )
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step2_patch_grid.jpg"), img_preview)
    print(f"[Step 2] Saved → output/step2_patch_grid.jpg")

    # ── Step 3: HOG feature extraction ────────────────────────────────────
    print(f"\n[Step 3] Computing HOG features for {len(patches)} patches...")

    feature_matrix = compute_hog_features(patches)
    feat_dim       = get_feature_dim(PATCH_SIZE)

    print(f"[Step 3] Feature vector length : {feat_dim}")
    print(f"[Step 3] Feature matrix shape  : {feature_matrix.shape}")
    print(f"[Step 3]   → {feature_matrix.shape[0]} patches × {feature_matrix.shape[1]} features")

    # ── Step 3 visualisation: 8 sample patches + HOG maps ─────────────────
    sample_indices = np.linspace(0, len(patches) - 1, 8, dtype=int)

    fig, axes = plt.subplots(2, 8, figsize=(18, 5))
    fig.suptitle(
        f"Step 3 — HOG Feature Extraction  |  patch {PATCH_SIZE}×{PATCH_SIZE} px  |  "
        f"feature dim = {feat_dim}",
        fontsize=13, fontweight="bold"
    )

    for i, idx in enumerate(sample_indices):
        patch = patches[idx]
        fd, hog_img = compute_hog_single(patch)

        # Top row: original patch
        axes[0, i].imshow(patch, cmap="gray")
        axes[0, i].set_title(f"patch #{idx}", fontsize=8)
        axes[0, i].axis("off")

        # Bottom row: HOG gradient map
        axes[1, i].imshow(hog_img, cmap="magma")
        axes[1, i].set_title(f"HOG", fontsize=8)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("HOG map", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step3_hog_samples.png"), dpi=150)
    plt.close()

    # ── Step 3 visualisation: feature value distribution ──────────────────
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.hist(feature_matrix.flatten(), bins=60, color="#4A90D9", edgecolor="white", linewidth=0.4)
    ax.set_title("Step 3 — HOG Feature Value Distribution (all patches)", fontsize=13)
    ax.set_xlabel("HOG feature value")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step3_feature_distribution.png"), dpi=150)
    plt.close()

    print(f"[Step 3] Saved → output/step3_hog_samples.png")
    print(f"[Step 3] Saved → output/step3_feature_distribution.png")
    print("\nStep 3 complete ✓  Ready for Step 4: KMeans clustering.")


if __name__ == "__main__":
    main()