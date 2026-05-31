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
from src.segmentation   import (
    run_kmeans, build_label_map, labels_to_color_map, CLUSTER_NAMES, CLUSTER_COLORS
)
from src.visualization  import overlay_segmentation, save_segmentation_figure

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_PATH  = "images/forest_path.jpg"
PATCH_SIZE  = 16
STRIDE      = 16
N_CLUSTERS  = 3
OUTPUT_DIR  = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 55)
    print("  Forest Path Segmentation — HOG + KMeans")
    print("=" * 55)

    # ── Step 2: Load + patch extraction ──────────────────────────────────
    img_bgr, img_gray = load_image(IMAGE_PATH, max_width=600)
    h, w = img_gray.shape
    print(f"\n[Step 2] Image loaded      : {w} x {h} px")

    patches, positions = extract_patches(img_gray, PATCH_SIZE, STRIDE)
    n_rows, n_cols     = get_grid_shape(img_gray, PATCH_SIZE, STRIDE)
    print(f"[Step 2] Grid              : {n_rows} rows x {n_cols} cols")
    print(f"[Step 2] Total patches     : {len(patches)}")

    # Draw + save patch grid
    img_preview = img_bgr.copy()
    for (row, col) in positions:
        cv2.rectangle(img_preview, (col, row),
                      (col + PATCH_SIZE, row + PATCH_SIZE),
                      (0, 200, 0), 1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step2_patch_grid.jpg"), img_preview)

    # ── Step 3: HOG features ──────────────────────────────────────────────
    print(f"\n[Step 3] Extracting HOG features...")
    feature_matrix = compute_hog_features(patches)
    feat_dim       = get_feature_dim(PATCH_SIZE)
    print(f"[Step 3] Feature matrix    : {feature_matrix.shape}  "
          f"({feature_matrix.shape[0]} patches × {feat_dim} features)")

    # Save HOG sample visualisation
    sample_indices = np.linspace(0, len(patches) - 1, 8, dtype=int)
    fig, axes = plt.subplots(2, 8, figsize=(18, 5))
    fig.suptitle(
        f"Step 3 — HOG Feature Extraction  |  {PATCH_SIZE}×{PATCH_SIZE} px  |  dim={feat_dim}",
        fontsize=12, fontweight="bold"
    )
    for i, idx in enumerate(sample_indices):
        fd, hog_img = compute_hog_single(patches[idx])
        axes[0, i].imshow(patches[idx], cmap="gray")
        axes[0, i].set_title(f"#{idx}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(hog_img, cmap="magma")
        axes[1, i].set_title("HOG", fontsize=8)
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step3_hog_samples.png"), dpi=150)
    plt.close()

    # ── Step 4: KMeans clustering ─────────────────────────────────────────
    print(f"\n[Step 4] Running KMeans  (k={N_CLUSTERS})...")
    labels, kmeans, scaler = run_kmeans(feature_matrix, n_clusters=N_CLUSTERS)

    print(f"[Step 4] Clustering done.")
    for i in range(N_CLUSTERS):
        count = int(np.sum(labels == i))
        pct   = count / len(labels) * 100
        print(f"         {CLUSTER_NAMES[i]}  →  {count} patches  ({pct:.1f}%)")

    # Build label map + color map
    label_map = build_label_map(labels, n_rows, n_cols)
    color_map = labels_to_color_map(label_map, PATCH_SIZE, (h, w))
    overlay   = overlay_segmentation(img_bgr, color_map, alpha=0.45)

    # Save results
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step4_color_map.jpg"),
                cv2.cvtColor(color_map, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step4_overlay.jpg"), overlay)

    save_segmentation_figure(
        img_bgr, color_map, overlay, labels,
        N_CLUSTERS,
        os.path.join(OUTPUT_DIR, "step4_segmentation.png"),
    )

    print("\nStep 4 complete ✓  Ready for Step 5: visualization polish.")


if __name__ == "__main__":
    main()