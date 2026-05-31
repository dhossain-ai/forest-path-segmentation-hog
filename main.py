import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing import load_image, extract_patches, get_grid_shape
from src.hog_features  import (
    compute_hog_features, compute_hog_single,
    compute_color_features, extract_bgr_patches, get_feature_dim,
)
from src.segmentation  import (
    run_kmeans, auto_remap_clusters,
    build_label_map, smooth_label_map,
    labels_to_color_map, CLUSTER_NAMES, CLUSTER_COLORS,
)
from src.visualization import overlay_segmentation, save_segmentation_figure

# ── Config ────────────────────────────────────────────────────────────────────
IMAGE_PATH  = "images/forest_path.jpg"
PATCH_SIZE  = 16
STRIDE      = 16
N_CLUSTERS  = 3
SMOOTH_SIZE = 5
OUTPUT_DIR  = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 55)
    print("  Forest Path Segmentation — HOG + Color + KMeans")
    print("=" * 55)

    # ── Step 2: Load + patch extraction ──────────────────────────────────
    img_bgr, img_gray = load_image(IMAGE_PATH, max_width=600)
    h, w = img_gray.shape
    print(f"\n[Step 2] Image loaded      : {w} x {h} px")

    patches_gray, positions = extract_patches(img_gray, PATCH_SIZE, STRIDE)
    patches_bgr             = extract_bgr_patches(img_bgr, PATCH_SIZE, STRIDE)
    n_rows, n_cols          = get_grid_shape(img_gray, PATCH_SIZE, STRIDE)

    print(f"[Step 2] Grid              : {n_rows} rows x {n_cols} cols")
    print(f"[Step 2] Total patches     : {len(patches_gray)}")

    img_preview = img_bgr.copy()
    for (row, col) in positions:
        cv2.rectangle(img_preview, (col, row),
                      (col + PATCH_SIZE, row + PATCH_SIZE), (0, 200, 0), 1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step2_patch_grid.jpg"), img_preview)

    # ── Step 3: HOG + Color features ─────────────────────────────────────
    print(f"\n[Step 3] Extracting HOG features...")
    hog_features   = compute_hog_features(patches_gray)

    print(f"[Step 3] Extracting color (HSV) features...")
    color_features = compute_color_features(patches_bgr)  # raw, unscaled

    feature_matrix = np.hstack([hog_features, color_features])
    hog_dim        = get_feature_dim(PATCH_SIZE)

    print(f"[Step 3] HOG dim           : {hog_dim}")
    print(f"[Step 3] Color dim         : {color_features.shape[1]}")
    print(f"[Step 3] Combined dim      : {feature_matrix.shape[1]}")
    print(f"[Step 3] Feature matrix    : {feature_matrix.shape}")

    # HOG sample visualisation
    sample_indices = np.linspace(0, len(patches_gray) - 1, 8, dtype=int)
    fig, axes = plt.subplots(2, 8, figsize=(18, 5))
    fig.suptitle(
        f"Step 3 — HOG Extraction  |  {PATCH_SIZE}×{PATCH_SIZE} px  |  "
        f"HOG={hog_dim}  Color=6  Total={feature_matrix.shape[1]}",
        fontsize=11, fontweight="bold"
    )
    for i, idx in enumerate(sample_indices):
        fd, hog_img = compute_hog_single(patches_gray[idx])
        axes[0, i].imshow(patches_gray[idx], cmap="gray")
        axes[0, i].set_title(f"#{idx}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(hog_img, cmap="magma")
        axes[1, i].set_title("HOG", fontsize=8)
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step3_hog_samples.png"), dpi=150)
    plt.close()

    # ── Step 4: KMeans + auto semantic remapping ──────────────────────────
    print(f"\n[Step 4] Running KMeans  (k={N_CLUSTERS})...")
    labels_raw, kmeans, scaler = run_kmeans(feature_matrix, n_clusters=N_CLUSTERS)

    print(f"[Step 4] Raw cluster distribution:")
    for i in range(N_CLUSTERS):
        count = int(np.sum(labels_raw == i))
        print(f"         Cluster {i}  →  {count:4d} patches  ({count/len(labels_raw)*100:.1f}%)")

    # ── Auto-remap to semantic labels ─────────────────────────────────────
    print(f"\n[Step 4] Auto-remapping clusters to semantic labels...")
    labels = auto_remap_clusters(labels_raw, color_features, n_clusters=N_CLUSTERS)

    print(f"[Step 4] Semantic distribution (after remap):")
    for i in range(N_CLUSTERS):
        count = int(np.sum(labels == i))
        pct   = count / len(labels) * 100
        print(f"         {CLUSTER_NAMES[i]:10s}  →  {count:4d} patches  ({pct:.1f}%)")

    label_map     = build_label_map(labels, n_rows, n_cols)
    color_map_raw = labels_to_color_map(label_map, PATCH_SIZE, (h, w))

    # ── Step 5: Smoothing + final visualization ───────────────────────────
    print(f"\n[Step 5] Applying spatial smoothing (median {SMOOTH_SIZE}×{SMOOTH_SIZE})...")
    label_map_smooth   = smooth_label_map(label_map, size=SMOOTH_SIZE)
    labels_smooth_flat = label_map_smooth.flatten()

    print(f"[Step 5] Smoothed distribution:")
    for i in range(N_CLUSTERS):
        count = int(np.sum(labels_smooth_flat == i))
        pct   = count / len(labels_smooth_flat) * 100
        print(f"         {CLUSTER_NAMES[i]:10s}  →  {count:4d} patches  ({pct:.1f}%)")

    color_map_smooth = labels_to_color_map(label_map_smooth, PATCH_SIZE, (h, w))
    overlay          = overlay_segmentation(img_bgr, color_map_smooth, alpha=0.45)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "step5_color_map_smooth.jpg"),
                cv2.cvtColor(color_map_smooth, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step5_overlay.jpg"), overlay)

    save_segmentation_figure(
        img_bgr,
        color_map_raw,
        color_map_smooth,
        overlay,
        labels,
        labels_smooth_flat,
        N_CLUSTERS,
        os.path.join(OUTPUT_DIR, "step5_final_segmentation.png"),
    )

    print("\n" + "=" * 55)
    print("  ALL STEPS COMPLETE ✓")
    print("  Check output/step5_final_segmentation.png")
    print("=" * 55)


if __name__ == "__main__":
    main()