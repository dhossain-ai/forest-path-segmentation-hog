import os
import cv2
import matplotlib.pyplot as plt

from src.preprocessing import load_image, extract_patches, get_grid_shape

# ── Config ───────────────────────────────────────────────────────────────────
IMAGE_PATH = "images/forest_path.jpg"
PATCH_SIZE  = 16   # pixels per patch side
STRIDE      = 16   # non-overlapping patches
OUTPUT_DIR  = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Forest Path Segmentation — HOG Texture")
    print("=" * 50)

    # ── Load image ──────────────────────────────────────────────────────────
    img_bgr, img_gray = load_image(IMAGE_PATH, max_width=600)
    h, w = img_gray.shape
    print(f"\n[Step 2] Image loaded  : {w} x {h} px")

    # ── Extract patches ─────────────────────────────────────────────────────
    patches, positions = extract_patches(img_gray, PATCH_SIZE, STRIDE)
    n_rows, n_cols     = get_grid_shape(img_gray, PATCH_SIZE, STRIDE)

    print(f"[Step 2] Patch size    : {PATCH_SIZE} x {PATCH_SIZE} px")
    print(f"[Step 2] Grid          : {n_rows} rows × {n_cols} cols")
    print(f"[Step 2] Total patches : {len(patches)}")

    # ── Draw patch grid overlay ─────────────────────────────────────────────
    img_preview = img_bgr.copy()
    for (row, col) in positions:
        cv2.rectangle(
            img_preview,
            (col, row),
            (col + PATCH_SIZE, row + PATCH_SIZE),
            color=(0, 200, 0),
            thickness=1
        )

    # ── Save + display ──────────────────────────────────────────────────────
    cv2.imwrite(os.path.join(OUTPUT_DIR, "step2_patch_grid.jpg"), img_preview)

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))

    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image (resized)", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(img_preview, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"Patch Grid  |  {PATCH_SIZE}×{PATCH_SIZE} px  |  {len(patches)} patches",
        fontsize=13
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step2_preview.png"), dpi=150)
    plt.show()
    print(f"\n[Step 2] Saved → output/step2_preview.png")
    print("\nStep 2 complete ✓  Ready for Step 3: HOG feature extraction.")

if __name__ == "__main__":
    main()