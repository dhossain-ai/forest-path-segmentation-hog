import numpy as np
from skimage.feature import hog
from skimage import exposure
import cv2

# ── HOG config ────────────────────────────────────────────────────────────────
HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (1, 1)


def compute_hog_single(patch: np.ndarray):
    """HOG descriptor + visualisation for ONE grayscale patch."""
    fd, hog_image = hog(
        patch,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=True,
        feature_vector=True,
    )
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return fd, hog_image


def compute_hog_features(patches: list) -> np.ndarray:
    """HOG descriptors for all grayscale patches → (n_patches, hog_dim)."""
    features = []
    for patch in patches:
        fd, _ = hog(
            patch,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            visualize=True,
            feature_vector=True,
        )
        features.append(fd)
    return np.array(features)


def compute_color_features(patches_bgr: list) -> np.ndarray:
    """
    Compute HSV color statistics per patch.
    Returns (n_patches, 6): [mean_H, mean_S, mean_V, std_H, std_S, std_V]

    Why HSV?
      - H (Hue)        separates brown path vs green grass vs dark trees
      - S (Saturation) separates vivid grass vs washed-out sky
      - V (Value)      separates bright patches from dark canopy
    """
    features = []
    for patch_bgr in patches_bgr:
        hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        means = hsv.mean(axis=(0, 1))   # [mean_H, mean_S, mean_V]
        stds  = hsv.std(axis=(0, 1))    # [std_H,  std_S,  std_V]
        features.append(np.concatenate([means, stds]))
    return np.array(features)


def extract_bgr_patches(img_bgr: np.ndarray, patch_size: int, stride: int) -> list:
    """Extract color (BGR) patches — parallel to grayscale patch list."""
    patches = []
    h, w    = img_bgr.shape[:2]
    for row in range(0, h - patch_size + 1, stride):
        for col in range(0, w - patch_size + 1, stride):
            patches.append(img_bgr[row:row + patch_size, col:col + patch_size])
    return patches


def get_feature_dim(patch_size: int = 16) -> int:
    """HOG feature vector length for given patch size."""
    cells = patch_size // HOG_PIXELS_PER_CELL[0]
    return HOG_ORIENTATIONS * cells * cells