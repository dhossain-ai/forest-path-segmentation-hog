import numpy as np
from skimage.feature import hog
from skimage import exposure


# ── HOG config (shared across the project) ───────────────────────────────────
HOG_ORIENTATIONS    = 9       # number of gradient direction bins
HOG_PIXELS_PER_CELL = (8, 8)  # cell size inside each patch
HOG_CELLS_PER_BLOCK = (1, 1)  # block normalisation window


def compute_hog_single(patch: np.ndarray):
    """
    Compute HOG descriptor + visualisation image for ONE patch.

    Returns:
        fd        : 1-D feature vector (length = feature_dim)
        hog_image : gradient visualisation array (same size as patch)
    """
    fd, hog_image = hog(
        patch,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        visualize=True,
        feature_vector=True,
    )
    # Rescale for better visibility
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return fd, hog_image


def compute_hog_features(patches: list) -> np.ndarray:
    """
    Compute HOG descriptors for ALL patches.

    Args:
        patches : list of (patch_size x patch_size) grayscale arrays

    Returns:
        features : np.ndarray of shape (n_patches, feature_dim)
    """
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


def get_feature_dim(patch_size: int = 16) -> int:
    """
    Returns the HOG feature vector length for a given patch size.
    Formula: orientations * (patch_size / pixels_per_cell) ^ 2
    """
    cells = patch_size // HOG_PIXELS_PER_CELL[0]
    return HOG_ORIENTATIONS * cells * cells