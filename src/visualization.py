import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.ndimage import median_filter

# ── Cluster visual identity ───────────────────────────────────────────────────
CLUSTER_COLORS = [
    [139,  90,  43],   # cluster 0 — brown       (path)
    [ 60, 179,  60],   # cluster 1 — green        (grass)
    [ 47,  79, 159],   # cluster 2 — dark blue    (trees/canopy)
]
CLUSTER_NAMES = ["Path", "Grass", "Trees"]


def normalize_features(features: np.ndarray):
    """StandardScaler — zero mean, unit variance per feature."""
    scaler = StandardScaler()
    return scaler.fit_transform(features), scaler


def run_kmeans(features: np.ndarray, n_clusters: int = 3, random_state: int = 42):
    """
    Normalize + KMeans cluster HOG+color features.

    Returns:
        labels : (n_patches,) cluster ids
        kmeans : fitted KMeans
        scaler : fitted StandardScaler
    """
    features_scaled, scaler = normalize_features(features)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=15,
        max_iter=500,
    )
    labels = kmeans.fit_predict(features_scaled)
    return labels, kmeans, scaler


def smooth_label_map(label_map: np.ndarray, size: int = 5) -> np.ndarray:
    """
    Apply median filter to 2D label map to remove noisy isolated patches.

    Args:
        label_map : (n_rows, n_cols) cluster labels
        size      : filter kernel size — larger = smoother regions

    Returns:
        smoothed label map (same shape)
    """
    return median_filter(label_map, size=size).astype(np.int32)


def build_label_map(labels: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Reshape flat labels → 2D grid (n_rows × n_cols)."""
    return labels[:n_rows * n_cols].reshape(n_rows, n_cols)


def labels_to_color_map(
    label_map: np.ndarray,
    patch_size: int,
    img_shape: tuple,
) -> np.ndarray:
    """
    Convert 2D label grid → full-resolution RGB segmentation image.

    Returns:
        color_map : (H, W, 3) uint8 RGB image
    """
    H, W           = img_shape
    n_rows, n_cols = label_map.shape
    color_map      = np.zeros((H, W, 3), dtype=np.uint8)

    for r in range(n_rows):
        for c in range(n_cols):
            label  = int(label_map[r, c])
            r0, r1 = r * patch_size, min((r + 1) * patch_size, H)
            c0, c1 = c * patch_size, min((c + 1) * patch_size, W)
            color_map[r0:r1, c0:c1] = CLUSTER_COLORS[label % len(CLUSTER_COLORS)]

    return color_map