import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ── Cluster visual identity ───────────────────────────────────────────────────
CLUSTER_COLORS = [
    [205,  92,  92],   # cluster 0 — indian red   (likely: path)
    [ 60, 179,  60],   # cluster 1 — medium green (likely: grass)
    [ 47,  79, 159],   # cluster 2 — steel blue   (likely: trees/canopy)
]

CLUSTER_NAMES = ["Cluster 0", "Cluster 1", "Cluster 2"]


def normalize_features(features: np.ndarray):
    """StandardScaler normalization — zero mean, unit variance per feature."""
    scaler = StandardScaler()
    return scaler.fit_transform(features), scaler


def run_kmeans(features: np.ndarray, n_clusters: int = 3, random_state: int = 42):
    """
    Normalize features then apply KMeans clustering.

    Returns:
        labels  : (n_patches,) int array of cluster ids
        kmeans  : fitted KMeans object
        scaler  : fitted StandardScaler
    """
    features_scaled, scaler = normalize_features(features)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(features_scaled)
    return labels, kmeans, scaler


def build_label_map(labels: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Reshape flat label array → 2D grid (n_rows × n_cols)."""
    return labels[: n_rows * n_cols].reshape(n_rows, n_cols)


def labels_to_color_map(
    label_map: np.ndarray,
    patch_size: int,
    img_shape: tuple,
) -> np.ndarray:
    """
    Convert 2D label grid → full-resolution RGB color segmentation image.

    Args:
        label_map  : (n_rows, n_cols) cluster labels
        patch_size : pixels per patch side
        img_shape  : (H, W) of original image

    Returns:
        color_map : (H, W, 3) uint8 RGB image
    """
    H, W       = img_shape
    n_rows, n_cols = label_map.shape
    color_map  = np.zeros((H, W, 3), dtype=np.uint8)

    for r in range(n_rows):
        for c in range(n_cols):
            label  = int(label_map[r, c])
            r0, r1 = r * patch_size, min((r + 1) * patch_size, H)
            c0, c1 = c * patch_size, min((c + 1) * patch_size, W)
            color_map[r0:r1, c0:c1] = CLUSTER_COLORS[label % len(CLUSTER_COLORS)]

    return color_map