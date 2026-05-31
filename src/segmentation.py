import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.ndimage import median_filter

# ── Cluster visual identity ───────────────────────────────────────────────────
CLUSTER_COLORS = [
    [139,  90,  43],   # 0 — brown  (Path)
    [ 60, 179,  60],   # 1 — green  (Grass)
    [ 47,  79, 159],   # 2 — blue   (Trees/Canopy)
]
CLUSTER_NAMES = ["Path", "Grass", "Trees"]


def normalize_features(features: np.ndarray):
    """StandardScaler — zero mean, unit variance per feature."""
    scaler = StandardScaler()
    return scaler.fit_transform(features), scaler


def run_kmeans(features: np.ndarray, n_clusters: int = 3, random_state: int = 42):
    """Normalize + KMeans cluster combined HOG+color features."""
    features_scaled, scaler = normalize_features(features)
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=15,
        max_iter=500,
    )
    labels = kmeans.fit_predict(features_scaled)
    return labels, kmeans, scaler


def auto_remap_clusters(
    labels: np.ndarray,
    color_features_raw: np.ndarray,
    n_clusters: int = 3,
) -> np.ndarray:
    """
    Remap KMeans cluster IDs → correct semantic labels:
        0 = Path  (brownish — low H in OpenCV HSV 0-180 scale)
        1 = Grass (greenish — medium-high H, brighter)
        2 = Trees (dark canopy — lowest mean V/brightness)

    Strategy:
        Step 1 → darkest cluster (lowest mean V) → Trees
        Step 2 → among remaining: lowest mean H (warmest/brownish) → Path
        Step 3 → remaining → Grass

    Args:
        labels            : raw KMeans labels (n_patches,)
        color_features_raw: unscaled HSV features (n_patches, 6)
                            cols: [mean_H, mean_S, mean_V, std_H, std_S, std_V]
    Returns:
        new_labels : semantically corrected label array
    """
    stats = {}
    for i in range(n_clusters):
        mask     = labels == i
        stats[i] = {
            'H': float(color_features_raw[mask, 0].mean()),
            'S': float(color_features_raw[mask, 1].mean()),
            'V': float(color_features_raw[mask, 2].mean()),
        }

    print("\n    [Auto-remap] Raw cluster HSV centroids:")
    for i, s in stats.items():
        print(f"      Cluster {i}: H={s['H']:5.1f}  "
              f"S={s['S']:5.1f}  V={s['V']:5.1f}")

    # Step 1 — darkest (lowest V) → Trees
    trees_id  = min(stats, key=lambda i: stats[i]['V'])

    # Step 2 — among remaining, most brownish (lowest H) → Path
    remaining = [c for c in range(n_clusters) if c != trees_id]
    path_id   = min(remaining, key=lambda i: stats[i]['H'])

    # Step 3 — leftover → Grass
    grass_id  = [c for c in remaining if c != path_id][0]

    remap = {path_id: 0, grass_id: 1, trees_id: 2}
    print(f"    [Auto-remap] Cluster {path_id} → Path  |  "
          f"Cluster {grass_id} → Grass  |  "
          f"Cluster {trees_id} → Trees\n")

    return np.vectorize(remap.get)(labels)


def smooth_label_map(label_map: np.ndarray, size: int = 5) -> np.ndarray:
    """Median filter on 2D label grid to remove noisy isolated patches."""
    return median_filter(label_map, size=size).astype(np.int32)


def build_label_map(labels: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Reshape flat labels → 2D grid (n_rows × n_cols)."""
    return labels[:n_rows * n_cols].reshape(n_rows, n_cols)


def labels_to_color_map(
    label_map: np.ndarray,
    patch_size: int,
    img_shape: tuple,
) -> np.ndarray:
    """Convert 2D label grid → full-resolution RGB segmentation image."""
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