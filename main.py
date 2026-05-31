import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.segmentation import CLUSTER_COLORS, CLUSTER_NAMES


def overlay_segmentation(
    img_bgr: np.ndarray,
    color_map_rgb: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend RGB color_map over original BGR image.

    Returns:
        overlay : BGR image with segmentation blended in
    """
    color_map_bgr = cv2.cvtColor(color_map_rgb, cv2.COLOR_RGB2BGR)
    cm_h, cm_w    = color_map_bgr.shape[:2]
    overlay       = img_bgr.copy()

    overlay[:cm_h, :cm_w] = cv2.addWeighted(
        img_bgr[:cm_h, :cm_w], 1 - alpha,
        color_map_bgr,          alpha,
        0,
    )
    return overlay


def save_segmentation_figure(
    img_bgr: np.ndarray,
    color_map: np.ndarray,
    overlay: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    output_path: str,
) -> None:
    """Save 3-panel figure: Original | Segmentation Map | Overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    fig.patch.set_facecolor("#1e1e1e")

    titles = [
        "Original Image",
        f"KMeans Segmentation  |  {n_clusters} clusters",
        "Blended Overlay",
    ]
    images = [
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        color_map,
        cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
    ]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color="white", pad=10)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")

    # Legend: show cluster + patch count
    legend_patches = []
    for i in range(n_clusters):
        count      = int(np.sum(labels == i))
        pct        = count / len(labels) * 100
        color_norm = [c / 255 for c in CLUSTER_COLORS[i]]
        legend_patches.append(
            mpatches.Patch(
                color=color_norm,
                label=f"{CLUSTER_NAMES[i]}  —  {count} patches ({pct:.1f}%)",
            )
        )

    axes[1].legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=n_clusters,
        fontsize=10,
        framealpha=0.85,
        facecolor="#2a2a2a",
        labelcolor="white",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Step 4] Saved → {output_path}")